#!/usr/bin/env python3
from flask import Flask, Response, render_template_string, request, jsonify
import threading
import time
import socket

import cv2
import numpy as np

# =========================
# Camera config
# =========================
DEVICE = "/dev/video4"   # change to /dev/video0 if needed
WIDTH = 640
HEIGHT = 480
FPS = 15
PORT = 7072
JPEG_QUALITY = 80

app = Flask(__name__)

# =========================
# Shared camera buffers
# =========================
_latest_lock = threading.Lock()
_latest_bgr = None
_latest_jpeg = None
_latest_gray_jpeg = None
_latest_seq = 0
_cam_stop = threading.Event()

def detect_local_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    ips.discard("127.0.0.1")
    return sorted(ips)

def open_camera():
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(DEVICE)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera device: {DEVICE}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    return cap

def camera_worker():
    global _latest_bgr, _latest_jpeg, _latest_gray_jpeg, _latest_seq
    cap = open_camera()
    try:
        while not _cam_stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
                frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

            # Encode COLOR JPEG
            ok2, jpg = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)],
            )
            if not ok2:
                continue

            # Encode GRAYSCALE JPEG
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ok3, gray_jpg = cv2.imencode(
                ".jpg",
                gray,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)],
            )
            if not ok3:
                continue

            with _latest_lock:
                _latest_bgr = frame
                _latest_jpeg = jpg.tobytes()
                _latest_gray_jpeg = gray_jpg.tobytes()
                _latest_seq += 1

            time.sleep(max(0.0, 1.0 / float(FPS)))
    finally:
        try:
            cap.release()
        except Exception:
            pass

def multipart_mjpeg_generator(which="color"):
    boundary = b"--frame\r\n"
    last_seq = -1
    while True:
        with _latest_lock:
            seq = _latest_seq
            if which == "gray":
                jpg = _latest_gray_jpeg
            else:
                jpg = _latest_jpeg

        if jpg is None or seq == last_seq:
            time.sleep(0.005)
            continue

        last_seq = seq
        headers = (
            boundary +
            b"Content-Type: image/jpeg\r\n" +
            f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
        )
        yield headers + jpg + b"\r\n"

def get_latest_bgr():
    with _latest_lock:
        if _latest_bgr is None:
            return None
        return _latest_bgr.copy()

def bgr_to_hex(bgr):
    b = int(bgr[0])
    g = int(bgr[1])
    r = int(bgr[2])
    return f"#{r:02x}{g:02x}{b:02x}", [r, g, b], [b, g, r]

def gray_from_bgr(bgr):
    px = np.uint8([[bgr]])
    g = cv2.cvtColor(px, cv2.COLOR_BGR2GRAY)[0, 0]
    return int(g)

def gray_to_hex(gray):
    gray = int(max(0, min(255, int(gray))))
    return f"#{gray:02x}{gray:02x}{gray:02x}", [gray, gray, gray]

HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no" />
  <title>Grayscale Dual View Lab</title>
  <style>
    :root{
      --bg:#0b0f14;
      --panel:rgba(255,255,255,0.08);
      --stroke:rgba(255,255,255,0.14);
      --text:#e8eef7;
      --muted:rgba(232,238,247,0.75);
    }
    *{box-sizing:border-box;-webkit-tap-highlight-color:transparent;}
    html,body{height:100%;margin:0;background:var(--bg);color:var(--text);
      font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;}
    .wrap{height:100%;display:grid;grid-template-rows:auto 1fr;}
    header{padding:10px 12px;border-bottom:1px solid var(--stroke);
      display:flex;justify-content:space-between;align-items:center;gap:12px;
      background:linear-gradient(to bottom, rgba(255,255,255,0.06), rgba(255,255,255,0.00));}
    .meta{font-size:14px;color:var(--muted);line-height:1.3;}
    .status{font-size:14px;padding:8px 12px;border:1px solid var(--stroke);
      border-radius:999px;background:var(--panel);}

    .main{position:relative;overflow:hidden;}

    /* Two video panes */
    .videoGrid{
      position:absolute;
      inset:0;
      display:grid;
      grid-template-columns: 1fr 1fr;
      gap:2px;
      background:#000;
    }
    .pane{
      position:relative;
      overflow:hidden;
      background:#000;
      display:grid;
      place-items:center;
    }
    .pane img{
      width:100%;
      height:100%;
      object-fit:contain;
      background:#000;
      touch-action:manipulation;
    }
    .paneLabel{
      position:absolute;
      left:10px;
      top:10px;
      z-index:2;
      font-size:16px;
      font-weight:900;
      padding:6px 10px;
      border-radius:10px;
      border:1px solid var(--stroke);
      background:rgba(0,0,0,0.45);
    }
    @media (orientation: portrait){
      .videoGrid{ grid-template-columns: 1fr; grid-template-rows: 1fr 1fr; }
    }

    /* Overlay panel */
    .panel{
      position:absolute;left:12px;top:12px;right:12px;
      display:grid;grid-template-columns: 1.2fr 0.8fr;
      gap:12px;
      pointer-events:none;
    }
    .card{
      pointer-events:auto;
      border:1px solid var(--stroke);
      background:rgba(0,0,0,0.45);
      border-radius:14px;
      padding:12px;
      backdrop-filter: blur(6px);
    }

    .bigLine{font-size:30px;font-weight:900;letter-spacing:0.2px;}
    .bigSub{font-size:20px;font-weight:800;color:var(--muted);margin-top:6px;}
    .small{font-size:14px;color:var(--muted);margin-top:8px;line-height:1.35;}
    .mono{font-variant-numeric: tabular-nums;}

    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;}
    label{display:block;font-size:14px;color:var(--muted);margin-bottom:6px;}
    input{
      width:100%;
      font-size:22px;
      padding:10px 10px;
      border-radius:10px;
      border:1px solid var(--stroke);
      background:rgba(255,255,255,0.08);
      color:var(--text);
      outline:none;
    }
    button{
      width:100%;
      font-size:20px;
      font-weight:900;
      padding:12px 10px;
      border-radius:12px;
      border:1px solid var(--stroke);
      background:rgba(255,255,255,0.10);
      color:var(--text);
    }
    button:active{transform:scale(0.99);}

    .swatch{
      height:120px;
      border-radius:14px;
      border:1px solid var(--stroke);
      background:#222;
      margin-top:10px;
    }
    .tapHint{
      margin-top:10px;
      padding:10px 12px;
      border-radius:12px;
      border:1px solid var(--stroke);
      background:rgba(255,255,255,0.06);
      font-size:18px;
      font-weight:800;
    }
  </style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <div style="font-size:18px;font-weight:900;">Grayscale Dual View Lab</div>
      <div class="meta">
        Tap either view to read grayscale intensity at that pixel. Enter a gray value to preview the colour.<br/>
        Grayscale range: 0 (black) → 255 (white).
      </div>
    </div>
    <div id="status" class="status">Connecting…</div>
  </header>

  <div class="main">
    <div class="videoGrid">
      <div class="pane">
        <div class="paneLabel">Original (Colour)</div>
        <img id="streamColor" src="/mjpg" alt="colour stream" />
      </div>
      <div class="pane">
        <div class="paneLabel">After Grayscale</div>
        <img id="streamGray" src="/mjpg_gray" alt="grayscale stream" />
      </div>
    </div>

    <div class="panel">
      <div class="card">
        <div class="bigLine mono" id="tapGray">Tapped Gray: —</div>
        <div class="bigSub mono" id="tapExtra">RGB: —   HEX: —</div>
        <div class="tapHint">Tap bright vs shadow areas 👆</div>
        <div class="small">Different colours can have similar gray if their brightness is similar.</div>
      </div>

      <div class="card">
        <div style="font-size:18px;font-weight:900;">Gray → Colour</div>
        <div class="grid2">
          <div>
            <label for="inG">Gray (0–255)</label>
            <input id="inG" type="number" min="0" max="255" value="128"/>
          </div>
          <div style="display:grid;align-content:end;">
            <button id="btnShow">Show Grey</button>
          </div>
        </div>

        <div class="swatch" id="swatch"></div>
        <div class="bigSub mono" id="outText">RGB: —   HEX: —</div>
        <div class="small">Grayscale colour has equal channels: R=G=B=Gray.</div>
      </div>
    </div>
  </div>
</div>

<script>
(() => {
  const statusEl = document.getElementById("status");

  const imgColor = document.getElementById("streamColor");
  const imgGray  = document.getElementById("streamGray");

  const tapGrayEl = document.getElementById("tapGray");
  const tapExtraEl = document.getElementById("tapExtra");

  const inG = document.getElementById("inG");
  const btnShow = document.getElementById("btnShow");
  const swatch = document.getElementById("swatch");
  const outText = document.getElementById("outText");

  async function apiPost(url, body){
    const r = await fetch(url, {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(body || {}),
      cache:"no-store",
    });
    return r.json();
  }

  async function ping(){
    try{
      await apiPost("/api/gray_to_color", {g: 0});
      statusEl.textContent = "Connected";
      statusEl.style.opacity = "1.0";
    }catch(e){
      statusEl.textContent = "Disconnected…";
      statusEl.style.opacity = "0.7";
    }finally{
      setTimeout(ping, 800);
    }
  }
  ping();

  async function handleTapOnImage(imgEl, e){
    e.preventDefault();

    const rect = imgEl.getBoundingClientRect();
    const x = Math.round((e.clientX - rect.left) / rect.width * {{w}});
    const y = Math.round((e.clientY - rect.top) / rect.height * {{h}});

    try{
      const js = await apiPost("/api/pick_pixel", {x, y});
      if (!js || !js.ok) return;

      tapGrayEl.textContent = `Tapped Gray: ${js.gray}`;
      tapExtraEl.textContent = `RGB: ${js.rgb[0]}, ${js.rgb[1]}, ${js.rgb[2]}   HEX: ${js.hex}`;

      // Load into input and preview
      inG.value = js.gray;
      swatch.style.background = js.gray_hex;
      outText.textContent = `RGB: ${js.gray_rgb[0]}, ${js.gray_rgb[1]}, ${js.gray_rgb[2]}   HEX: ${js.gray_hex}`;
    }catch(err){
      // ignore
    }
  }

  imgColor.addEventListener("pointerdown", (e) => handleTapOnImage(imgColor, e), {passive:false});
  imgGray.addEventListener("pointerdown",  (e) => handleTapOnImage(imgGray, e),  {passive:false});

  btnShow.addEventListener("click", async (e) => {
    e.preventDefault();
    const g = parseInt(inG.value || "0", 10);
    const js = await apiPost("/api/gray_to_color", {g});
    if (!js || !js.ok) return;

    swatch.style.background = js.hex;
    outText.textContent = `RGB: ${js.rgb[0]}, ${js.rgb[1]}, ${js.rgb[2]}   HEX: ${js.hex}`;
  });
})();
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(HTML, w=WIDTH, h=HEIGHT)

@app.get("/mjpg")
def mjpg():
    resp = Response(
        multipart_mjpeg_generator("color"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp

@app.get("/mjpg_gray")
def mjpg_gray():
    resp = Response(
        multipart_mjpeg_generator("gray"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp

@app.post("/api/pick_pixel")
def api_pick_pixel():
    data = request.get_json(force=True, silent=True) or {}
    x = int(data.get("x", -1))
    y = int(data.get("y", -1))

    frame = get_latest_bgr()
    if frame is None:
        return jsonify(ok=False, msg="No frame yet."), 503

    x = max(0, min(WIDTH - 1, x))
    y = max(0, min(HEIGHT - 1, y))

    bgr = frame[y, x].astype(np.uint8)
    hexcol, rgb, bgr_list = bgr_to_hex(bgr)

    gray = gray_from_bgr(bgr)
    gray_hex, gray_rgb = gray_to_hex(gray)

    return jsonify(
        ok=True,
        x=x, y=y,
        gray=gray,
        rgb=rgb,
        bgr=bgr_list,
        hex=hexcol,
        gray_rgb=gray_rgb,
        gray_hex=gray_hex,
    )

@app.post("/api/gray_to_color")
def api_gray_to_color():
    data = request.get_json(force=True, silent=True) or {}
    g = data.get("g", 0)
    g = int(max(0, min(255, int(g))))

    hexcol, rgb = gray_to_hex(g)
    return jsonify(ok=True, gray=g, rgb=rgb, hex=hexcol)

def main():
    t = threading.Thread(target=camera_worker, daemon=True)
    t.start()
    ips = detect_local_ips()
    if ips:
        print("Open on student device:")
        for ip in ips:
            print(f"  http://{ip}:{PORT}/")
    else:
        print(f"Open: http://<this-pc-ip>:{PORT}/")

    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()
