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
PORT = 7071

JPEG_QUALITY = 80

app = Flask(__name__)

# =========================
# Shared camera buffers
# =========================
_latest_lock = threading.Lock()
_latest_bgr = None
_latest_jpeg = None
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
        # fallback backend
        cap = cv2.VideoCapture(DEVICE)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera device: {DEVICE}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Try to reduce latency (may be ignored by some drivers)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    return cap

def camera_worker():
    global _latest_bgr, _latest_jpeg, _latest_seq
    cap = open_camera()
    try:
        while not _cam_stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            # Ensure size
            if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
                frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

            ok2, jpg = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)],
            )
            if not ok2:
                continue

            with _latest_lock:
                _latest_bgr = frame
                _latest_jpeg = jpg.tobytes()
                _latest_seq += 1

            time.sleep(max(0.0, 1.0 / float(FPS)))
    finally:
        try:
            cap.release()
        except Exception:
            pass

def multipart_mjpeg_generator():
    boundary = b"--frame\r\n"
    last_seq = -1
    while True:
        with _latest_lock:
            seq = _latest_seq
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

def hsv_opencv_to_color(h, s, v):
    h = int(max(0, min(179, int(h))))
    s = int(max(0, min(255, int(s))))
    v = int(max(0, min(255, int(v))))

    hsv_px = np.uint8([[[h, s, v]]])
    bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0, 0]
    hexcol, rgb, bgr = bgr_to_hex(bgr_px)
    return hexcol, rgb, bgr

HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no" />
  <title>HSV Tap Lab</title>
  <style>
    :root{
      --bg:#0b0f14;
      --panel:rgba(255,255,255,0.08);
      --stroke:rgba(255,255,255,0.14);
      --text:#e8eef7;
      --muted:rgba(232,238,247,0.75);
      --accent:rgba(33,150,243,0.35);
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
    .video{position:absolute;inset:0;background:#000;display:grid;place-items:center;}
    .video img{width:100%;height:100%;object-fit:contain;background:#000;touch-action:manipulation;}

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

    /* BIG OUTPUT TEXT */
    .bigLine{font-size:28px;font-weight:900;letter-spacing:0.2px;}
    .bigSub{font-size:20px;font-weight:800;color:var(--muted);margin-top:6px;}
    .small{font-size:14px;color:var(--muted);margin-top:8px;line-height:1.35;}

    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;}
    label{display:block;font-size:14px;color:var(--muted);margin-bottom:6px;}
    input{
      width:100%;
      font-size:20px;
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
      background:rgba(33,150,243,0.25);
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
    .mono{font-variant-numeric: tabular-nums;}
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
      <div style="font-size:18px;font-weight:900;">HSV Tap Lab</div>
      <div class="meta">Tap the video to read HSV at that pixel. Enter HSV to preview the colour.<br/>
      OpenCV HSV ranges: H 0–179, S 0–255, V 0–255.</div>
    </div>
    <div id="status" class="status">Connecting…</div>
  </header>

  <div class="main">
    <div class="video">
      <img id="stream" src="/mjpg" alt="stream" />
    </div>

    <div class="panel">
      <div class="card">
        <div class="bigLine mono" id="tapHSV">Tapped HSV: —</div>
        <div class="bigSub mono" id="tapRGB">RGB: —   HEX: —</div>
        <div class="tapHint">Tap anywhere on the video 👆</div>
        <div class="small">Tip: Try tapping on shadow vs bright areas and watch how V changes.</div>
      </div>

      <div class="card">
        <div style="font-size:18px;font-weight:900;">HSV → Colour</div>
        <div class="grid2">
          <div>
            <label for="inH">H (0–179)</label>
            <input id="inH" type="number" min="0" max="179" value="30"/>
          </div>
          <div>
            <label for="inS">S (0–255)</label>
            <input id="inS" type="number" min="0" max="255" value="200"/>
          </div>
          <div>
            <label for="inV">V (0–255)</label>
            <input id="inV" type="number" min="0" max="255" value="200"/>
          </div>
          <div style="display:grid;align-content:end;">
            <button id="btnShow">Show Colour</button>
          </div>
        </div>

        <div class="swatch" id="swatch"></div>
        <div class="bigSub mono" id="outText">RGB: —   HEX: —</div>
        <div class="small">Try: S=0 makes grey (no colour). V=0 makes black.</div>
      </div>
    </div>
  </div>
</div>

<script>
(() => {
  const statusEl = document.getElementById("status");
  const img = document.getElementById("stream");

  const tapHSVEl = document.getElementById("tapHSV");
  const tapRGBEl = document.getElementById("tapRGB");

  const inH = document.getElementById("inH");
  const inS = document.getElementById("inS");
  const inV = document.getElementById("inV");
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

  // Basic connection indicator (pings by converting a sample HSV)
  async function ping(){
    try{
      await apiPost("/api/hsv_to_color", {h: 0, s: 0, v: 0});
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

  // Tap on image to read HSV
  img.addEventListener("pointerdown", async (e) => {
    e.preventDefault();

    const rect = img.getBoundingClientRect();
    const x = Math.round((e.clientX - rect.left) / rect.width * {{w}});
    const y = Math.round((e.clientY - rect.top) / rect.height * {{h}});

    try{
      const js = await apiPost("/api/pick_pixel", {x, y});
      if (!js || !js.ok) return;

      const hsv = js.hsv; // [H,S,V]
      const rgb = js.rgb; // [R,G,B]
      tapHSVEl.textContent = `Tapped HSV: H=${hsv[0]}  S=${hsv[1]}  V=${hsv[2]}`;
      tapRGBEl.textContent = `RGB: ${rgb[0]}, ${rgb[1]}, ${rgb[2]}   HEX: ${js.hex}`;

      // Also load into inputs (nice for learning)
      inH.value = hsv[0];
      inS.value = hsv[1];
      inV.value = hsv[2];

      // Preview that exact HSV as a colour
      swatch.style.background = js.hex;
      outText.textContent = `RGB: ${rgb[0]}, ${rgb[1]}, ${rgb[2]}   HEX: ${js.hex}`;
    }catch(err){
      // ignore
    }
  }, {passive:false});

  // HSV -> colour button
  btnShow.addEventListener("click", async (e) => {
    e.preventDefault();
    const h = parseInt(inH.value || "0", 10);
    const s = parseInt(inS.value || "0", 10);
    const v = parseInt(inV.value || "0", 10);

    const js = await apiPost("/api/hsv_to_color", {h, s, v});
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
        multipart_mjpeg_generator(),
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

    # Convert a single pixel to HSV (OpenCV scale)
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    hexcol, rgb, bgr_list = bgr_to_hex(bgr)
    return jsonify(
        ok=True,
        x=x, y=y,
        hsv=[h, s, v],
        rgb=rgb,
        bgr=bgr_list,
        hex=hexcol,
    )

@app.post("/api/hsv_to_color")
def api_hsv_to_color():
    data = request.get_json(force=True, silent=True) or {}
    h = data.get("h", 0)
    s = data.get("s", 0)
    v = data.get("v", 0)

    hexcol, rgb, bgr = hsv_opencv_to_color(h, s, v)
    return jsonify(ok=True, hsv=[int(max(0, min(179, int(h)))), int(max(0, min(255, int(s)))), int(max(0, min(255, int(v))))],
                   rgb=rgb, bgr=bgr, hex=hexcol)

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
