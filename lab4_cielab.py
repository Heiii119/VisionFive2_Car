#!/usr/bin/env python3
from flask import Flask, Response, render_template_string, request, jsonify
import threading
import time
import socket

import cv2
import numpy as np

# =========================
# Camera / Server config
# =========================
DEVICE = "/dev/video4"   # change to "/dev/video0" if needed
WIDTH = 640
HEIGHT = 480
FPS = 15
PORT = 7072
JPEG_QUALITY = 80

app = Flask(__name__)

# =========================
# Shared buffers (latest frame + JPEGs)
# =========================
_latest_lock = threading.Lock()
_latest_bgr = None
_latest_jpeg_color = None
_latest_jpeg_L = None
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

    # Try to reduce latency if supported
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    return cap

def bgr_to_hex_and_rgb(bgr):
    b = int(bgr[0])
    g = int(bgr[1])
    r = int(bgr[2])
    return f"#{r:02x}{g:02x}{b:02x}", [r, g, b]

def opencv_lab8_from_bgr(bgr):
    """
    OpenCV Lab for uint8 images:
      L channel:   0..255  (represents L* scaled)
      a channel:   0..255  (represents a* shifted by +128)
      b channel:   0..255  (represents b* shifted by +128)
    """
    px = np.uint8([[bgr]])
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0]  # [L8, a8, b8]
    return int(lab[0]), int(lab[1]), int(lab[2])

def labstar_from_opencv_lab8(L8, a8, b8):
    """
    Convert OpenCV's 8-bit Lab approx back to CIELAB-style numbers.
    Common classroom-friendly mapping:
      L*  ≈ L8 * 100/255
      a*  ≈ a8 - 128
      b*  ≈ b8 - 128
    """
    Lstar = float(L8) * 100.0 / 255.0
    astar = float(a8) - 128.0
    bstar = float(b8) - 128.0
    return Lstar, astar, bstar

def camera_worker():
    global _latest_bgr, _latest_jpeg_color, _latest_jpeg_L, _latest_seq
    cap = open_camera()
    try:
        while not _cam_stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
                frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

            # Color JPEG
            ok2, jpg_color = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)],
            )
            if not ok2:
                continue

            # L* channel visualization (grayscale)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            L = lab[:, :, 0]  # uint8 0..255
            ok3, jpg_L = cv2.imencode(
                ".jpg",
                L,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)],
            )
            if not ok3:
                continue

            with _latest_lock:
                _latest_bgr = frame
                _latest_jpeg_color = jpg_color.tobytes()
                _latest_jpeg_L = jpg_L.tobytes()
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
            if which == "L":
                jpg = _latest_jpeg_L
            else:
                jpg = _latest_jpeg_color

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

def get_latest_bgr_copy():
    with _latest_lock:
        if _latest_bgr is None:
            return None
        return _latest_bgr.copy()

HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no" />
  <title>CIELAB Tap Lab</title>
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
    .videoGrid{
      position:absolute; inset:0;
      display:grid; grid-template-columns: 1fr 1fr;
      gap:2px; background:#000;
    }
    .pane{position:relative; overflow:hidden; background:#000; display:grid; place-items:center;}
    .pane img{
      width:100%; height:100%;
      object-fit:contain;
      background:#000;
      touch-action:manipulation;
    }
    .paneLabel{
      position:absolute; left:10px; top:10px; z-index:2;
      font-size:16px; font-weight:900;
      padding:6px 10px; border-radius:10px;
      border:1px solid var(--stroke);
      background:rgba(0,0,0,0.45);
    }
    @media (orientation: portrait){
      .videoGrid{ grid-template-columns: 1fr; grid-template-rows: 1fr 1fr; }
    }

    .panel{
      position:absolute; left:12px; top:12px; right:12px;
      display:grid; grid-template-columns: 1.25fr 0.75fr;
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
    .bigLine{font-size:26px;font-weight:900;letter-spacing:0.2px;}
    .bigSub{font-size:18px;font-weight:800;color:var(--muted);margin-top:6px;}
    .small{font-size:14px;color:var(--muted);margin-top:8px;line-height:1.35;}
    .mono{font-variant-numeric: tabular-nums;}
    .swatch{
      height:90px;
      border-radius:14px;
      border:1px solid var(--stroke);
      background:#222;
      margin-top:10px;
    }
  </style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <div style="font-size:18px;font-weight:900;">CIELAB Tap Lab</div>
      <div class="meta">
        Tap either view to sample the camera pixel. Left: colour. Right: L channel (lightness view).
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
        <div class="paneLabel">Lab: L channel</div>
        <img id="streamL" src="/mjpg_L" alt="L channel stream" />
      </div>
    </div>

    <div class="panel">
      <div class="card">
        <div class="bigLine mono" id="line1">Tap a pixel…</div>
        <div class="bigSub mono" id="line2">RGB: —   HEX: —</div>
        <div class="bigSub mono" id="line3">Lab (OpenCV 8-bit): —</div>
        <div class="bigSub mono" id="line4">Lab (approx CIELAB): —</div>
        <div class="swatch" id="swatch"></div>
        <div class="small">
          Note: OpenCV stores Lab as 8-bit values; a and b are shifted by +128.
        </div>
      </div>

      <div class="card">
        <div style="font-size:18px;font-weight:900;">Class prompts</div>
        <div class="small">
          • Tap the same object in shadow vs light — how does L change?<br/>
          • Tap red vs green areas — how do a and b change?<br/>
          • Tap blue vs yellow areas — what happens to b?
        </div>
      </div>
    </div>
  </div>
</div>

<script>
(() => {
  const statusEl = document.getElementById("status");
  const imgColor = document.getElementById("streamColor");
  const imgL = document.getElementById("streamL");

  const line1 = document.getElementById("line1");
  const line2 = document.getElementById("line2");
  const line3 = document.getElementById("line3");
  const line4 = document.getElementById("line4");
  const swatch = document.getElementById("swatch");

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
      await apiPost("/api/pick_pixel", {x: 0, y: 0});
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

  async function handleTap(imgEl, e){
    e.preventDefault();

    const rect = imgEl.getBoundingClientRect();
    const x = Math.round((e.clientX - rect.left) / rect.width * {{w}});
    const y = Math.round((e.clientY - rect.top) / rect.height * {{h}});

    try{
      const js = await apiPost("/api/pick_pixel", {x, y});
      if (!js || !js.ok) return;

      line1.textContent = `x=${js.x}  y=${js.y}`;
      line2.textContent = `RGB: ${js.rgb[0]}, ${js.rgb[1]}, ${js.rgb[2]}   HEX: ${js.hex}`;
      line3.textContent = `Lab (OpenCV 8-bit): L=${js.lab8[0]}  a=${js.lab8[1]}  b=${js.lab8[2]}`;
      line4.textContent = `Lab (approx CIELAB): L*=${js.labStar[0].toFixed(2)}  a*=${js.labStar[1].toFixed(2)}  b*=${js.labStar[2].toFixed(2)}`;

      swatch.style.background = js.hex;
    }catch(err){
      // ignore
    }
  }

  imgColor.addEventListener("pointerdown", (e) => handleTap(imgColor, e), {passive:false});
  imgL.addEventListener("pointerdown", (e) => handleTap(imgL, e), {passive:false});
})();
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(HTML, w=WIDTH, h=HEIGHT)

@app.get("/mjpg")
def mjpg_color():
    resp = Response(
        multipart_mjpeg_generator("color"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp

@app.get("/mjpg_L")
def mjpg_L():
    resp = Response(
        multipart_mjpeg_generator("L"),
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

    frame = get_latest_bgr_copy()
    if frame is None:
        return jsonify(ok=False, msg="No frame yet."), 503

    x = max(0, min(WIDTH - 1, x))
    y = max(0, min(HEIGHT - 1, y))

    bgr = frame[y, x].astype(np.uint8)
    hexcol, rgb = bgr_to_hex_and_rgb(bgr)

    L8, a8, b8 = opencv_lab8_from_bgr(bgr)
    Lstar, astar, bstar = labstar_from_opencv_lab8(L8, a8, b8)

    return jsonify(
        ok=True,
        x=x, y=y,
        rgb=rgb,
        hex=hexcol,
        lab8=[L8, a8, b8],
        labStar=[Lstar, astar, bstar],
    )

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
