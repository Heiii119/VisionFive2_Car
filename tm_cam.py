#!/usr/bin/env python3
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2

_latest_jpeg = None
_latest_lock = threading.Lock()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def make_index_html(model_base_url: str) -> str:
    # model_base_url must end with /
    if not model_base_url.endswith("/"):
        model_base_url += "/"

    # Uses Teachable Machine image library + TFJS via script tags
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Teachable Machine + Stream</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; }}
    .row {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }}
    .card {{ border: 1px solid #ddd; padding: 12px; border-radius: 8px; }}
    #stream {{ max-width: 640px; width: 100%; height: auto; border-radius: 8px; border: 1px solid #ccc; }}
    #status {{ white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .big {{ font-size: 18px; font-weight: 600; }}
  </style>
</head>
<body>
  <h2>Teachable Machine (Classification) + MJPEG Stream</h2>

  <div class="row">
    <div class="card">
      <div class="big">Live stream</div>
      <img id="stream" src="/mjpeg" alt="stream"/>
      <canvas id="canvas" width="224" height="224" style="display:none;"></canvas>
      <div style="margin-top:8px;">
        <button id="btnStart">Start prediction</button>
        <button id="btnStop">Stop</button>
      </div>
      <div style="margin-top:8px;">
        Predict FPS:
        <select id="fps">
          <option value="2">2</option>
          <option value="5" selected>5</option>
          <option value="10">10</option>
        </select>
      </div>
    </div>

    <div class="card" style="min-width: 320px;">
      <div class="big">Prediction</div>
      <div id="top"></div>
      <hr/>
      <div id="status">Loading…</div>
      <hr/>
      <div>
        <div class="big">Model URL</div>
        <div id="modelUrl"></div>
      </div>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@0.8.3/dist/teachablemachine-image.min.js"></script>

  <script>
    const MODEL_BASE = {model_base_url!r}; // must end with /
    document.getElementById("modelUrl").textContent = MODEL_BASE;

    const img = document.getElementById("stream");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    const statusEl = document.getElementById("status");
    const topEl = document.getElementById("top");
    const btnStart = document.getElementById("btnStart");
    const btnStop = document.getElementById("btnStop");
    const fpsSel = document.getElementById("fps");

    let model = null;
    let labels = [];
    let timer = null;

    async function loadModel() {{
      const modelURL = MODEL_BASE + "model.json";
      const metadataURL = MODEL_BASE + "metadata.json";
      statusEl.textContent = "Loading model…\\n" + modelURL;
      model = await tmImage.load(modelURL, metadataURL);
      labels = model.getClassLabels ? model.getClassLabels() : [];
      statusEl.textContent = "Model loaded. Classes: " + (labels.length ? labels.join(", ") : "(unknown)");
    }}

    function drawToCanvas224() {{
      // draw current MJPEG frame into a square canvas (224x224)
      // (Teachable Machine commonly trains on square inputs)
      const w = canvas.width, h = canvas.height;
      try {{
        ctx.drawImage(img, 0, 0, w, h);
        return true;
      }} catch (e) {{
        return false;
      }}
    }}

    async function predictOnce() {{
      if (!model) return;
      const ok = drawToCanvas224();
      if (!ok) return;

      // predict can take a canvas element
      const preds = await model.predict(canvas);
      preds.sort((a,b) => b.probability - a.probability);

      if (preds.length > 0) {{
        const p0 = preds[0];
        topEl.textContent = "Top: " + p0.className + "  (" + p0.probability.toFixed(3) + ")";
        topEl.className = "big";
      }}

      // show all scores
      const lines = preds.map(p => p.className + ": " + p.probability.toFixed(3));
      statusEl.textContent = lines.join("\\n");
    }}

    function startLoop() {{
      stopLoop();
      const fps = parseInt(fpsSel.value, 10) || 5;
      const intervalMs = Math.round(1000 / fps);
      timer = setInterval(() => {{
        predictOnce().catch(err => {{
          statusEl.textContent = "Prediction error: " + err;
        }});
      }}, intervalMs);
    }}

    function stopLoop() {{
      if (timer) {{
        clearInterval(timer);
        timer = null;
      }}
    }}

    btnStart.onclick = startLoop;
    btnStop.onclick = stopLoop;

    // wait for <img> to start receiving frames, then load model
    img.onload = async () => {{
      if (!model) {{
        try {{
          await loadModel();
        }} catch (e) {{
          statusEl.textContent = "Failed to load model.\\n" + e + "\\n\\nCheck Internet access for CDN + Teachable Machine URL.";
        }}
      }}
    }};
  </script>
</body>
</html>
"""

class StreamHandler(BaseHTTPRequestHandler):
    # We’ll inject the model URL via server config stored on the class:
    MODEL_BASE_URL = "https://teachablemachine.withgoogle.com/models/6QK_nvAsJ/"

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = make_index_html(self.MODEL_BASE_URL)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
            return

        if self.path == "/mjpeg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            try:
                while True:
                    with _latest_lock:
                        jpg = _latest_jpeg

                    if jpg is None:
                        time.sleep(0.05)
                        continue

                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                return

        self.send_response(404)
        self.end_headers()

def start_server(host, port, model_base_url):
    StreamHandler.MODEL_BASE_URL = model_base_url
    server = ThreadedHTTPServer((host, port), StreamHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server

def open_capture(source: str):
    source = source.rstrip("/")  # tolerate /dev/video4/
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    return cap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/dev/video4", help="Camera index (0) or device path (/dev/video4)")
    ap.add_argument("--host", default="0.0.0.0", help="Bind host")
    ap.add_argument("--port", type=int, default=9080, help="Bind port")
    ap.add_argument("--jpeg_quality", type=int, default=80, help="JPEG quality 1-100")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--model_url", default="https://teachablemachine.withgoogle.com/models/6QK_nvAsJ/",
                    help="Teachable Machine model base URL (must be the /models/<id>/ page)")
    args = ap.parse_args()

    cap = open_capture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera source: {args.source}")

    # Try to request a reasonable format
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    start_server(args.host, args.port, args.model_url)

    print(f"Open on another device: http://<BOARD_IP>:{args.port}/")
    print(f"Raw MJPEG endpoint:    http://<BOARD_IP>:{args.port}/mjpeg")
    print(f"Camera source:         {args.source}")
    print(f"Model base URL:        {args.model_url}")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            ok2, jpg = cv2.imencode(".jpg", frame, encode_params)
            if ok2:
                global _latest_jpeg
                with _latest_lock:
                    _latest_jpeg = jpg.tobytes()
    finally:
        cap.release()

if __name__ == "__main__":
    main()
