#!/usr/bin/env python3
"""
MJPEG live stream + ONE big label overlay ("LOST LINE" / "TURN LEFT" / "TURN RIGHT" / "GO STRAIGHT")

Endpoints:
  /       - web page showing the stream
  /mjpg   - MJPEG stream with overlay label (single stream)

This removes /debug_mjpg and /status to keep it simple and reliable.
"""

import time
import threading
import socket
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, render_template_string

# =========================
# Camera config
# =========================
DEVICE = "/dev/video4"
WIDTH = 1280
HEIGHT = 720
FPS = 30

STREAM_W = 640
STREAM_H = 480
STREAM_FPS = 15
JPEG_QUALITY = 75

# =========================
# Flask config
# =========================
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 9000

# =========================
# Line-follow tuning
# =========================
ROI_Y_START = 0.55

CAL_PATCH_W = 120
CAL_PATCH_H = 90

H_MARGIN = 12
S_MIN = 60
V_MIN = 60

MORPH_K = 5
MIN_CONTOUR_AREA = 900

CENTROID_HISTORY = 5
LOST_LINE_TIMEOUT_SEC = 0.6

# Decision thresholds (error_norm in [-1, +1])
TURN_THRESH = 0.15

# =========================
# Flask + shared frame
# =========================
app = Flask(__name__)

HTML = """
<!doctype html>
<title>Line Follow</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  body { font-family: system-ui, sans-serif; margin: 16px; }
  .wrap { max-width: 980px; margin: 0 auto; }
  img { width: 100%; height: auto; border: 1px solid #ccc; border-radius: 10px; display:block; }
  .pill { display:inline-block; padding: 6px 10px; border:1px solid #ddd; border-radius: 999px; background:#fafafa; margin-bottom: 10px;}
  code { background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }
</style>
<div class="wrap">
  <h1>Live Stream (with label overlay)</h1>
  <div class="pill">URL: <code>/mjpg</code></div>
  <img src="/mjpg" />
</div>
"""

class SharedFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpg = None
        self._ts = 0.0

    def update(self, jpg_bytes: bytes):
        with self._lock:
            self._jpg = jpg_bytes
            self._ts = time.time()

    def get(self):
        with self._lock:
            return self._jpg, self._ts

shared = SharedFrame()

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

def encode_jpg(frame_bgr):
    ok, jpg = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)],
    )
    if not ok:
        return None
    return jpg.tobytes()

def mjpeg_generator():
    boundary = b"--frame\r\n"
    last_ts = 0.0
    while True:
        jpg, ts = shared.get()
        if jpg is None or ts == last_ts:
            time.sleep(0.01)
            continue
        last_ts = ts
        headers = (
            boundary
            + b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
        )
        yield headers + jpg + b"\r\n"

@app.get("/")
def index():
    return render_template_string(HTML)

@app.get("/mjpg")
def mjpg():
    print("[http] /mjpg client connected")
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

def start_flask_in_thread():
    def _run():
        ips = detect_local_ips()
        print(f"\n[web] Running on port {FLASK_PORT}. Open:")
        if ips:
            for ip in ips:
                print(f"  http://{ip}:{FLASK_PORT}/")
        else:
            print(f"  http://127.0.0.1:{FLASK_PORT}/")
        app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True, use_reloader=False)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t

# =========================
# Vision / label
# =========================
def open_camera():
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera: {DEVICE}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap

def compute_label(line_found: bool, err_norm, last_seen_age: float):
    if (not line_found) or (err_norm is None) or (last_seen_age > LOST_LINE_TIMEOUT_SEC):
        return "LOST LINE"
    if err_norm > TURN_THRESH:
        return "TURN RIGHT"
    if err_norm < -TURN_THRESH:
        return "TURN LEFT"
    return "GO STRAIGHT"

def draw_label(frame_bgr, label: str):
    # Big banner at top
    h, w = frame_bgr.shape[:2]
    banner_h = max(60, h // 8)

    if label == "LOST LINE":
        color = (0, 0, 255)
    elif label in ("TURN LEFT", "TURN RIGHT"):
        color = (0, 165, 255)
    else:
        color = (0, 200, 0)

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    alpha = 0.55
    cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)

    # Text with outline
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.4 if w >= 640 else 1.1
    thickness = 3

    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    x = max(12, (w - tw) // 2)
    y = (banner_h + th) // 2

    cv2.putText(frame_bgr, label, (x, y), font, scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(frame_bgr, label, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    return frame_bgr

def main():
    start_flask_in_thread()
    cap = open_camera()

    calibrated = False
    h_low = None
    h_high = None

    centroid_buf = deque(maxlen=CENTROID_HISTORY)
    last_seen = 0.0

    last_pub = 0.0

    print("\n[run] Streaming on /mjpg with label overlay. Ctrl+C to exit.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            now = time.time()

            frame_small = cv2.resize(frame, (STREAM_W, STREAM_H), interpolation=cv2.INTER_AREA)
            h, w = frame_small.shape[:2]

            roi_y0 = int(h * ROI_Y_START)
            roi = frame_small[roi_y0:h, :]

            # Auto-calibrate once using center patch in ROI
            if not calibrated:
                cx0 = w // 2 - CAL_PATCH_W // 2
                cy0 = roi.shape[0] // 2 - CAL_PATCH_H // 2
                patch = roi[cy0:cy0 + CAL_PATCH_H, cx0:cx0 + CAL_PATCH_W]
                hsvp = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                h_med = int(np.median(hsvp[:, :, 0]))
                h_low = int(max(0, h_med - H_MARGIN))
                h_high = int(min(179, h_med + H_MARGIN))
                calibrated = True
                print(f"[cal] h={h_med} => [{h_low},{h_high}]")

            # Threshold in HSV in ROI
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower = np.array([h_low, S_MIN, V_MIN], dtype=np.uint8)
            upper = np.array([h_high, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)

            # Morphology
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

            # Find biggest contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            line_found = False
            cx = None
            err_norm = None

            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) >= MIN_CONTOUR_AREA:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = (M["m10"] / M["m00"])
                        line_found = True

            if line_found and cx is not None:
                centroid_buf.append(cx)
                cx_s = float(np.mean(centroid_buf))
                err_norm = (cx_s - (w / 2.0)) / (w / 2.0)
                last_seen = now

            age = now - last_seen if last_seen > 0 else 1e9
            label = compute_label(line_found, err_norm, age)

            # Draw label overlay onto the streamed frame
            out = draw_label(frame_small, label)

            # Publish at STREAM_FPS
            if now - last_pub >= (1.0 / STREAM_FPS):
                jpg = encode_jpg(out)
                if jpg is not None:
                    shared.update(jpg)
                last_pub = now

    except KeyboardInterrupt:
        print("\n[exit] KeyboardInterrupt")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        print("[exit] done")

if __name__ == "__main__":
    main()
