#!/usr/bin/env python3
from flask import Flask, Response
import threading
import time
import socket
import cv2
import numpy as np

# =========================
# Camera config
# =========================
DEVICE = "/dev/video4"
WIDTH = 320     # Lower = faster
HEIGHT = 240
FPS = 10
PORT = 5050
JPEG_QUALITY = 80

app = Flask(__name__)

# =========================
# Load MobileNet-SSD
# =========================
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)

# =========================
# Shared buffers
# =========================
_latest_lock = threading.Lock()
_latest_jpeg = None
_latest_seq = 0

# =========================
# Utility
# =========================
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

# =========================
# Camera thread
# =========================
def camera_worker():
    global _latest_jpeg, _latest_seq

    cap = open_camera()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            (h, w) = frame.shape[:2]

            # =========================
            # Detection
            # =========================
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                0.007843,
                (300, 300),
                127.5
            )

            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    label = f"{CLASSES[idx]}: {confidence:.2f}"

                    cv2.rectangle(frame, (startX, startY),
                                  (endX, endY), (0, 255, 0), 2)

                    cv2.putText(frame, label,
                                (startX, startY - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 255, 0), 1)

            # Encode JPEG
            ok2, jpg = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if not ok2:
                continue

            with _latest_lock:
                _latest_jpeg = jpg.tobytes()
                _latest_seq += 1

            time.sleep(max(0.0, 1.0 / FPS))

    finally:
        cap.release()

# =========================
# MJPEG stream
# =========================
def generate_stream():
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

        yield (
            boundary +
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpg +
            b"\r\n"
        )

# =========================
# Flask routes
# =========================
@app.route("/")
def index():
    return """
    <html>
    <body style="margin:0;background:black;">
        <img src="/video" style="width:100%;height:auto;">
    </body>
    </html>
    """

@app.route("/video")
def video():
    return Response(generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# =========================
# Main
# =========================
if __name__ == "__main__":
    t = threading.Thread(target=camera_worker, daemon=True)
    t.start()

    ips = detect_local_ips()
    for ip in ips:
        print(f"Open in browser: http://{ip}:{PORT}/")

    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)
