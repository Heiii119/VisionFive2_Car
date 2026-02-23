from flask import Flask, Response, render_template_string
import cv2
import threading
import time
import socket

DEVICE = "/dev/video4"   # C270 HD WEBCAM
WIDTH = 640
HEIGHT = 480
FPS = 30

app = Flask(__name__)

latest_jpeg = None
lock = threading.Lock()

HTML = """
<!doctype html>
<title>Webcam Stream</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 20px; }
  img { max-width: 100%; height: auto; border: 1px solid #ccc; }
</style>
<h1>Webcam Stream</h1>
<p>Device: {{device}} ({{w}}x{{h}} @ {{fps}}fps)</p>
<img src="/mjpg" />
"""

def detect_local_ips():
    """
    Returns a sorted list of likely LAN IPs for this host.
    - Uses a UDP 'connect' trick to learn the preferred outbound IP (no packets sent).
    - Also tries hostname resolution as a fallback.
    """
    ips = set()

    # Preferred outbound IP (best single answer most of the time)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # no traffic required for UDP connect()
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass

    # Fallback: hostname-based IPs
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass

    # Remove loopback
    ips.discard("127.0.0.1")

    return sorted(ips)

def camera_thread():
    global latest_jpeg

    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {DEVICE}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Try MJPG to reduce CPU (often supported by Logitech C270)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        with lock:
            latest_jpeg = buf.tobytes()

def mjpeg_generator():
    boundary = b"--frame\r\n"
    while True:
        with lock:
            frame = latest_jpeg
        if frame is None:
            time.sleep(0.02)
            continue

        yield (
            boundary +
            b"Content-Type: image/jpeg\r\n" +
            f"Content-Length: {len(frame)}\r\n\r\n".encode() +
            frame +
            b"\r\n"
        )
        time.sleep(0.001)

@app.get("/")
def index():
    return render_template_string(HTML, device=DEVICE, w=WIDTH, h=HEIGHT, fps=FPS)

@app.get("/mjpg")
def mjpg():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    print('URL pattern: "http://<starfive-ip>:5000/"')
    print('Tip: Find the board IP by: $ ip a')

    ips = detect_local_ips()
    if ips:
        print("Detected IP address(es):")
        for ip in ips:
            print(f"  http://{ip}:5000/")
    else:
        print("Could not auto-detect a non-loopback IP. Use: ip a")

    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=5000, threaded=True)
