from flask import Flask, Response, render_template_string
import subprocess
import socket

DEVICE = "/dev/video4"

# Your camera reports YUYV (uncompressed), so encoding to JPEG costs CPU.
# Start with 640x480 @ 15fps for better performance on StarFive.
WIDTH = 640
HEIGHT = 480
FPS = 15

PORT = 5000

app = Flask(__name__)

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
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # UDP connect trick: no packets need to be sent to learn outbound IP
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass

    ips.discard("127.0.0.1")
    return sorted(ips)

def ffmpeg_jpeg_pipe():
    """
    Produce a sequence of individual JPEG images to stdout.

    Your webcam node (/dev/video4) exposes 'YUYV' only, so we must:
    - capture yuyv422 from v4l2
    - encode to mjpeg in ffmpeg
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",

        "-f", "video4linux2",
        "-input_format", "yuyv422",
        "-framerate", str(FPS),
        "-video_size", f"{WIDTH}x{HEIGHT}",
        "-i", DEVICE,

        "-an",

        # Encode each frame as JPEG (adjust q:v for quality/CPU tradeoff)
        "-c:v", "mjpeg",
        "-q:v", "7",

        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0
    )

def multipart_mjpeg_generator():
    p = ffmpeg_jpeg_pipe()
    boundary = b"--frame\r\n"

    def read_one_jpeg(stream):
        # JPEG starts with 0xFFD8 and ends with 0xFFD9
        start = stream.read(2)
        if not start:
            return None

        # Sync to JPEG SOI marker
        while start != b"\xff\xd8":
            nxt = stream.read(1)
            if not nxt:
                return None
            start = start[1:] + nxt

        buf = bytearray(start)
        while True:
            chunk = stream.read(4096)
            if not chunk:
                return None
            buf.extend(chunk)

            eoi = buf.find(b"\xff\xd9")
            if eoi != -1:
                return bytes(buf[: eoi + 2])

    try:
        while True:
            jpg = read_one_jpeg(p.stdout)
            if jpg is None:
                break

            headers = (
                boundary +
                b"Content-Type: image/jpeg\r\n" +
                f"Content-Length: {len(jpg)}\r\n\r\n".encode()
            )
            yield headers + jpg + b"\r\n"
    finally:
        try:
            p.kill()
        except Exception:
            pass

@app.get("/")
def index():
    return render_template_string(HTML, device=DEVICE, w=WIDTH, h=HEIGHT, fps=FPS)

@app.get("/mjpg")
def mjpg():
    return Response(
        multipart_mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

if __name__ == "__main__":
    print(f'Device: {DEVICE} (YUYV) -> FFmpeg MJPEG -> Browser')
    print(f'URL pattern: "http://<starfive-ip>:{PORT}/"')
    print("Tip: Find the board IP by: $ ip a")

    ips = detect_local_ips()
    if ips:
        print("Detected IP address(es):")
        for ip in ips:
            print(f"  http://{ip}:{PORT}/")
    else:
        print("Could not auto-detect a non-loopback IP. Use: ip a")

    app.run(host="0.0.0.0", port=PORT, threaded=True)
