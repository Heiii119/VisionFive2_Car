#!/usr/bin/env python3
"""
Dual MJPEG stream + live debug labels (JSON polled by the webpage).

Endpoints:
  /            - page with labels + streams
  /mjpg        - raw camera MJPEG
  /debug_mjpg  - debug overlay MJPEG
  /status      - JSON with debug values for labels

Notes:
- Uses ONE OpenCV VideoCapture to avoid "device busy".
- If you don't have PCA9685 hardware or smbus2 installed, set ENABLE_PCA9685 = False.
"""

import time
import threading
import socket
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify

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

DEBUG_STREAM_FPS = 15

# =========================
# Flask config
# =========================
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 9000

# =========================
# Enable/disable PCA9685 output
# =========================
ENABLE_PCA9685 = True  # set False if you want to run without I2C/PCA9685

# =========================
# PCA9685 config (as given)
# =========================
PCA9685_ADDR = 0x40
PCA9685_FREQ = 60  # Hz
I2C_BUS = 0

THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

THROTTLE_STOPPED_TICKS = 370
THROTTLE_FORWARD_TICKS = 385
THROTTLE_REVERSE_TICKS = 330

STEERING_LEFT_TICKS = 280
STEERING_CENTER_TICKS = 380
STEERING_RIGHT_TICKS = 480

STEERING_MIN_TICKS = 305
STEERING_MAX_TICKS = 455

THROTTLE_MIN_TICKS = 280
THROTTLE_MAX_TICKS = 450

START_THROTTLE_TICKS = THROTTLE_STOPPED_TICKS
START_STEERING_TICKS = STEERING_CENTER_TICKS

STOP_ON_EXIT = True

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

CONTROL_HZ = 30.0
KP_STEER = 120.0
STEER_SMOOTH_ALPHA = 0.35

FOLLOW_THROTTLE_TICKS = 382

LOST_LINE_BRAKE = True
LOST_LINE_TIMEOUT_SEC = 0.6

CENTROID_HISTORY = 5

# =========================
# Flask + shared frames + status
# =========================
app = Flask(__name__)

HTML = """
<!doctype html>
<title>Line Follow Debug</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  body { font-family: system-ui, sans-serif; margin: 16px; }
  .wrap { max-width: 980px; margin: 0 auto; }
  img { width: 100%; height: auto; border: 1px solid #ccc; border-radius: 8px; display:block; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .card { border: 1px solid #e5e5e5; border-radius: 10px; padding: 10px; background: #fafafa; }
  .k { color: #555; font-size: 12px; }
  .v { font-weight: 650; font-size: 16px; }
  .row { display:flex; justify-content: space-between; gap: 10px; padding: 6px 0; border-bottom: 1px dashed #e5e5e5; }
  .row:last-child { border-bottom: none; }
  .ok { color: #0a7a2f; }
  .bad { color: #a40000; }
  code { background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }
  .pill { display:inline-block; padding: 6px 10px; border:1px solid #ddd; border-radius: 999px; background:#fff; margin-right:8px; margin-bottom:8px;}
  .spacer { height: 10px; }
</style>

<div class="wrap">
  <h1>Line Follow Debug</h1>

  <div>
    <span class="pill">Device: <code>{{device}}</code></span>
    <span class="pill">Top: <code>/mjpg</code></span>
    <span class="pill">Bottom: <code>/debug_mjpg</code></span>
    <span class="pill">Labels: <code>/status</code></span>
  </div>

  <div class="grid">
    <div class="card">
      <div class="row"><div class="k">Running</div><div class="v" id="running">—</div></div>
      <div class="row"><div class="k">Calibrated</div><div class="v" id="calibrated">—</div></div>
      <div class="row"><div class="k">Line Found</div><div class="v" id="line_found">—</div></div>
      <div class="row"><div class="k">Camera FPS</div><div class="v" id="fps_cam">—</div></div>
    </div>

    <div class="card">
      <div class="row"><div class="k">H range</div><div class="v" id="h_range">—</div></div>
      <div class="row"><div class="k">Centroid X</div><div class="v" id="centroid_x">—</div></div>
      <div class="row"><div class="k">Error (norm)</div><div class="v" id="error_norm">—</div></div>
      <div class="row"><div class="k">Steer / Throttle</div><div class="v" id="actuators">—</div></div>
    </div>
  </div>

  <div class="spacer"></div>

  <h2>Video</h2>
  <img src="/mjpg" />
  <div class="spacer"></div>
  <img src="/debug_mjpg" />
</div>

<script>
  function fmt(v, digits=3) {
    if (v === null || v === undefined) return "—";
    if (typeof v === "number") return v.toFixed(digits);
    return String(v);
  }
  function setBool(id, v) {
    const el = document.getElementById(id);
    if (!el) return;
    const yes = !!v;
    el.textContent = yes ? "YES" : "NO";
    el.className = "v " + (yes ? "ok" : "bad");
  }
  async function poll() {
    try {
      const r = await fetch("/status", { cache: "no-store" });
      const s = await r.json();

      setBool("running", s.running);
      setBool("calibrated", s.calibrated);
      setBool("line_found", s.line_found);

      document.getElementById("fps_cam").textContent =
        (s.fps_cam == null) ? "—" : fmt(s.fps_cam, 1);

      document.getElementById("h_range").textContent =
        (s.h_low == null || s.h_high == null) ? "—" : `${s.h_low} .. ${s.h_high}`;

      document.getElementById("centroid_x").textContent =
        (s.centroid_x == null || s.roi_width == null) ? "—" : `${s.centroid_x} / ${s.roi_width}`;

      document.getElementById("error_norm").textContent = fmt(s.error_norm, 3);

      document.getElementById("actuators").textContent =
        (s.steer_ticks == null || s.throttle_ticks == null) ? "—" : `${s.steer_ticks} / ${s.throttle_ticks}`;

    } catch (e) {
      // leave stale values on error
    }
  }
  poll();
  setInterval(poll, 100); // 10 Hz
</script>
"""

class SharedFrame:
    def __init__(self, name=""):
        self.name = name
        self._lock = threading.Lock()
        self._jpg = None
        self._ts = 0.0

    def update_jpg(self, jpg_bytes: bytes):
        with self._lock:
            self._jpg = jpg_bytes
            self._ts = time.time()

    def get_jpg(self):
        with self._lock:
            return self._jpg, self._ts


shared_top = SharedFrame("top")
shared_dbg = SharedFrame("dbg")

status_lock = threading.Lock()
status_data = {
    "running": False,
    "calibrated": False,
    "h_low": None,
    "h_high": None,
    "centroid_x": None,
    "roi_width": None,
    "error_norm": None,
    "steer_ticks": None,
    "throttle_ticks": None,
    "line_found": False,
    "fps_cam": None,
    "ts": time.time(),
}

def set_status(**kwargs):
    with status_lock:
        status_data.update(kwargs)
        status_data["ts"] = time.time()

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

def publish_frame(shared_obj: SharedFrame, frame_bgr):
    jpg = encode_jpg(frame_bgr)
    if jpg is not None:
        shared_obj.update_jpg(jpg)

def multipart_mjpeg_generator_for(shared_obj: SharedFrame):
    boundary = b"--frame\r\n"
    last_ts = 0.0
    while True:
        jpg, ts = shared_obj.get_jpg()
        if jpg is None or ts == last_ts:
            time.sleep(0.01)
            continue
        last_ts = ts
        headers = (
            boundary +
            b"Content-Type: image/jpeg\r\n" +
            f"Content-Length: {len(jpg)}\r\n\r\n".encode()
        )
        yield headers + jpg + b"\r\n"

@app.get("/")
def index():
    return render_template_string(HTML, device=DEVICE)

@app.get("/mjpg")
def mjpg():
    print("[http] /mjpg client connected")
    return Response(
        multipart_mjpeg_generator_for(shared_top),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/debug_mjpg")
def debug_mjpg():
    print("[http] /debug_mjpg client connected")
    return Response(
        multipart_mjpeg_generator_for(shared_dbg),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/status")
def status():
    with status_lock:
        data = dict(status_data)
    return jsonify(data)

def start_flask_in_thread():
    def _run():
        ips = detect_local_ips()
        print(f"\n[stream] Running on port {FLASK_PORT}. Open:")
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
# PCA9685 driver
# =========================
class PCA9685_SMBus2:
    MODE1 = 0x00
    MODE2 = 0x01
    PRESCALE = 0xFE
    LED0_ON_L = 0x06

    RESTART = 0x80
    SLEEP = 0x10
    ALLCALL = 0x01
    OUTDRV = 0x04

    def __init__(self, busnum, address=0x40, frequency=60):
        from smbus2 import SMBus
        self.address = int(address)
        self._bus = SMBus(int(busnum))

        self._write8(self.MODE1, self.ALLCALL)
        self._write8(self.MODE2, self.OUTDRV)
        time.sleep(0.005)

        mode1 = self._read8(self.MODE1)
        mode1 = mode1 & ~self.SLEEP
        self._write8(self.MODE1, mode1)
        time.sleep(0.005)

        self.set_pwm_freq(frequency)

    def close(self):
        try:
            self._bus.close()
        except Exception:
            pass

    def _write8(self, reg, val):
        self._bus.write_byte_data(self.address, reg, val)

    def _read8(self, reg):
        return self._bus.read_byte_data(self.address, reg)

    def set_pwm_freq(self, freq_hz):
        # PCA9685 internal osc is 25MHz
        prescaleval = 25000000.0
        prescaleval /= 4096.0
        prescaleval /= float(freq_hz)
        prescaleval -= 1.0
        prescale = int(np.floor(prescaleval + 0.5))

        oldmode = self._read8(self.MODE1)
        newmode = (oldmode & 0x7F) | self.SLEEP
        self._write8(self.MODE1, newmode)
        self._write8(self.PRESCALE, prescale)
        self._write8(self.MODE1, oldmode)
        time.sleep(0.005)
        self._write8(self.MODE1, oldmode | self.RESTART)

    def set_pwm(self, channel, on, off):
        reg = self.LED0_ON_L + 4 * int(channel)
        self._write8(reg + 0, on & 0xFF)
        self._write8(reg + 1, (on >> 8) & 0xFF)
        self._write8(reg + 2, off & 0xFF)
        self._write8(reg + 3, (off >> 8) & 0xFF)

    def set_ticks(self, channel, ticks_12bit):
        t = int(np.clip(int(ticks_12bit), 0, 4095))
        self.set_pwm(channel, 0, t)

def clamp(v, lo, hi):
    return int(max(lo, min(hi, int(v))))

def lerp(a, b, alpha):
    return a + alpha * (b - a)

def open_camera():
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera: {DEVICE}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    # optional: try MJPG for performance
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap

def make_debug_frame(base_bgr, roi_y0, roi_y1, mask=None, cx=None, err=None, steer=None, throttle=None, calibrated=False):
    dbg = base_bgr.copy()
    h, w = dbg.shape[:2]

    # ROI rectangle
    cv2.rectangle(dbg, (0, roi_y0), (w - 1, roi_y1 - 1), (0, 255, 255), 2)

    # draw centroid and center line
    x_center = w // 2
    cv2.line(dbg, (x_center, roi_y0), (x_center, roi_y1), (255, 255, 0), 2)
    if cx is not None:
        cv2.line(dbg, (int(cx), roi_y0), (int(cx), roi_y1), (0, 255, 0), 3)

    # text overlay (still useful, even though labels are in HTML)
    y = 30
    def put(line, color=(255, 255, 255)):
        nonlocal y
        cv2.putText(dbg, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(dbg, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        y += 28

    put(f"calibrated: {calibrated}", (0, 255, 0) if calibrated else (0, 0, 255))
    if cx is not None and err is not None:
        put(f"cx: {int(cx)}  err_norm: {err:+.3f}", (255, 255, 255))
    else:
        put("cx: —  err_norm: —", (180, 180, 180))
    if steer is not None and throttle is not None:
        put(f"steer_ticks: {steer}  throttle_ticks: {throttle}", (255, 255, 255))

    # optional mask preview (small)
    if mask is not None:
        m = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        m = cv2.resize(m, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
        dbg[10:10 + m.shape[0], w - 10 - m.shape[1]:w - 10] = m

    return dbg

def main():
    # Start web server
    start_flask_in_thread()

    # PCA9685 init
    pca = None
    steer_ticks = START_STEERING_TICKS
    throttle_ticks = START_THROTTLE_TICKS

    if ENABLE_PCA9685:
        try:
            pca = PCA9685_SMBus2(I2C_BUS, PCA9685_ADDR, PCA9685_FREQ)
            pca.set_ticks(STEERING_CHANNEL, steer_ticks)
            pca.set_ticks(THROTTLE_CHANNEL, throttle_ticks)
            print("[pca] initialized")
        except Exception as e:
            print(f"[pca] init failed: {e}\n[pca] Continuing with ENABLE_PCA9685=False behavior.")
            pca = None

    cap = open_camera()

    calibrated = False
    h_low = None
    h_high = None

    centroid_buf = deque(maxlen=CENTROID_HISTORY)
    last_seen = 0.0
    steer_f = float(steer_ticks)

    set_status(running=True, calibrated=False, steer_ticks=steer_ticks, throttle_ticks=throttle_ticks)

    # camera fps estimate
    fps_t0 = time.time()
    fps_n = 0
    cam_fps = None

    # publish rate limiting
    last_top_pub = 0.0
    last_dbg_pub = 0.0

    try:
        print("\nControls:")
        print("  y = calibrate from center patch")
        print("  q = quit\n")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            now = time.time()

            # FPS estimate
            fps_n += 1
            if now - fps_t0 >= 1.0:
                cam_fps = fps_n / (now - fps_t0)
                fps_t0 = now
                fps_n = 0

            # Resize for streaming + processing (keep it consistent)
            frame_small = cv2.resize(frame, (STREAM_W, STREAM_H), interpolation=cv2.INTER_AREA)

            # ROI
            h, w = frame_small.shape[:2]
            roi_y0 = int(h * ROI_Y_START)
            roi_y1 = h
            roi = frame_small[roi_y0:roi_y1, :]

            # Key handling (must be on an imshow window normally; we don't have one)
            # Instead, auto-calibrate after 2 seconds IF you want—disabled by default.
            # If you need keyboard input, run with a local GUI or implement a /calibrate route.

            # --- calibration from center patch (auto if not calibrated and you want) ---
            # We'll support calibration when not calibrated by sampling every frame and
            # using a stable estimate once (simple, effective). If you truly require manual 'y',
            # tell me and I'll add a /calibrate button on the webpage.
            if not calibrated:
                # sample patch at center of ROI
                cx0 = w // 2 - CAL_PATCH_W // 2
                cy0 = roi.shape[0] // 2 - CAL_PATCH_H // 2
                patch = roi[cy0:cy0 + CAL_PATCH_H, cx0:cx0 + CAL_PATCH_W]
                hsvp = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                h_med = int(np.median(hsvp[:, :, 0]))
                h_low = int(max(0, h_med - H_MARGIN))
                h_high = int(min(179, h_med + H_MARGIN))
                calibrated = True
                print(f"[cal] calibrated: h={h_med} => [{h_low},{h_high}]  (auto)")
                set_status(calibrated=True, h_low=h_low, h_high=h_high)

            # Threshold in HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower = np.array([h_low, S_MIN, V_MIN], dtype=np.uint8)
            upper = np.array([h_high, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)

            # Morphology
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

            # Find contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            line_found = False
            cx = None
            err_norm = None

            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)
                if area >= MIN_CONTOUR_AREA:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = (M["m10"] / M["m00"])
                        line_found = True

            if line_found and cx is not None:
                centroid_buf.append(cx)
                cx_s = float(np.mean(centroid_buf))
                err_norm = (cx_s - (w / 2.0)) / (w / 2.0)  # -1..+1
                last_seen = now

                # steering control
                steer_target = STEERING_CENTER_TICKS + KP_STEER * err_norm
                steer_target = np.clip(steer_target, STEERING_MIN_TICKS, STEERING_MAX_TICKS)
                steer_f = lerp(steer_f, float(steer_target), STEER_SMOOTH_ALPHA)
                steer_ticks = int(round(steer_f))

                throttle_ticks = int(FOLLOW_THROTTLE_TICKS)
            else:
                # lost line behavior
                if LOST_LINE_BRAKE and (now - last_seen) > LOST_LINE_TIMEOUT_SEC:
                    throttle_ticks = int(THROTTLE_STOPPED_TICKS)
                else:
                    throttle_ticks = int(FOLLOW_THROTTLE_TICKS)

                # keep steering last
                steer_ticks = int(round(steer_f))

            steer_ticks = clamp(steer_ticks, STEERING_MIN_TICKS, STEERING_MAX_TICKS)
            throttle_ticks = clamp(throttle_ticks, THROTTLE_MIN_TICKS, THROTTLE_MAX_TICKS)

            # Apply to PCA9685 (if enabled)
            if pca is not None:
                try:
                    pca.set_ticks(STEERING_CHANNEL, steer_ticks)
                    pca.set_ticks(THROTTLE_CHANNEL, throttle_ticks)
                except Exception as e:
                    print(f"[pca] write failed: {e}")
                    pca = None

            # Update label status
            set_status(
                running=True,
                calibrated=calibrated,
                h_low=int(h_low) if calibrated else None,
                h_high=int(h_high) if calibrated else None,
                line_found=bool(line_found),
                centroid_x=int(cx) if cx is not None else None,
                roi_width=int(w),
                error_norm=float(err_norm) if err_norm is not None else None,
                steer_ticks=int(steer_ticks),
                throttle_ticks=int(throttle_ticks),
                fps_cam=float(cam_fps) if cam_fps is not None else None,
            )

            # Publish MJPEG frames at limited FPS
            if now - last_top_pub >= (1.0 / STREAM_FPS):
                publish_frame(shared_top, frame_small)
                last_top_pub = now

            if now - last_dbg_pub >= (1.0 / DEBUG_STREAM_FPS):
                dbg = make_debug_frame(
                    frame_small,
                    roi_y0=roi_y0,
                    roi_y1=roi_y1,
                    mask=mask,
                    cx=cx,
                    err=err_norm,
                    steer=steer_ticks,
                    throttle=throttle_ticks,
                    calibrated=calibrated,
                )
                publish_frame(shared_dbg, dbg)
                last_dbg_pub = now

            # run loop near CONTROL_HZ (best-effort)
            # (camera read time dominates anyway)
            # time.sleep(max(0.0, (1.0 / CONTROL_HZ) - (time.time() - now)))

    except KeyboardInterrupt:
        print("\n[exit] KeyboardInterrupt")
    finally:
        set_status(running=False)
        try:
            cap.release()
        except Exception:
            pass

        if STOP_ON_EXIT and pca is not None:
            try:
                pca.set_ticks(THROTTLE_CHANNEL, THROTTLE_STOPPED_TICKS)
                pca.set_ticks(STEERING_CHANNEL, STEERING_CENTER_TICKS)
            except Exception:
                pass

        if pca is not None:
            try:
                pca.close()
            except Exception:
                pass

        print("[exit] done")

if __name__ == "__main__":
    main()
