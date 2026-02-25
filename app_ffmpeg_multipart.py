#!/usr/bin/env python3
"""
Live MJPEG stream + big driving label overlay
+ PCA9685 motor/servo output
+ WEB Emergency Stop (E-STOP latch)

Endpoints:
  /            - page with stream + E-STOP controls
  /mjpg        - MJPEG stream (overlay label + E-STOP state)
  POST /estop  - engage E-STOP (latches)
  POST /release- release E-STOP (unlatch)
  GET  /estop?on=1 or /estop?on=0 (optional convenience)
  GET  /estop_state - returns JSON {estop: bool}
"""

import time
import threading
import socket
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request, redirect, jsonify

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
# Enable/disable PCA9685 output
# =========================
ENABLE_PCA9685 = True  # set False to test without hardware

# =========================
# PCA9685 config (your values)
# =========================
PCA9685_ADDR = 0x40
PCA9685_FREQ = 60  # Hz
I2C_BUS = 0

THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

THROTTLE_STOPPED_TICKS = 370
THROTTLE_FORWARD_TICKS = 415
THROTTLE_REVERSE_TICKS = 330

STEERING_LEFT_TICKS = 455
STEERING_CENTER_TICKS = 380
STEERING_RIGHT_TICKS = 305

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

# Decision thresholds for the LABEL (not for PWM)
TURN_THRESH = 0.15

# =========================
# Global: E-STOP latch
# =========================
estop_lock = threading.Lock()
estop_latched = False
estop_reason = "manual"

def set_estop(on: bool, reason: str = "manual"):
    global estop_latched, estop_reason
    with estop_lock:
        estop_latched = bool(on)
        if on:
            estop_reason = reason

def get_estop():
    with estop_lock:
        return bool(estop_latched), str(estop_reason)

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
  .row { display:flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 12px; }
  .pill { display:inline-block; padding: 6px 10px; border:1px solid #ddd; border-radius: 999px; background:#fafafa; }
  code { background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }
  button {
    border: 0; border-radius: 12px; padding: 12px 16px;
    font-weight: 800; cursor: pointer;
  }
  .stop { background: #b00020; color: #fff; }
  .rel  { background: #1b5e20; color: #fff; }
  .small { font-size: 12px; color: #555; }
</style>

<div class="wrap">
  <h1>Live Stream</h1>

  <div class="row">
    <span class="pill">Stream: <code>/mjpg</code></span>

    <form method="POST" action="/estop">
      <button class="stop" type="submit">EMERGENCY STOP</button>
    </form>

    <form method="POST" action="/release">
      <button class="rel" type="submit">Release</button>
    </form>

    <span class="pill" id="estopPill">E-STOP: …</span>
  </div>

  <div class="small">
    Tip: If you lose Wi‑Fi, this button can’t help. Consider adding a physical kill switch too.
  </div>

  <img src="/mjpg" />
</div>

<script>
  async function poll() {
    try {
      const r = await fetch("/estop_state", { cache: "no-store" });
      const s = await r.json();
      const pill = document.getElementById("estopPill");
      if (s.estop) {
        pill.textContent = "E-STOP: ON (" + (s.reason || "manual") + ")";
        pill.style.background = "#ffe6e6";
        pill.style.borderColor = "#b00020";
      } else {
        pill.textContent = "E-STOP: OFF";
        pill.style.background = "#eaffea";
        pill.style.borderColor = "#1b5e20";
      }
    } catch (e) {
      // ignore
    }
  }
  poll();
  setInterval(poll, 500);
</script>
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

@app.post("/estop")
def estop_post():
    set_estop(True, "web")
    return redirect("/")

@app.post("/release")
def release_post():
    set_estop(False, "web")
    return redirect("/")

@app.get("/estop")
def estop_get():
    # convenience: /estop?on=1 or /estop?on=0
    on = request.args.get("on", "").strip()
    if on in ("1", "true", "on", "yes"):
        set_estop(True, "web")
    elif on in ("0", "false", "off", "no"):
        set_estop(False, "web")
    return redirect("/")

@app.get("/estop_state")
def estop_state():
    on, reason = get_estop()
    return jsonify({"estop": on, "reason": reason})

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

# =========================
# Helpers
# =========================
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

def draw_banner(frame_bgr, label: str, steer_ticks: int, throttle_ticks: int, estop_on: bool):
    h, w = frame_bgr.shape[:2]
    banner_h = max(80, h // 7)

    if estop_on:
        label2 = "E-STOP"
        color = (0, 0, 255)
    else:
        label2 = label
        if label2 == "LOST LINE":
            color = (0, 0, 255)
        elif label2 in ("TURN LEFT", "TURN RIGHT"):
            color = (0, 165, 255)
        else:
            color = (0, 200, 0)

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.35 if w >= 640 else 1.1
    thickness = 3

    (tw, th), _ = cv2.getTextSize(label2, font, scale, thickness)
    x = max(12, (w - tw) // 2)
    y = (banner_h + th) // 2 - 8
    cv2.putText(frame_bgr, label2, (x, y), font, scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(frame_bgr, label2, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    small = f"steer={steer_ticks}  throttle={throttle_ticks}"
    cv2.putText(frame_bgr, small, (12, banner_h - 14), font, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame_bgr, small, (12, banner_h - 14), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return frame_bgr

# =========================
# Main loop
# =========================
def main():
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
            print(f"[pca] init failed: {e}\n[pca] continuing without PWM output")
            pca = None

    cap = open_camera()

    calibrated = False
    h_low = None
    h_high = None

    centroid_buf = deque(maxlen=CENTROID_HISTORY)
    last_seen = 0.0
    steer_f = float(steer_ticks)

    last_pub = 0.0
    print("\n[run] /mjpg stream + E-STOP. Ctrl+C to exit.\n")

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

            # Auto-calibrate once using center patch
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

            # Default outputs
            line_found = False
            cx = None
            err_norm = None

            # Vision only matters if not E-STOP (we still compute label though)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower = np.array([h_low, S_MIN, V_MIN], dtype=np.uint8)
            upper = np.array([h_high, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)

            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                err_norm = (cx_s - (w / 2.0)) / (w / 2.0)  # -1..+1
                last_seen = now

            age = now - last_seen if last_seen > 0 else 1e9
            label = compute_label(line_found, err_norm, age)

            # Compute steering/throttle normally...
            if line_found and err_norm is not None:
                steer_target = STEERING_CENTER_TICKS + KP_STEER * err_norm
                steer_target = np.clip(steer_target, STEERING_MIN_TICKS, STEERING_MAX_TICKS)
                steer_f = lerp(steer_f, float(steer_target), STEER_SMOOTH_ALPHA)
                steer_ticks_cmd = int(round(steer_f))
                throttle_ticks_cmd = int(FOLLOW_THROTTLE_TICKS)
            else:
                steer_ticks_cmd = int(round(steer_f))
                if LOST_LINE_BRAKE and (now - last_seen) > LOST_LINE_TIMEOUT_SEC:
                    throttle_ticks_cmd = int(THROTTLE_STOPPED_TICKS)
                else:
                    throttle_ticks_cmd = int(FOLLOW_THROTTLE_TICKS)

            steer_ticks_cmd = clamp(steer_ticks_cmd, STEERING_MIN_TICKS, STEERING_MAX_TICKS)
            throttle_ticks_cmd = clamp(throttle_ticks_cmd, THROTTLE_MIN_TICKS, THROTTLE_MAX_TICKS)

            # ...but E-STOP overrides throttle to STOPPED (latched)
            estop_on, _ = get_estop()
            if estop_on:
                throttle_ticks = int(THROTTLE_STOPPED_TICKS)
                steer_ticks = int(steer_ticks_cmd)  # keep steering (or set to center if you prefer)
            else:
                throttle_ticks = int(throttle_ticks_cmd)
                steer_ticks = int(steer_ticks_cmd)

            # Output to PCA9685
            if pca is not None:
                try:
                    pca.set_ticks(STEERING_CHANNEL, steer_ticks)
                    pca.set_ticks(THROTTLE_CHANNEL, throttle_ticks)
                except Exception as e:
                    print(f"[pca] write failed: {e} (disabling)")
                    pca = None

            out = draw_banner(frame_small, label, steer_ticks, throttle_ticks, estop_on)

            if now - last_pub >= (1.0 / STREAM_FPS):
                jpg = encode_jpg(out)
                if jpg is not None:
                    shared.update(jpg)
                last_pub = now

            # Optionally pace CPU a bit
            # time.sleep(max(0.0, (1.0/CONTROL_HZ) - (time.time() - now)))

    except KeyboardInterrupt:
        print("\n[exit] KeyboardInterrupt")
    finally:
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
