#!/usr/bin/env python3
import time
import sys
import signal
import threading
import subprocess
import socket

from flask import Flask, Response, render_template_string
from flask_socketio import SocketIO


# =========================
# Camera config (FFmpeg)
# =========================
DEVICE = "/dev/video4"
WIDTH = 1280
HEIGHT = 720
FPS = 30

# If your camera isn't YUYV, set input format accordingly or remove -input_format line
V4L2_INPUT_FORMAT = "yuyv422"  # common: yuyv422, mjpeg

# Web server
PORT = 5000


# =========================
# PCA9685 config (your values)
# =========================
PCA9685_ADDR = 0x40
PCA9685_FREQ = 60   # Hz
I2C_BUS      = 0
DRIVER_PREFER = "smbus2"   # "smbus2", "legacy", or "auto"

THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

THROTTLE_STOPPED_TICKS = 370
THROTTLE_FORWARD_TICKS = 385
THROTTLE_REVERSE_TICKS = 330

STEERING_LEFT_TICKS   = 280
STEERING_CENTER_TICKS = 380
STEERING_RIGHT_TICKS  = 480

STEERING_MIN_TICKS = 305
STEERING_MAX_TICKS = 455

THROTTLE_MIN_TICKS = 280
THROTTLE_MAX_TICKS = 450

STOP_ON_EXIT = True


# =========================
# Control mapping
# =========================
CONTROL_HZ = 60.0
CONTROL_DT = 1.0 / CONTROL_HZ

# When buttons held, what commands to apply:
THROTTLE_STEP = 8        # ticks added/subtracted from STOPPED when pressing up/down
STEER_STEP = 45          # ticks added/subtracted from CENTER when pressing left/right
BRAKE_TICKS = THROTTLE_STOPPED_TICKS  # "brake" = stopped (you can change if your ESC supports braking)

FAILSAFE_TIMEOUT_SEC = 0.35  # if no updates from phone -> brake+center


# =========================
# PCA9685 driver (unchanged)
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
        try:
            from smbus2 import SMBus
        except Exception as e:
            raise SystemExit("Missing smbus2. Install with: python -m pip install smbus2") from e

        self.address = int(address)
        self._bus = SMBus(int(busnum))
        self._frequency = None

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
        self._bus.write_byte_data(self.address, reg, val & 0xFF)

    def _read8(self, reg):
        return self._bus.read_byte_data(self.address, reg) & 0xFF

    def set_pwm_freq(self, freq_hz):
        osc = 25_000_000.0
        freq_hz = float(freq_hz)
        prescaleval = (osc / (4096.0 * freq_hz)) - 1.0
        prescale = int(round(prescaleval))
        prescale = max(3, min(255, prescale))

        oldmode = self._read8(self.MODE1)
        newmode = (oldmode & 0x7F) | self.SLEEP
        self._write8(self.MODE1, newmode)
        self._write8(self.PRESCALE, prescale)
        self._write8(self.MODE1, oldmode)
        time.sleep(0.005)
        self._write8(self.MODE1, oldmode | self.RESTART)
        self._frequency = freq_hz

    def set_pwm(self, channel, on, off):
        ch = int(channel)
        on = int(on) & 0x0FFF
        off = int(off) & 0x0FFF
        base = self.LED0_ON_L + 4 * ch
        self._write8(base + 0, on & 0xFF)
        self._write8(base + 1, (on >> 8) & 0xFF)
        self._write8(base + 2, off & 0xFF)
        self._write8(base + 3, (off >> 8) & 0xFF)

    def set_pwm_12bit(self, channel, value_12bit):
        v = max(0, min(4095, int(value_12bit)))
        self.set_pwm(channel, 0, v)


class PCA9685Driver:
    def __init__(self, address=0x40, busnum=1, frequency=60, prefer="smbus2"):
        self._mode = None
        self._drv = None
        smbus2_err = None

        if prefer in ("smbus2", "auto"):
            try:
                self._drv = PCA9685_SMBus2(busnum=busnum, address=address, frequency=frequency)
                self._mode = "smbus2"
                return
            except Exception as e:
                if prefer == "smbus2":
                    raise
                smbus2_err = e

        try:
            import Adafruit_PCA9685 as LegacyPCA9685
            self._drv = LegacyPCA9685.PCA9685(address=address, busnum=busnum)
            self._drv.set_pwm_freq(frequency)
            self._mode = "legacy"
        except Exception as e:
            raise SystemExit(
                "Could not initialize PCA9685.\n"
                "Tried:\n"
                f"  - smbus2 direct driver: {smbus2_err}\n"
                f"  - Adafruit_PCA9685: {e}\n\n"
                "Fix:\n"
                "  python -m pip install smbus2\n"
                "and ensure /dev/i2c-<bus> exists and i2cdetect shows 0x40.\n"
            )

    def close(self):
        if hasattr(self._drv, "close"):
            self._drv.close()

    def set_pwm_12bit(self, channel, value_12bit):
        if self._mode == "legacy":
            v = max(0, min(4095, int(value_12bit)))
            self._drv.set_pwm(channel, 0, v)
        else:
            self._drv.set_pwm_12bit(channel, value_12bit)


# =========================
# Helpers
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, int(v)))

def clamp_throttle(t):
    return clamp(t, THROTTLE_MIN_TICKS, THROTTLE_MAX_TICKS)

def clamp_steering(s):
    return clamp(s, STEERING_MIN_TICKS, STEERING_MAX_TICKS)

def safe_stop(pwm: PCA9685Driver):
    try:
        pwm.set_pwm_12bit(THROTTLE_CHANNEL, clamp_throttle(THROTTLE_STOPPED_TICKS))
        pwm.set_pwm_12bit(STEERING_CHANNEL, clamp_steering(STEERING_CENTER_TICKS))
    except Exception:
        pass

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


# =========================
# Control state shared with WebSocket
# =========================
state_lock = threading.Lock()
control_state = {
    "up": False,
    "down": False,
    "left": False,
    "right": False,
    "center": False,
    "brake": False,
    "last_seen": 0.0,
}


def compute_ticks_from_state(s: dict):
    """
    Map button state -> PWM ticks.
    This is deliberately simple. You can refine the mapping later.
    """
    # Steering
    if s.get("center", False):
        steer = STEERING_CENTER_TICKS
    elif s.get("left", False) and not s.get("right", False):
        steer = STEERING_CENTER_TICKS - STEER_STEP
    elif s.get("right", False) and not s.get("left", False):
        steer = STEERING_CENTER_TICKS + STEER_STEP
    else:
        steer = STEERING_CENTER_TICKS

    # Throttle
    if s.get("brake", False):
        throttle = BRAKE_TICKS
    else:
        if s.get("up", False) and not s.get("down", False):
            throttle = THROTTLE_STOPPED_TICKS + THROTTLE_STEP
        elif s.get("down", False) and not s.get("up", False):
            throttle = THROTTLE_STOPPED_TICKS - THROTTLE_STEP
        else:
            throttle = THROTTLE_STOPPED_TICKS

    return clamp_steering(steer), clamp_throttle(throttle)


def control_loop(pwm: PCA9685Driver):
    next_t = time.perf_counter()
    while True:
        next_t += CONTROL_DT

        with state_lock:
            s = dict(control_state)

        now = time.perf_counter()
        if (now - s["last_seen"]) > FAILSAFE_TIMEOUT_SEC:
            s["up"] = False
            s["down"] = False
            s["left"] = False
            s["right"] = False
            s["center"] = False
            s["brake"] = True

        steer, throttle = compute_ticks_from_state(s)

        try:
            pwm.set_pwm_12bit(STEERING_CHANNEL, steer)
            pwm.set_pwm_12bit(THROTTLE_CHANNEL, throttle)
        except Exception:
            # if i2c hiccups, keep loop alive
            pass

        remaining = next_t - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        else:
            next_t = time.perf_counter()


# =========================
# MJPEG streaming via FFmpeg
# =========================
def ffmpeg_jpeg_pipe():
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",

        "-f", "video4linux2",
        "-input_format", V4L2_INPUT_FORMAT,
        "-framerate", str(FPS),
        "-video_size", f"{WIDTH}x{HEIGHT}",
        "-i", DEVICE,

        "-an",
        "-c:v", "mjpeg",
        "-q:v", "7",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)


def multipart_mjpeg_generator():
    p = ffmpeg_jpeg_pipe()
    boundary = b"--frame\r\n"

    def read_one_jpeg(stream):
        # find SOI 0xFFD8
        start = stream.read(2)
        if not start:
            return None
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


# =========================
# Flask + SocketIO UI
# =========================
app = Flask(__name__)
app.config["SECRET_KEY"] = "fpv"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no" />
  <title>FPV Drive</title>
  <style>
    :root { --bg:#0b0f14; --panel:rgba(255,255,255,0.08); --panel2:rgba(0,0,0,0.35);
      --text:#e8eef7; --muted:rgba(232,238,247,0.75); --btn:rgba(255,255,255,0.10);
      --btnActive:rgba(76,175,80,0.35); --btnBrake:rgba(244,67,54,0.40); --stroke:rgba(255,255,255,0.14); }
    * { box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
    html, body { height:100%; margin:0; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
    .wrap { height:100%; display:grid; grid-template-rows:auto 1fr; }
    header { padding:10px 12px; display:flex; gap:12px; align-items:center; justify-content:space-between;
      border-bottom:1px solid var(--stroke); background:linear-gradient(to bottom, rgba(255,255,255,0.06), rgba(255,255,255,0.00)); }
    header .meta { font-size:12px; color:var(--muted); }
    header .status { font-size:12px; padding:6px 10px; border:1px solid var(--stroke); border-radius:999px; background:var(--panel); }
    .main { display:grid; grid-template-columns:1fr; grid-template-rows:1fr; position:relative; overflow:hidden; }
    .video { position:absolute; inset:0; display:grid; place-items:center; background:#000; }
    .video img { width:100%; height:100%; object-fit:contain; background:#000; }
    .controls { position:absolute; inset:0; padding:10px; display:grid; grid-template-columns:1fr 1fr; pointer-events:none; }
    .cluster { pointer-events:none; display:grid; align-content:end; gap:10px; }
    .cluster.left { justify-items:start; } .cluster.right { justify-items:end; }
    .pad { pointer-events:auto; background:var(--panel2); border:1px solid var(--stroke); border-radius:14px;
      padding:10px; display:grid; gap:10px; width:min(46vw, 340px); user-select:none; touch-action:none; backdrop-filter:blur(6px); }
    .row { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; }
    .row.two { grid-template-columns:1fr 1fr; }
    button.btn { width:100%; height:64px; border-radius:12px; border:1px solid var(--stroke); background:var(--btn);
      color:var(--text); font-size:16px; font-weight:700; letter-spacing:0.2px; }
    button.btn.active { background:var(--btnActive); }
    button.btn.brake.active { background:var(--btnBrake); }
    .hint { pointer-events:none; position:absolute; left:12px; top:56px; font-size:12px; color:var(--muted);
      background:rgba(0,0,0,0.35); border:1px solid var(--stroke); padding:6px 8px; border-radius:10px; }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <div><strong>FPV Drive</strong></div>
        <div class="meta">{{device}} • {{w}}x{{h}} @ {{fps}}fps</div>
      </div>
      <div id="status" class="status">Connecting…</div>
    </header>

    <div class="main">
      <div class="video"><img id="stream" src="/mjpg" alt="stream" /></div>
      <div class="hint">Hold buttons to drive • Arrow keys work too • Brake = Space</div>

      <div class="controls">
        <div class="cluster left">
          <div class="pad">
            <div class="row">
              <button class="btn" id="left">◀</button>
              <button class="btn" id="center">C</button>
              <button class="btn" id="right">▶</button>
            </div>
          </div>
        </div>

        <div class="cluster right">
          <div class="pad">
            <div class="row two">
              <button class="btn" id="up">▲</button>
              <button class="btn brake" id="brake">BRAKE</button>
            </div>
            <div class="row two">
              <button class="btn" id="down">▼</button>
              <button class="btn" disabled style="opacity:0.35"> </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script src="https://cdn.jsdelivr.net/npm/socket.io-client@4.7.5/dist/socket.io.min.js"></script>
<script>
(() => {
  const HZ = 60;
  const INTERVAL_MS = Math.round(1000 / HZ);

  const statusEl = document.getElementById('status');
  const btnIds = ["up","down","left","right","center","brake"];
  const btn = Object.fromEntries(btnIds.map(id => [id, document.getElementById(id)]));

  const state = { up:false, down:false, left:false, right:false, center:false, brake:false };

  function setActive(id, on) {
    state[id] = !!on;
    btn[id].classList.toggle('active', !!on);
  }

  function bindHold(id) {
    const el = btn[id];
    const down = (e) => { e.preventDefault(); setActive(id, true); };
    const up = (e) => { e.preventDefault(); setActive(id, false); };

    el.addEventListener('pointerdown', down);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
    el.addEventListener('pointerleave', up);
  }
  btnIds.forEach(bindHold);

  window.addEventListener('keydown', (e) => {
    if (e.repeat) return;
    if (e.key === "ArrowUp") setActive("up", true);
    if (e.key === "ArrowDown") setActive("down", true);
    if (e.key === "ArrowLeft") setActive("left", true);
    if (e.key === "ArrowRight") setActive("right", true);
    if (e.key === "c" || e.key === "C") setActive("center", true);
    if (e.code === "Space") setActive("brake", true);
  });
  window.addEventListener('keyup', (e) => {
    if (e.key === "ArrowUp") setActive("up", false);
    if (e.key === "ArrowDown") setActive("down", false);
    if (e.key === "ArrowLeft") setActive("left", false);
    if (e.key === "ArrowRight") setActive("right", false);
    if (e.key === "c" || e.key === "C") setActive("center", false);
    if (e.code === "Space") setActive("brake", false);
  });

  const socket = io({ transports: ["websocket"] });

  socket.on("connect", () => {
    statusEl.textContent = "Connected (WebSocket)";
    statusEl.style.opacity = "1.0";
  });
  socket.on("disconnect", () => {
    statusEl.textContent = "Disconnected…";
    statusEl.style.opacity = "0.7";
  });

  setInterval(() => {
    if (socket.connected) socket.emit("control", state);
  }, INTERVAL_MS);

  // Failsafe UX on tab switch
  const failSafe = () => {
    btnIds.forEach(id => setActive(id, false));
    setActive("brake", true);
    setTimeout(() => setActive("brake", false), 250);
  };
  window.addEventListener("blur", failSafe);
  document.addEventListener("visibilitychange", () => { if (document.hidden) failSafe(); });
})();
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(HTML, device=DEVICE, w=WIDTH, h=HEIGHT, fps=FPS)

@app.get("/mjpg")
def mjpg():
    return Response(
        multipart_mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@socketio.on("control")
def ws_control(data):
    def b(name):
        return bool((data or {}).get(name, False))
    with state_lock:
        control_state["up"] = b("up")
        control_state["down"] = b("down")
        control_state["left"] = b("left")
        control_state["right"] = b("right")
        control_state["center"] = b("center")
        control_state["brake"] = b("brake")
        control_state["last_seen"] = time.perf_counter()


def main():
    pwm = PCA9685Driver(
        address=PCA9685_ADDR,
        busnum=I2C_BUS,
        frequency=PCA9685_FREQ,
        prefer=DRIVER_PREFER,
    )

    def handle_exit(sig=None, frame=None):
        if STOP_ON_EXIT:
            safe_stop(pwm)
        try:
            pwm.close()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    safe_stop(pwm)

    t = threading.Thread(target=control_loop, args=(pwm,), daemon=True)
    t.start()

    ips = detect_local_ips()
    print(f"Open on phone: http://<board-ip>:{PORT}/")
    for ip in ips:
        print(f"  http://{ip}:{PORT}/")

    # IMPORTANT: use socketio.run (not app.run)
    socketio.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
