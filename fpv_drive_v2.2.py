#!/usr/bin/env python3
from flask import Flask, Response, render_template_string, request, jsonify
import subprocess
import socket
import time
import threading
import signal

# =========================
# Camera / streaming config
# =========================
DEVICE = "/dev/video4"
WIDTH = 320
HEIGHT = 240
FPS = 10
PORT = 6060

# =========================
# PCA9685 / PWM CONFIG
# =========================
PCA9685_ADDR = 0x40
PCA9685_FREQ = 60   # Hz
I2C_BUS      = 0
DRIVER_PREFER = "smbus2"   # "smbus2", "legacy", or "auto"

THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

# TICKS (0..4095) calibration/presets
THROTTLE_STOPPED_TICKS = 370
THROTTLE_FORWARD_TICKS = 415
THROTTLE_REVERSE_TICKS = 305

STEERING_LEFT_TICKS   = 280
STEERING_CENTER_TICKS = 380
STEERING_RIGHT_TICKS  = 480

STEERING_MIN_TICKS = 305
STEERING_MAX_TICKS = 480

START_THROTTLE_TICKS = THROTTLE_STOPPED_TICKS
START_STEERING_TICKS = STEERING_CENTER_TICKS

STOP_ON_EXIT = True

# How fast values change while you HOLD a button
STEP = 5
STEERING_STEP = 25

# Safety behavior for web driving
THROTTLE_RELEASE_TO_STOP = True
STEERING_RELEASE_TO_CENTER = False  # set True if you want auto-center when you release L/R

# =========================
# Flask app
# =========================
app = Flask(__name__)

# -------------------------
# Control state (60Hz loop)
# -------------------------
CONTROL_HZ = 60.0
CONTROL_DT = 1.0 / CONTROL_HZ

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

FAILSAFE_TIMEOUT_SEC = 0.35

# -------------------------
# Web UI
#   FIXED: control sender avoids request queue by ensuring only 1 in-flight request
#   and still sends a heartbeat (so failsafe doesn't trigger)
# -------------------------
HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no" />
  <title>META DOT FPV Drive</title>
  <style>
    /* Prevent iOS long-press text selection / callout */
    html, body {
      -webkit-user-select: none;
      user-select: none;
      -webkit-touch-callout: none; /* disables Copy/Look Up bubble in many cases */
    }

    /* Also apply to controls explicitly */
    button, .btn, .pad, .hint, header, .meta, .status {
      -webkit-user-select: none;
      user-select: none;
      -webkit-touch-callout: none;
    }
    :root { --bg:#0b0f14; --panel:rgba(255,255,255,0.08); --panel2:rgba(0,0,0,0.35);
      --text:#e8eef7; --muted:rgba(232,238,247,0.75); --btn:rgba(255,255,255,0.10);
      --btnActive:rgba(76,175,80,0.35); --btnBrake:rgba(244,67,54,0.40); --stroke:rgba(255,255,255,0.14); }
    *{box-sizing:border-box;-webkit-tap-highlight-color:transparent;}
    html,body{height:100%;margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;}
    .wrap{height:100%;display:grid;grid-template-rows:auto 1fr;}
    header{padding:10px 12px;display:flex;gap:12px;align-items:center;justify-content:space-between;border-bottom:1px solid var(--stroke);
      background:linear-gradient(to bottom, rgba(255,255,255,0.06), rgba(255,255,255,0.00));}
    header .meta{font-size:12px;color:var(--muted);}
    header .status{font-size:12px;padding:6px 10px;border:1px solid var(--stroke);border-radius:999px;background:var(--panel);}
    .main{display:grid;grid-template-columns:1fr;grid-template-rows:1fr;position:relative;overflow:hidden;}
    .video{position:absolute;inset:0;display:grid;place-items:center;background:#000;}
    .video img{width:100%;height:100%;object-fit:contain;background:#000;}
    .controls{position:absolute;inset:0;padding:10px;display:grid;grid-template-columns:1fr 1fr;pointer-events:none;}
    .cluster{pointer-events:none;display:grid;align-content:end;gap:10px;}
    .cluster.left{justify-items:start;} .cluster.right{justify-items:end;}
    .pad{pointer-events:auto;background:var(--panel2);border:1px solid var(--stroke);border-radius:14px;padding:10px;display:grid;gap:10px;
      width:min(46vw,320px);user-select:none;touch-action:none;backdrop-filter:blur(6px);}
    .row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;} .row.two{grid-template-columns:1fr 1fr;}
    button.btn{width:100%;height:64px;border-radius:12px;border:1px solid var(--stroke);background:var(--btn);color:var(--text);
      font-size:16px;font-weight:700;letter-spacing:0.2px;}
    button.btn:active{transform:scale(0.99);} button.btn.active{background:var(--btnActive);} button.btn.brake.active{background:var(--btnBrake);}
    .hint{pointer-events:none;position:absolute;left:12px;top:56px;font-size:12px;color:var(--muted);background:rgba(0,0,0,0.35);
      border:1px solid var(--stroke);padding:6px 8px;border-radius:10px;}
    @media (orientation: portrait){ .hint::after{content:" (rotate phone to landscape)";} }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <div><strong>FPV Drive</strong></div>
        <div class="meta">Device: {{device}} • {{w}}x{{h}} @ {{fps}}fps • Control: 60Hz</div>
      </div>
      <div id="status" class="status">Connecting…</div>
    </header>

    <div class="main">
      <div class="video"><img id="stream" src="/mjpg" alt="stream" /></div>
      <div class="hint">Controls: Left/Right + Center (C) on left • Up/Down + Brake (Space) on right</div>

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
              <button class="btn" id="noop" disabled style="opacity:0.35"> </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
(() => {
  // Client -> server send rate.
  // Key point: we DO NOT allow request overlap, so no queue buildup.
  const HZ = 20;
  const INTERVAL_MS = Math.round(1000 / HZ);

  const statusEl = document.getElementById('status');
  const btnIds = ["up","down","left","right","center","brake"];
  const btn = Object.fromEntries(btnIds.map(id => [id, document.getElementById(id)]));

  const state = { up:false, down:false, left:false, right:false, center:false, brake:false };
  let connected = false;
  let lastOk = 0;

  // Flags for coalescing / pacing
  let inFlight = false;
  let dirty = true;       // force first send immediately
  let lastSentAt = 0;

  function markDirty() { dirty = true; }

  function setActive(id, on) {
    const v = !!on;
    if (state[id] === v) return;   // no change
    state[id] = v;
    btn[id].classList.toggle('active', v);
    markDirty();
  }

  function bindHold(id) {
    const el = btn[id];
    const down = (e) => { e.preventDefault(); setActive(id, true); };
    const up   = (e) => { e.preventDefault(); setActive(id, false); };
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

  async function sendOnce() {
    const r = await fetch("/control", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(state),
      cache: "no-store",
    });
    if (!r.ok) throw new Error("HTTP " + r.status);
  }

  function updateStatus() {
    const now = performance.now();
    const okRecent = connected && (now - lastOk) < 800;
    statusEl.textContent = okRecent ? `Connected (control ${HZ}Hz)` : "Disconnected…";
    statusEl.style.opacity = okRecent ? "1.0" : "0.7";
  }

  async function pump() {
    // Never overlap requests => no queue buildup
    if (inFlight) {
      setTimeout(pump, 5);
      return;
    }

    const now = performance.now();
    const due = (now - lastSentAt) >= INTERVAL_MS;

    // IMPORTANT: we still send periodically (heartbeat) so the server failsafe doesn't trigger.
    // Send if:
    // - state changed (dirty), OR
    // - it's time for the next heartbeat (due)
    if (!dirty && !due) {
      updateStatus();
      setTimeout(pump, 5);
      return;
    }

    inFlight = true;
    dirty = false;

    try {
      await sendOnce();
      connected = true;
      lastOk = performance.now();
      lastSentAt = lastOk;
    } catch (e) {
      connected = false;
    } finally {
      inFlight = false;
      updateStatus();
      // If input changed while request was in-flight, dirty will be true and we’ll send again immediately.
      setTimeout(pump, 0);
    }
  }

  pump();

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

# -------------------------
# Utilities
# -------------------------
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

# -------------------------
# Camera: 1x ffmpeg producer -> latest JPEG buffer -> MJPEG clients (drop frames)
# -------------------------
_latest_lock = threading.Lock()
_latest_jpeg = None
_latest_seq = 0
_camera_stop = threading.Event()

def ffmpeg_jpeg_pipe():
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
        "-c:v", "mjpeg",
        "-q:v", "7",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

def iter_jpegs(byte_stream):
    buf = bytearray()
    while True:
        chunk = byte_stream.read(4096)
        if not chunk:
            return
        buf.extend(chunk)

        while True:
            soi = buf.find(b"\xff\xd8")
            if soi == -1:
                if len(buf) > 1:
                    del buf[:-1]
                break

            eoi = buf.find(b"\xff\xd9", soi + 2)
            if eoi == -1:
                if soi > 0:
                    del buf[:soi]
                break

            jpg = bytes(buf[soi:eoi + 2])
            del buf[:eoi + 2]
            yield jpg

def camera_worker():
    global _latest_jpeg, _latest_seq
    while not _camera_stop.is_set():
        p = ffmpeg_jpeg_pipe()
        try:
            for jpg in iter_jpegs(p.stdout):
                if _camera_stop.is_set():
                    break
                with _latest_lock:
                    _latest_jpeg = jpg
                    _latest_seq += 1
        except Exception:
            pass
        finally:
            try:
                p.kill()
            except Exception:
                pass
            try:
                p.wait(timeout=1)
            except Exception:
                pass
        time.sleep(0.2)

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

# =========================
# PCA9685 driver (from your pwm.py, curses removed)
# =========================
class PCA9685_SMBus2:
    MODE1 = 0x00
    MODE2 = 0x01
    PRESCALE = 0xFE
    LED0_ON_L = 0x06
    ALL_LED_ON_L = 0xFA
    ALL_LED_OFF_L = 0xFC

    RESTART = 0x80
    SLEEP = 0x10
    ALLCALL = 0x01
    OUTDRV = 0x04

    def __init__(self, busnum, address=0x40, frequency=60):
        try:
            from smbus2 import SMBus
        except Exception as e:
            raise SystemExit("Missing smbus2. Install with: pip3 install --user smbus2") from e

        self.busnum = int(busnum)
        self.address = int(address)
        self._bus = SMBus(self.busnum)
        self._frequency = None

        self._write8(self.MODE1, self.ALLCALL)
        self._write8(self.MODE2, self.OUTDRV)
        time.sleep(0.005)

        mode1 = self._read8(self.MODE1)
        mode1 = mode1 & ~self.SLEEP
        self._write8(self.MODE1, mode1)
        time.sleep(0.005)

        self.set_pwm_freq(frequency)
        self.set_all_pwm(0, 0)

    def close(self):
        try:
            self._bus.close()
        except Exception:
            pass

    @property
    def frequency(self):
        return self._frequency

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

    def set_all_pwm(self, on, off):
        on = int(on) & 0x0FFF
        off = int(off) & 0x0FFF
        self._write8(self.ALL_LED_ON_L + 0, on & 0xFF)
        self._write8(self.ALL_LED_ON_L + 1, (on >> 8) & 0xFF)
        self._write8(self.ALL_LED_OFF_L + 0, off & 0xFF)
        self._write8(self.ALL_LED_OFF_L + 1, (off >> 8) & 0xFF)

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
                "  pip3 install --user smbus2\n"
                "and ensure /dev/i2c-<bus> exists and i2cdetect shows 0x40.\n"
            )

    @property
    def frequency(self):
        if self._mode == "smbus2":
            return self._drv.frequency
        return getattr(self._drv, "frequency", PCA9685_FREQ)

    def close(self):
        if hasattr(self._drv, "close"):
            self._drv.close()

    def set_pwm_freq(self, freq_hz):
        self._drv.set_pwm_freq(freq_hz)

    def set_pwm_12bit(self, channel, value_12bit):
        if self._mode == "legacy":
            v = max(0, min(4095, int(value_12bit)))
            self._drv.set_pwm(channel, 0, v)
        else:
            self._drv.set_pwm_12bit(channel, value_12bit)

# =========================
# PWM control runtime state
# =========================
pwm = None
pwm_lock = threading.Lock()
values = {
    "throttle": int(START_THROTTLE_TICKS),
    "steering": int(START_STEERING_TICKS),
}

def clamp12(v):
    return max(0, min(4095, int(v)))

def clamp_steering(v):
    return max(STEERING_MIN_TICKS, min(STEERING_MAX_TICKS, int(v)))

def pwm_init():
    global pwm
    pwm = PCA9685Driver(
        address=PCA9685_ADDR,
        busnum=I2C_BUS,
        frequency=PCA9685_FREQ,
        prefer=DRIVER_PREFER,
    )
    values["throttle"] = clamp12(values["throttle"])
    values["steering"] = clamp_steering(values["steering"])
    with pwm_lock:
        pwm.set_pwm_12bit(THROTTLE_CHANNEL, values["throttle"])
        pwm.set_pwm_12bit(STEERING_CHANNEL, values["steering"])

def pwm_safe_exit():
    global pwm
    if pwm is None:
        return
    if STOP_ON_EXIT:
        try:
            with pwm_lock:
                pwm.set_pwm_12bit(THROTTLE_CHANNEL, clamp12(THROTTLE_STOPPED_TICKS))
        except Exception:
            pass
    try:
        pwm.close()
    except Exception:
        pass
    pwm = None

# -------------------------
# Car control: apply web state to PWM ticks
# -------------------------
def send_control(s: dict):
    if pwm is None:
        return

    # THROTTLE
    if s.get("brake", False):
        values["throttle"] = clamp12(THROTTLE_STOPPED_TICKS)
    else:
        up = bool(s.get("up", False))
        down = bool(s.get("down", False))
        if up and not down:
            values["throttle"] = clamp12(values["throttle"] + STEP)
        elif down and not up:
            values["throttle"] = clamp12(values["throttle"] - STEP)
        else:
            if THROTTLE_RELEASE_TO_STOP:
                values["throttle"] = clamp12(THROTTLE_STOPPED_TICKS)

        values["throttle"] = max(
            THROTTLE_REVERSE_TICKS,
            min(THROTTLE_FORWARD_TICKS, values["throttle"])
        )

    # STEERING
    if s.get("center", False):
        values["steering"] = clamp_steering(STEERING_CENTER_TICKS)
    else:
        left = bool(s.get("left", False))
        right = bool(s.get("right", False))
        if left and not right:
            values["steering"] = clamp_steering(values["steering"] + STEERING_STEP)
        elif right and not left:
            values["steering"] = clamp_steering(values["steering"] - STEERING_STEP)
        else:
            if STEERING_RELEASE_TO_CENTER:
                values["steering"] = clamp_steering(STEERING_CENTER_TICKS)

    with pwm_lock:
        pwm.set_pwm_12bit(THROTTLE_CHANNEL, values["throttle"])
        pwm.set_pwm_12bit(STEERING_CHANNEL, values["steering"])

def control_loop():
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

        send_control(s)

        remaining = next_t - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        else:
            next_t = time.perf_counter()

# -------------------------
# Routes
# -------------------------
@app.get("/")
def index():
    return render_template_string(HTML, device=DEVICE, w=WIDTH, h=HEIGHT, fps=FPS)

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

@app.post("/control")
def control():
    data = request.get_json(force=True, silent=True) or {}

    def b(name):
        return bool(data.get(name, False))

    with state_lock:
        control_state["up"] = b("up")
        control_state["down"] = b("down")
        control_state["left"] = b("left")
        control_state["right"] = b("right")
        control_state["center"] = b("center")
        control_state["brake"] = b("brake")
        control_state["last_seen"] = time.perf_counter()

    return jsonify(ok=True)

# -------------------------
# Main
# -------------------------
def _handle_exit(signum, frame):
    pwm_safe_exit()
    raise SystemExit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    pwm_init()

    cam_t = threading.Thread(target=camera_worker, daemon=True)
    cam_t.start()

    t = threading.Thread(target=control_loop, daemon=True)
    t.start()

    ips = detect_local_ips()
    if ips:
        print("Open on phone:")
        for ip in ips:
            print(f"  http://{ip}:{PORT}/")
    else:
        print(f"Open on phone: http://<board-ip>:{PORT}/")

    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)
