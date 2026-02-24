from flask import Flask, Response, render_template_string, request, jsonify
import subprocess
import socket
import time
import threading

# =========================
# Camera / streaming config
# =========================
DEVICE = "/dev/video4"
WIDTH = 640
HEIGHT = 480
FPS = 15
PORT = 5000

app = Flask(__name__)

# =========================
# PCA9685 config
# =========================
PCA9685_ADDR = 0x40
PCA9685_FREQ = 60   # Hz
I2C_BUS      = 0
DRIVER_PREFER = "smbus2"   # "smbus2", "legacy", or "auto"

THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

# TICKS (0..4095) calibration/presets
THROTTLE_STOPPED_TICKS = 370
THROTTLE_FORWARD_TICKS = 385
THROTTLE_REVERSE_TICKS = 330

STEERING_LEFT_TICKS   = 280
STEERING_CENTER_TICKS = 380
STEERING_RIGHT_TICKS  = 480

# Safety limits (ticks)
STEERING_MIN_TICKS = 305
STEERING_MAX_TICKS = 455

THROTTLE_MIN_TICKS = 280
THROTTLE_MAX_TICKS = 450

# Startup outputs (ticks)
START_THROTTLE_TICKS = THROTTLE_STOPPED_TICKS
START_STEERING_TICKS = STEERING_CENTER_TICKS

STOP_ON_EXIT = True

# =========================
# Control feel tuning
# =========================
CONTROL_HZ = 60.0
CONTROL_DT = 1.0 / CONTROL_HZ

# Output smoothing: how fast outputs can change (ticks/sec)
THROTTLE_RAMP_TICKS_PER_SEC = 400.0
STEERING_RAMP_TICKS_PER_SEC = 900.0

# Throttle STEP increments (tap/hold)
THROTTLE_STEP_TICKS = 5
THROTTLE_REPEAT_DELAY_SEC = 0.18
THROTTLE_REPEAT_RATE_HZ = 12.0

# Steering hold targets (clamped)
TARGET_STEER_LEFT  = STEERING_LEFT_TICKS
TARGET_STEER_RIGHT = STEERING_RIGHT_TICKS

# Safety: if phone stops sending => brake + center
FAILSAFE_TIMEOUT_SEC = 0.35

# =========================
# Web UI
# =========================
HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no" />
  <title>FPV Drive</title>
  <style>
    :root {
      --bg: #0b0f14;
      --panel: rgba(255,255,255,0.08);
      --panel2: rgba(0,0,0,0.35);
      --text: #e8eef7;
      --muted: rgba(232,238,247,0.75);
      --btn: rgba(255,255,255,0.10);
      --btnActive: rgba(76, 175, 80, 0.35);
      --btnBrake: rgba(244, 67, 54, 0.40);
      --stroke: rgba(255,255,255,0.14);
    }
    * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
    html, body { height: 100%; margin: 0; background: var(--bg); color: var(--text);
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
    .wrap { height: 100%; display: grid; grid-template-rows: auto 1fr; }
    header {
      padding: 10px 12px; display: flex; gap: 12px; align-items: center;
      justify-content: space-between; border-bottom: 1px solid var(--stroke);
      background: linear-gradient(to bottom, rgba(255,255,255,0.06), rgba(255,255,255,0.00));
    }
    header .meta { font-size: 12px; color: var(--muted); }
    header .status {
      font-size: 12px; padding: 6px 10px; border: 1px solid var(--stroke);
      border-radius: 999px; background: var(--panel);
    }
    .main { position: relative; overflow: hidden; }
    .video { position: absolute; inset: 0; display: grid; place-items: center; background: #000; }
    .video img { width: 100%; height: 100%; object-fit: contain; background: #000; }

    .controls {
      position: absolute; inset: 0; padding: 10px;
      display: grid; grid-template-columns: 1fr 1fr; pointer-events: none;
    }
    .cluster { pointer-events: none; display: grid; align-content: end; gap: 10px; }
    .cluster.left { justify-items: start; }
    .cluster.right { justify-items: end; }

    .pad {
      pointer-events: auto;
      background: var(--panel2); border: 1px solid var(--stroke);
      border-radius: 14px; padding: 10px;
      display: grid; gap: 10px; width: min(46vw, 320px);
      user-select: none; touch-action: none; backdrop-filter: blur(6px);
    }
    .row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
    .row.two { grid-template-columns: 1fr 1fr; }

    button.btn {
      width: 100%; height: 64px; border-radius: 12px;
      border: 1px solid var(--stroke); background: var(--btn); color: var(--text);
      font-size: 16px; font-weight: 700;
    }
    button.btn.active { background: var(--btnActive); }
    button.btn.brake.active { background: var(--btnBrake); }

    .hint {
      pointer-events: none; position: absolute; left: 12px; top: 56px;
      font-size: 12px; color: var(--muted);
      background: rgba(0,0,0,0.35); border: 1px solid var(--stroke);
      padding: 6px 8px; border-radius: 10px;
    }
    @media (orientation: portrait) { .hint::after { content: " (rotate phone to landscape)"; } }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <div><strong>FPV Drive</strong></div>
        <div class="meta">Cam: {{device}} • {{w}}x{{h}} @ {{fps}}fps • Control: 60Hz • PCA9685: 60Hz</div>
      </div>
      <div id="status" class="status">Connecting…</div>
    </header>

    <div class="main">
      <div class="video"><img id="stream" src="/mjpg" alt="stream" /></div>
      <div class="hint">Left/Right + Center(C) on left • Up/Down + Brake(Space) on right</div>

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
  const HZ = 60;
  const INTERVAL_MS = Math.round(1000 / HZ);

  const statusEl = document.getElementById('status');
  const btnIds = ["up","down","left","right","center","brake"];
  const btn = Object.fromEntries(btnIds.map(id => [id, document.getElementById(id)]));

  const state = { up:false, down:false, left:false, right:false, center:false, brake:false };
  let lastOk = 0;

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

  async function send() {
    try {
      const r = await fetch("/control", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(state),
        cache: "no-store",
      });
      if (!r.ok) throw new Error("HTTP " + r.status);
      lastOk = performance.now();
    } catch (e) {}

    const okRecent = (performance.now() - lastOk) < 500;
    statusEl.textContent = okRecent ? "Connected (60Hz)" : "Disconnected…";
    statusEl.style.opacity = okRecent ? "1.0" : "0.7";
  }

  setInterval(send, INTERVAL_MS);

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

# =========================
# PCA9685 driver (smbus2 + legacy fallback)
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
        return self._drv.frequency if self._mode == "smbus2" else getattr(self._drv, "frequency", PCA9685_FREQ)

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
# Shared control state
# =========================
state_lock = threading.Lock()
control_state = {
    "up": False, "down": False, "left": False, "right": False,
    "center": False, "brake": False,
    "last_seen": 0.0,
}

out_lock = threading.Lock()
current_throttle = float(START_THROTTLE_TICKS)   # actual output after ramp
current_steering = float(START_STEERING_TICKS)   # actual output after ramp

command_throttle = int(START_THROTTLE_TICKS)     # step-based target (ticks)

pwm = None  # initialized in main


def clamp12(v):
    return max(0, min(4095, int(v)))

def clamp_steering(v):
    return max(STEERING_MIN_TICKS, min(STEERING_MAX_TICKS, int(v)))

def clamp_throttle(v):
    return max(THROTTLE_MIN_TICKS, min(THROTTLE_MAX_TICKS, int(v)))

def ramp_toward(cur, target, max_delta):
    if cur < target:
        return min(target, cur + max_delta)
    if cur > target:
        return max(target, cur - max_delta)
    return cur

def apply_outputs(thr_ticks, steer_ticks):
    pwm.set_pwm_12bit(THROTTLE_CHANNEL, clamp_throttle(thr_ticks))
    pwm.set_pwm_12bit(STEERING_CHANNEL, clamp_steering(steer_ticks))


def control_loop():
    global current_throttle, current_steering, command_throttle

    next_t = time.perf_counter()

    # For step-repeat timing
    up_held_prev = False
    down_held_prev = False
    next_repeat_up = 0.0
    next_repeat_down = 0.0
    repeat_period = 1.0 / float(THROTTLE_REPEAT_RATE_HZ)

    # Startup safe outputs
    command_throttle = clamp_throttle(command_throttle)
    current_throttle = float(command_throttle)
    current_steering = float(clamp_steering(current_steering))
    apply_outputs(int(round(current_throttle)), int(round(current_steering)))

    while True:
        next_t += CONTROL_DT
        now = time.perf_counter()

        with state_lock:
            s = dict(control_state)

        # Failsafe if client not sending
        if (now - s["last_seen"]) > FAILSAFE_TIMEOUT_SEC:
            s = {"up": False, "down": False, "left": False, "right": False,
                 "center": True, "brake": True, "last_seen": s["last_seen"]}

        # -------------------------
        # THROTTLE: STEP increments
        # -------------------------
        if s["brake"]:
            command_throttle = THROTTLE_STOPPED_TICKS
            up_held_prev = False
            down_held_prev = False
            next_repeat_up = 0.0
            next_repeat_down = 0.0
        else:
            up = bool(s["up"])
            down = bool(s["down"])

            if up and down:
                # do nothing if both held
                pass
            else:
                # initial press: one step immediately
                if up and not up_held_prev:
                    command_throttle = clamp_throttle(command_throttle + THROTTLE_STEP_TICKS)
                    next_repeat_up = now + THROTTLE_REPEAT_DELAY_SEC
                if down and not down_held_prev:
                    command_throttle = clamp_throttle(command_throttle - THROTTLE_STEP_TICKS)
                    next_repeat_down = now + THROTTLE_REPEAT_DELAY_SEC

                # repeat while held
                if up and (now >= next_repeat_up):
                    command_throttle = clamp_throttle(command_throttle + THROTTLE_STEP_TICKS)
                    next_repeat_up = now + repeat_period
                if down and (now >= next_repeat_down):
                    command_throttle = clamp_throttle(command_throttle - THROTTLE_STEP_TICKS)
                    next_repeat_down = now + repeat_period

            up_held_prev = up
            down_held_prev = down

        command_throttle = clamp_throttle(command_throttle)

        # -------------------------
        # STEERING: hold-to-angle + center
        # -------------------------
        if s["center"]:
            target_steering = STEERING_CENTER_TICKS
        else:
            if s["left"] and not s["right"]:
                target_steering = TARGET_STEER_LEFT
            elif s["right"] and not s["left"]:
                target_steering = TARGET_STEER_RIGHT
            else:
                target_steering = STEERING_CENTER_TICKS
        target_steering = clamp_steering(target_steering)

        # -------------------------
        # Ramping
        # -------------------------
        max_thr_delta = THROTTLE_RAMP_TICKS_PER_SEC * CONTROL_DT
        max_ste_delta = STEERING_RAMP_TICKS_PER_SEC * CONTROL_DT

        with out_lock:
            current_throttle = ramp_toward(current_throttle, float(command_throttle), max_thr_delta)
            current_steering = ramp_toward(current_steering, float(target_steering), max_ste_delta)
            thr = int(round(current_throttle))
            ste = int(round(current_steering))

        try:
            apply_outputs(thr, ste)
        except Exception:
            pass

        remaining = next_t - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        else:
            next_t = time.perf_counter()


# =========================
# Camera streaming
# =========================
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
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

def multipart_mjpeg_generator():
    p = ffmpeg_jpeg_pipe()
    boundary = b"--frame\r\n"

    def read_one_jpeg(stream):
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
# Flask routes
# =========================
@app.get("/")
def index():
    return render_template_string(HTML, device=DEVICE, w=WIDTH, h=HEIGHT, fps=FPS)

@app.get("/mjpg")
def mjpg_route():
    return Response(
        multipart_mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.post("/control")
def control_route():
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

@app.get("/debug")
def debug_route():
    with state_lock:
        s = dict(control_state)
    with out_lock:
        o = {
            "command_throttle": int(command_throttle),
            "throttle_out": int(round(current_throttle)),
            "steering_out": int(round(current_steering)),
        }
    return jsonify(state=s, outputs=o)


# =========================
# Utilities / main
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

def safe_stop():
    global pwm
    if pwm is None:
        return
    try:
        pwm.set_pwm_12bit(THROTTLE_CHANNEL, clamp_throttle(THROTTLE_STOPPED_TICKS))
        pwm.set_pwm_12bit(STEERING_CHANNEL, clamp_steering(STEERING_CENTER_TICKS))
    except Exception:
        pass
    try:
        pwm.close()
    except Exception:
        pass


if __name__ == "__main__":
    pwm = PCA9685Driver(
        address=PCA9685_ADDR,
        busnum=I2C_BUS,
        frequency=PCA9685_FREQ,
        prefer=DRIVER_PREFER,
    )

    # Start control loop thread
    threading.Thread(target=control_loop, daemon=True).start()

    print("FPV + Drive server starting")
    print(f"Camera: {DEVICE} -> FFmpeg MJPEG")
    for ip in detect_local_ips():
        print(f"Open on phone: http://{ip}:{PORT}/")
    print("Debug JSON: /debug")

    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True)
    finally:
        if STOP_ON_EXIT:
            safe_stop()
