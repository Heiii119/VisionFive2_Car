#!/usr/bin/env python3
from flask import Flask, Response, render_template_string, request, jsonify
import subprocess
import socket
import time
import threading
import signal

# --- NEW: vision deps ---
try:
    import cv2
    import numpy as np
except Exception as e:
    raise SystemExit(
        "Missing OpenCV/NumPy.\n"
        "Install:\n"
        "  sudo apt-get install -y python3-opencv python3-numpy\n"
        f"\nOriginal error: {e}"
    )

# =========================
# Camera / streaming config
# =========================
DEVICE = "/dev/video4"
WIDTH = 320
HEIGHT = 240
FPS = 10
PORT = 7070

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
THROTTLE_FORWARD_TICKS = 410
THROTTLE_REVERSE_TICKS = 310

STEERING_LEFT_TICKS   = 280
STEERING_CENTER_TICKS = 380
STEERING_RIGHT_TICKS  = 480

STEERING_MIN_TICKS = 305
STEERING_MAX_TICKS = 480

START_THROTTLE_TICKS = THROTTLE_STOPPED_TICKS
START_STEERING_TICKS = STEERING_CENTER_TICKS

STOP_ON_EXIT = True

# Manual control behavior (your existing style)
STEP = 5
STEERING_STEP = 25
THROTTLE_RELEASE_TO_STOP = True
STEERING_RELEASE_TO_CENTER = False

# =========================
# Autopilot / Line following settings
# =========================
AUTO_HZ = 20.0
AUTO_DT = 1.0 / AUTO_HZ

# Region of Interest: bottom area only (ignore top 35%)
ROI_Y_START = int(HEIGHT * 0.35)

# Calibration tolerance around picked HSV color
HSV_TOL_H = 15
HSV_TOL_S = 80
HSV_TOL_V = 80

# Steering control (proportional)
AUTO_STEER_GAIN = 1.25

# Decision thresholds (in normalized error)
DECISION_DEADBAND = 0.12

# Autopilot throttle
AUTO_THROTTLE_TICKS = 410

# If line is lost for this long, stop the car
LINE_LOST_TIMEOUT_SEC = 0.35

# If autopilot is enabled, do we ignore manual steering/throttle inputs?
AUTOPILOT_LOCKS_MANUAL = True

# =========================
# Flask app
# =========================
app = Flask(__name__)

# -------------------------
# Manual control state (60Hz loop)
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
# Autopilot state
# -------------------------
auto_lock = threading.Lock()
auto_state = {
    "enabled": False,
    "calibration_armed": False,   # UI pressed "Calibrate" and waiting for tap
    "calibrated": False,
    "hsv_mean": None,             # <-- NEW: [H,S,V] of picked line colour
    "hsv_lower": None,            # list[int,int,int]
    "hsv_upper": None,            # list[int,int,int]
    "decision": "IDLE",
    "error": 0.0,
    "last_line_seen": 0.0,
    "last_update": 0.0,
}

# -------------------------
# Web UI
# -------------------------
HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no" />
  <title>META DOT FPV Drive + Line Follow</title>
  <style>
    :root { --bg:#0b0f14; --panel:rgba(255,255,255,0.08); --panel2:rgba(0,0,0,0.35);
      --text:#e8eef7; --muted:rgba(232,238,247,0.75); --btn:rgba(255,255,255,0.10);
      --btnActive:rgba(76,175,80,0.35); --btnBrake:rgba(244,67,54,0.40); --btnWarn:rgba(255,193,7,0.35); --stroke:rgba(255,255,255,0.14); }
    *{box-sizing:border-box;-webkit-tap-highlight-color:transparent;}

    html,body{
      height:100%;margin:0;background:var(--bg);color:var(--text);
      font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
      -webkit-user-select:none; user-select:none;
      -webkit-touch-callout:none;
    }
    button, .btn, .pad, .hint, header, .meta, .status {
      -webkit-user-select:none; user-select:none; -webkit-touch-callout:none;
    }

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
      width:min(46vw,340px);user-select:none;touch-action:none;backdrop-filter:blur(6px);}
    .row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;} .row.two{grid-template-columns:1fr 1fr;}

    button.btn{
      width:100%;height:64px;border-radius:12px;border:1px solid var(--stroke);
      background:var(--btn);color:var(--text);font-size:16px;font-weight:700;letter-spacing:0.2px;
      touch-action: manipulation;
    }
    button.btn:active{transform:scale(0.99);}
    button.btn.active{background:var(--btnActive);}
    button.btn.brake.active{background:var(--btnBrake);}
    button.btn.warn{background:var(--btnWarn);}

    .hint{pointer-events:none;position:absolute;left:12px;top:56px;font-size:12px;color:var(--muted);background:rgba(0,0,0,0.35);
      border:1px solid var(--stroke);padding:6px 8px;border-radius:10px;}

    .telemetry{
      pointer-events:none;
      position:absolute;
      right:12px; top:56px;
      font-size:12px; color:var(--text);
      background:rgba(0,0,0,0.35);
      border:1px solid var(--stroke);
      padding:8px 10px;
      border-radius:10px;
      min-width: 260px;
    }
    .telemetry .muted{color:var(--muted);}
    .telemetry .big{font-size:14px; font-weight:800; margin-top:2px;}
    .telemetry .kv{display:flex; justify-content:space-between; gap:10px;}
    .telemetry .kv span:last-child{font-variant-numeric: tabular-nums;}

    @media (orientation: portrait){ .hint::after{content:" (rotate phone to landscape)";} }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <div><strong>FPV Drive + Line Follow</strong></div>
        <div class="meta">Device: {{device}} • {{w}}x{{h}} @ {{fps}}fps • Manual loop: 60Hz • Auto: 20Hz</div>
      </div>
      <div id="status" class="status">Connecting…</div>
    </header>

    <div class="main">
      <div class="video">
        <img id="stream" src="/mjpg" alt="stream" />
      </div>

      <div class="hint">Manual: arrows + C + Space • Auto: Calibrate then Start Auto • Tap the line during calibration</div>

      <div class="telemetry">
          <div class="kv"><span class="muted">Mode</span><span id="mode">—</span></div>
          <div class="kv"><span class="muted">Calibrated</span><span id="calib">—</span></div>

          <div class="kv"><span class="muted">Line HSV</span><span id="lineHSV">—</span></div>
          <div class="kv"><span class="muted">HSV range</span><span id="hsvRange">—</span></div>

          <div class="kv"><span class="muted">Throttle ticks</span><span id="throttleTicks">—</span></div>
          <div class="kv"><span class="muted">Steering ticks</span><span id="steeringTicks">—</span></div>

          <div class="kv"><span class="muted">Decision</span><span id="decision">—</span></div>
          <div class="kv"><span class="muted">Error</span><span id="error">—</span></div>
          <div class="big" id="tapHint" style="display:none;">Tap the line in the video…</div>
    </div>

      <div class="controls">
        <div class="cluster left">
          <div class="pad">
            <div class="row">
              <button class="btn" id="left">◀</button>
              <button class="btn" id="center">C</button>
              <button class="btn" id="right">▶</button>
            </div>
            <div class="row two">
              <button class="btn warn" id="calibrate">CALIBRATE</button>
              <button class="btn" id="startAuto">START AUTO</button>
            </div>
            <div class="row two">
              <button class="btn brake" id="eStop">E-STOP</button>
              <button class="btn" id="stopAuto">STOP AUTO</button>
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
  const MANUAL_HZ = 20;
  const MANUAL_INTERVAL_MS = Math.round(1000 / MANUAL_HZ);

  const statusEl = document.getElementById('status');
  const btnIds = ["up","down","left","right","center","brake"];
  const btn = Object.fromEntries(btnIds.map(id => [id, document.getElementById(id)]));

  const streamImg = document.getElementById("stream");

  const modeEl = document.getElementById("mode");
  const calibEl = document.getElementById("calib");
  const decisionEl = document.getElementById("decision");
  const errorEl = document.getElementById("error");
  const tapHintEl = document.getElementById("tapHint");
  const throttleTicksEl = document.getElementById("throttleTicks");
  const steeringTicksEl = document.getElementById("steeringTicks");

  const lineHSVEl = document.getElementById("lineHSV");
  const hsvRangeEl = document.getElementById("hsvRange");

  const state = { up:false, down:false, left:false, right:false, center:false, brake:false };

  let connected = false;
  let lastOk = 0;

  let inFlight = false;
  let dirty = true;
  let lastSentAt = 0;

  function markDirty(){ dirty = true; }

  function setActive(id, on) {
    const v = !!on;
    if (state[id] === v) return;
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

  async function sendManualOnce() {
    const r = await fetch("/control", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(state),
      cache: "no-store",
    });
    if (!r.ok) throw new Error("HTTP " + r.status);
  }

  function updateStatusUI(){
    const now = performance.now();
    const okRecent = connected && (now - lastOk) < 800;
    statusEl.textContent = okRecent ? `Connected` : "Disconnected…";
    statusEl.style.opacity = okRecent ? "1.0" : "0.7";
  }

  async function manualPump(){
    if (inFlight) { setTimeout(manualPump, 5); return; }

    const now = performance.now();
    const due = (now - lastSentAt) >= MANUAL_INTERVAL_MS;

    if (!dirty && !due) { updateStatusUI(); setTimeout(manualPump, 5); return; }

    inFlight = true;
    dirty = false;
    try{
      await sendManualOnce();
      connected = true;
      lastOk = performance.now();
      lastSentAt = lastOk;
    }catch(e){
      connected = false;
    }finally{
      inFlight = false;
      updateStatusUI();
      setTimeout(manualPump, 0);
    }
  }
  manualPump();

  // -------- Autopilot UI --------
  let calibrationArmed = false;

  async function apiPost(url, body){
    const r = await fetch(url, {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: body ? JSON.stringify(body) : "{}",
      cache:"no-store",
    });
    return r.json();
  }

  document.getElementById("calibrate").addEventListener("click", async (e) => {
    e.preventDefault();
    const js = await apiPost("/api/calibrate/start");
    calibrationArmed = !!js.armed;
    tapHintEl.style.display = calibrationArmed ? "block" : "none";
  });

  document.getElementById("startAuto").addEventListener("click", async (e) => {
    e.preventDefault();
    await apiPost("/api/autopilot/start");
  });

  document.getElementById("stopAuto").addEventListener("click", async (e) => {
    e.preventDefault();
    await apiPost("/api/autopilot/stop");
  });

  document.getElementById("eStop").addEventListener("click", async (e) => {
    e.preventDefault();
    await apiPost("/api/emergency_stop");
  });

  // Tap on video to pick the line colour (during calibration)
  streamImg.addEventListener("click", async (e) => {
    if (!calibrationArmed) return;

    const rect = streamImg.getBoundingClientRect();
    const x = Math.round((e.clientX - rect.left) / rect.width * {{w}});
    const y = Math.round((e.clientY - rect.top) / rect.height * {{h}});

    const js = await apiPost("/api/calibrate/pick", {x, y});
    if (js && js.ok) {
      calibrationArmed = false;
      tapHintEl.style.display = "none";
    }
  });

  // Poll status (telemetry)
  async function pollStatus(){
    try{
      const r = await fetch("/api/status", {cache:"no-store"});
      const s = await r.json();

      modeEl.textContent = s.autopilot_enabled ? "AUTO" : "MANUAL";
      calibEl.textContent = s.calibrated ? "YES" : "NO";
      decisionEl.textContent = s.decision || "—";
      errorEl.textContent = (typeof s.error === "number") ? s.error.toFixed(3) : "—";

      throttleTicksEl.textContent =
        (typeof s.throttle_ticks === "number") ? String(s.throttle_ticks) : "—";
      steeringTicksEl.textContent =
        (typeof s.steering_ticks === "number") ? String(s.steering_ticks) : "—";

      if (Array.isArray(s.hsv_mean)) {
        lineHSVEl.textContent = `H:${s.hsv_mean[0]}  S:${s.hsv_mean[1]}  V:${s.hsv_mean[2]}`;
      } else {
        lineHSVEl.textContent = "—";
      }

      if (Array.isArray(s.hsv_lower) && Array.isArray(s.hsv_upper)) {
        hsvRangeEl.textContent = `[${s.hsv_lower.join(", ")}] → [${s.hsv_upper.join(", ")}]`;
      } else {
        hsvRangeEl.textContent = "—";
      }
    }catch(e){
      // ignore
    }finally{
      setTimeout(pollStatus, 200);
    }
  }
  pollStatus();

  // Failsafe: blur => brake
  const failSafe = () => {
    btnIds.forEach(id => setActive(id, false));
    setActive("brake", true);
    setTimeout(() => setActive("brake", false), 250);
  };
  window.addEventListener("blur", failSafe);
  document.addEventListener("visibilitychange", () => { if (document.hidden) failSafe(); });

  document.addEventListener("contextmenu", (e) => e.preventDefault(), { passive: false });
  document.addEventListener("selectstart", (e) => e.preventDefault(), { passive: false });
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
            try: p.kill()
            except Exception: pass
            try: p.wait(timeout=1)
            except Exception: pass
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

def get_latest_bgr_frame():
    with _latest_lock:
        jpg = _latest_jpeg
    if jpg is None:
        return None
    arr = np.frombuffer(jpg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

# =========================
# PCA9685 driver
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
# PWM runtime state
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

def emergency_stop_now():
    with auto_lock:
        auto_state["enabled"] = False
        auto_state["decision"] = "E-STOP"
    with pwm_lock:
        values["throttle"] = clamp12(THROTTLE_STOPPED_TICKS)
        pwm.set_pwm_12bit(THROTTLE_CHANNEL, values["throttle"])

# -------------------------
# Manual control: apply web state to PWM ticks
# -------------------------
def send_control(s: dict):
    if pwm is None:
        return

    with auto_lock:
        auto_enabled = bool(auto_state["enabled"])

    if AUTOPILOT_LOCKS_MANUAL and auto_enabled:
        if bool(s.get("brake", False)):
            emergency_stop_now()
        return

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

        values["throttle"] = max(THROTTLE_REVERSE_TICKS, min(THROTTLE_FORWARD_TICKS, values["throttle"]))

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
# Autopilot: vision + steering
# -------------------------
def hsv_clip(h, s, v):
    return [
        int(max(0, min(179, h))),
        int(max(0, min(255, s))),
        int(max(0, min(255, v))),
    ]

def compute_line_error_and_mask_info(bgr, hsv_lower, hsv_upper):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    roi = mask[ROI_Y_START:HEIGHT, :]

    kernel = np.ones((5, 5), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0, -1, roi.shape[1]

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 200:
        return False, 0.0, -1, roi.shape[1]

    M = cv2.moments(c)
    if M["m00"] == 0:
        return False, 0.0, -1, roi.shape[1]

    cx = int(M["m10"] / M["m00"])
    frame_cx = roi.shape[1] // 2
    error = (cx - frame_cx) / float(frame_cx if frame_cx != 0 else 1)

    return True, float(error), cx, roi.shape[1]

def autopilot_loop():
    next_t = time.perf_counter()
    while True:
        next_t += AUTO_DT

        with auto_lock:
            enabled = bool(auto_state["enabled"])
            calibrated = bool(auto_state["calibrated"])
            hsv_lower = auto_state["hsv_lower"]
            hsv_upper = auto_state["hsv_upper"]

        if not enabled:
            time.sleep(0.05)
            next_t = time.perf_counter()
            continue

        if not calibrated or hsv_lower is None or hsv_upper is None:
            with auto_lock:
                auto_state["decision"] = "NOT CALIBRATED"
                auto_state["error"] = 0.0
            emergency_stop_now()
            time.sleep(0.1)
            continue

        frame = get_latest_bgr_frame()
        if frame is None:
            time.sleep(0.02)
            continue

        found, err, cx, roi_w = compute_line_error_and_mask_info(frame, hsv_lower, hsv_upper)
        now = time.perf_counter()

        if not found:
            with auto_lock:
                last_seen = float(auto_state["last_line_seen"])
                auto_state["decision"] = "LINE LOST"
                auto_state["error"] = 0.0
                auto_state["last_update"] = now

            if (now - last_seen) > LINE_LOST_TIMEOUT_SEC:
                with pwm_lock:
                    values["throttle"] = clamp12(THROTTLE_STOPPED_TICKS)
                    pwm.set_pwm_12bit(THROTTLE_CHANNEL, values["throttle"])
        else:
            with auto_lock:
                auto_state["last_line_seen"] = now
                auto_state["error"] = float(err)
                auto_state["last_update"] = now

            if err > DECISION_DEADBAND:
                decision = "TURN RIGHT"
            elif err < -DECISION_DEADBAND:
                decision = "TURN LEFT"
            else:
                decision = "GO STRAIGHT"

            steer = STEERING_CENTER_TICKS - int(
                AUTO_STEER_GAIN * err * ((STEERING_MAX_TICKS - STEERING_MIN_TICKS) / 2.0)
            )
            steer = clamp_steering(steer)

            with auto_lock:
                auto_state["decision"] = decision

            with pwm_lock:
                values["throttle"] = clamp12(AUTO_THROTTLE_TICKS)
                values["throttle"] = max(THROTTLE_REVERSE_TICKS, min(THROTTLE_FORWARD_TICKS, values["throttle"]))
                values["steering"] = steer

                pwm.set_pwm_12bit(THROTTLE_CHANNEL, values["throttle"])
                pwm.set_pwm_12bit(STEERING_CHANNEL, values["steering"])

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

    if control_state["brake"]:
        emergency_stop_now()

    return jsonify(ok=True)

@app.get("/api/status")
def api_status():
    with auto_lock:
        s = dict(auto_state)

    with pwm_lock:
        throttle_ticks = int(values.get("throttle", THROTTLE_STOPPED_TICKS))
        steering_ticks = int(values.get("steering", STEERING_CENTER_TICKS))

    return jsonify(
        autopilot_enabled=bool(s["enabled"]),
        calibrated=bool(s["calibrated"]),
        decision=str(s["decision"]),
        error=float(s["error"]),
        throttle_ticks=throttle_ticks,
        steering_ticks=steering_ticks,

        # <-- NEW: expose calibration HSV info
        hsv_mean=s.get("hsv_mean"),
        hsv_lower=s.get("hsv_lower"),
        hsv_upper=s.get("hsv_upper"),
    )

@app.post("/api/calibrate/start")
def api_calibrate_start():
    with auto_lock:
        auto_state["calibration_armed"] = True
        auto_state["decision"] = "CALIBRATE: TAP LINE"
    return jsonify(ok=True, armed=True)

@app.post("/api/calibrate/pick")
def api_calibrate_pick():
    data = request.get_json(force=True, silent=True) or {}
    x = int(data.get("x", -1))
    y = int(data.get("y", -1))

    with auto_lock:
        armed = bool(auto_state["calibration_armed"])

    if not armed:
        return jsonify(ok=False, msg="Not armed. Press CALIBRATE first."), 400

    frame = get_latest_bgr_frame()
    if frame is None:
        return jsonify(ok=False, msg="No frame yet."), 503

    x = max(0, min(WIDTH - 1, x))
    y = max(0, min(HEIGHT - 1, y))

    r = 6
    x0, x1 = max(0, x - r), min(WIDTH, x + r + 1)
    y0, y1 = max(0, y - r), min(HEIGHT, y + r + 1)
    patch = frame[y0:y1, x0:x1]
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    h = int(np.mean(hsv_patch[:, :, 0]))
    s = int(np.mean(hsv_patch[:, :, 1]))
    v = int(np.mean(hsv_patch[:, :, 2]))

    lower = hsv_clip(h - HSV_TOL_H, s - HSV_TOL_S, v - HSV_TOL_V)
    upper = hsv_clip(h + HSV_TOL_H, s + HSV_TOL_S, v + HSV_TOL_V)

    with auto_lock:
        auto_state["hsv_mean"] = [h, s, v]     # <-- NEW: store picked HSV center
        auto_state["hsv_lower"] = lower
        auto_state["hsv_upper"] = upper
        auto_state["calibrated"] = True
        auto_state["calibration_armed"] = False
        auto_state["decision"] = "CALIBRATED"

    return jsonify(ok=True, hsv_mean=[h, s, v], hsv_lower=lower, hsv_upper=upper)

@app.post("/api/autopilot/start")
def api_autopilot_start():
    with auto_lock:
        if not auto_state["calibrated"]:
            auto_state["decision"] = "CALIBRATE FIRST"
            return jsonify(ok=False, msg="Calibrate first."), 400
        auto_state["enabled"] = True
        auto_state["decision"] = "AUTO START"
        auto_state["last_line_seen"] = time.perf_counter()
    return jsonify(ok=True)

@app.post("/api/autopilot/stop")
def api_autopilot_stop():
    with auto_lock:
        auto_state["enabled"] = False
        auto_state["decision"] = "AUTO STOP"
    with pwm_lock:
        values["throttle"] = clamp12(THROTTLE_STOPPED_TICKS)
        pwm.set_pwm_12bit(THROTTLE_CHANNEL, values["throttle"])
    return jsonify(ok=True)

@app.post("/api/emergency_stop")
def api_emergency_stop():
    emergency_stop_now()
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

    t_manual = threading.Thread(target=control_loop, daemon=True)
    t_manual.start()

    t_auto = threading.Thread(target=autopilot_loop, daemon=True)
    t_auto.start()

    ips = detect_local_ips()
    if ips:
        print("Open on phone:")
        for ip in ips:
            print(f"  http://{ip}:{PORT}/")
    else:
        print(f"Open on phone: http://<board-ip>:{PORT}/")

    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)
