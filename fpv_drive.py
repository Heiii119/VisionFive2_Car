from flask import Flask, Response, render_template_string
from flask_socketio import SocketIO
import subprocess
import socket
import time
import threading

# -------------------------
# Camera / streaming config
# -------------------------
DEVICE = "/dev/video4"
WIDTH = 640
HEIGHT = 480
FPS = 15
PORT = 5000

app = Flask(__name__)
app.config["SECRET_KEY"] = "fpv"

# WebSocket server
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

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
# Web UI (mobile landscape)
# -------------------------
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
      padding:10px; display:grid; gap:10px; width:min(46vw, 320px); user-select:none; touch-action:none; backdrop-filter:blur(6px); }
    .row { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; }
    .row.two { grid-template-columns:1fr 1fr; }
    button.btn { width:100%; height:64px; border-radius:12px; border:1px solid var(--stroke); background:var(--btn);
      color:var(--text); font-size:16px; font-weight:700; letter-spacing:0.2px; }
    button.btn:active { transform:scale(0.99); }
    button.btn.active { background:var(--btnActive); }
    button.btn.brake.active { background:var(--btnBrake); }
    .hint { pointer-events:none; position:absolute; left:12px; top:56px; font-size:12px; color:var(--muted);
      background:rgba(0,0,0,0.35); border:1px solid var(--stroke); padding:6px 8px; border-radius:10px; }
    @media (orientation: portrait) { .hint::after { content:" (rotate phone to landscape)"; } }
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
      <div class="video">
        <img id="stream" src="/mjpg" alt="stream" />
      </div>

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

  // Send at 60Hz over websocket (no request queue)
  setInterval(() => {
    if (socket.connected) socket.emit("control", state);
  }, INTERVAL_MS);

  // Safety: on blur, brake briefly
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
# Camera: ffmpeg -> jpeg pipe -> multipart MJPEG
# -------------------------
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

# -------------------------
# Car control hook (YOU implement)
# -------------------------
def send_control(state: dict):
    # Implement your real motor/servo output here.
    pass

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
    return Response(
        multipart_mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

# -------------------------
# WebSocket events
# -------------------------
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

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    t = threading.Thread(target=control_loop, daemon=True)
    t.start()

    print(f"FPV stream: {DEVICE} (YUYV) -> FFmpeg MJPEG -> Browser")
    print(f"Open on phone: http://<board-ip>:{PORT}/")
    ips = detect_local_ips()
    for ip in ips:
        print(f"  http://{ip}:{PORT}/")

    # IMPORTANT: use socketio.run (not app.run)
    socketio.run(app, host="0.0.0.0", port=PORT)
