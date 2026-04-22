#!/usr/bin/env python3

"""
AI Autonomous RC Car
Full System:
- Manual control
- autopilot AI
- mode switch
- start/pause btn
- E-stop
- live pwm display
- Lab color line following
- Road sign decision supervisor
- STOP / GO memory
- PERSON safety
- SLOW throttle control
- U-TURN timed maneuver
"""

from flask import Flask, Response, render_template_string, request, jsonify
import subprocess
import socket
import time
import threading
import signal
import cv2
import numpy as np

# =========================
# CONFIG
# =========================
DEVICE = "/dev/video4"
WIDTH = 320
HEIGHT = 240
FPS = 10
PORT = 6088

# =========================
# AI CONFIG
# =========================
MODEL_PATH = "model.onnx"
AI_ENABLED = True
CONF_THRESHOLD = 0.75
CLASS_NAMES = ["background", "stop", "person", "slow", "Uturn", "go"]

# =========================
# MODE STATE
# =========================
MODE = "MANUAL"     # MANUAL or AUTOPILOT
AUTOPILOT_RUNNING = False
E_STOP = False

# =========================
# PCA9685 CONFIG
# =========================
from smbus2 import SMBus

PCA9685_ADDR = 0x40
PCA9685_FREQ = 60
I2C_BUS = 0

THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

THROTTLE_STOPPED_TICKS = 370
THROTTLE_FORWARD_TICKS = 415
THROTTLE_SLOW_TICKS = 390
THROTTLE_REVERSE_TICKS = 305

STEERING_LEFT_TICKS   = 280
STEERING_CENTER_TICKS = 380
STEERING_RIGHT_TICKS  = 480

STEERING_MIN_TICKS = 305
STEERING_MAX_TICKS = 480

STEP = 5
STEERING_STEP = 25

CONTROL_HZ = 60.0
CONTROL_DT = 1.0 / CONTROL_HZ
FAILSAFE_TIMEOUT_SEC = 0.35

# =========================
# FLASK
# =========================
app = Flask(__name__)

# =========================
# GLOBAL STATE
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

values = {
    "throttle": THROTTLE_STOPPED_TICKS,
    "steering": STEERING_CENTER_TICKS,
}

# =========================
# PWM DRIVER
# =========================
class PCA9685:
    MODE1 = 0x00
    PRESCALE = 0xFE
    LED0_ON_L = 0x06
    RESTART = 0x80
    SLEEP = 0x10

    def __init__(self, busnum, address=0x40, freq=60):
        self.bus = SMBus(busnum)
        self.address = address
        self.set_pwm_freq(freq)

    def write8(self, reg, val):
        self.bus.write_byte_data(self.address, reg, val)

    def read8(self, reg):
        return self.bus.read_byte_data(self.address, reg)

    def set_pwm_freq(self, freq):
        prescaleval = int(25000000.0 / (4096 * freq) - 1)
        oldmode = self.read8(self.MODE1)
        self.write8(self.MODE1, oldmode | self.SLEEP)
        self.write8(self.PRESCALE, prescaleval)
        self.write8(self.MODE1, oldmode)
        time.sleep(0.005)
        self.write8(self.MODE1, oldmode | self.RESTART)

    def set_pwm(self, channel, on, off):
        base = self.LED0_ON_L + 4 * channel
        self.write8(base + 0, on & 0xFF)
        self.write8(base + 1, (on >> 8) & 0xFF)
        self.write8(base + 2, off & 0xFF)
        self.write8(base + 3, (off >> 8) & 0xFF)

    def set_pwm_12bit(self, channel, value):
        value = max(0, min(4095, int(value)))
        self.set_pwm(channel, 0, value)

pwm = None

# =========================
# AI MODEL
# =========================
net = None

# =========================
# CAMERA BUFFER
# =========================
_latest_lock = threading.Lock()
_latest_jpeg = None
_latest_seq = 0
_camera_stop = threading.Event()

# =========================
# CAMERA PIPE
# =========================
def ffmpeg_jpeg_pipe():
    cmd = [
        "ffmpeg",
        "-f", "video4linux2",
        "-input_format", "yuyv422",
        "-framerate", str(FPS),
        "-video_size", f"{WIDTH}x{HEIGHT}",
        "-i", DEVICE,
        "-an",
        "-c:v", "mjpeg",
        "-q:v", "7",
        "-f", "image2pipe",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=0)

def iter_jpegs(stream):
    buf = bytearray()
    while True:
        chunk = stream.read(4096)
        if not chunk:
            return
        buf.extend(chunk)
        while True:
            soi = buf.find(b"\xff\xd8")
            eoi = buf.find(b"\xff\xd9", soi + 2)
            if soi != -1 and eoi != -1:
                jpg = bytes(buf[soi:eoi+2])
                del buf[:eoi+2]
                yield jpg
            else:
                break

# =========================
# CAMERA + AI WORKER
# =========================
def camera_worker():
    global _latest_jpeg, _latest_seq
    frame_count = 0
    last_label = ""

    while not _camera_stop.is_set():
        p = ffmpeg_jpeg_pipe()

        for jpg in iter_jpegs(p.stdout):

            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame_count += 1

            # AUTOPILOT AI
            if (AI_ENABLED and net is not None and
                MODE == "AUTOPILOT" and AUTOPILOT_RUNNING and
                frame_count % 4 == 0 and not E_STOP):

                img = cv2.resize(frame, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=0)

                net.setInput(img)
                output = net.forward()[0]

                class_id = int(np.argmax(output))
                confidence = float(output[class_id])
                label = CLASS_NAMES[class_id]
                last_label = f"{label} ({confidence:.2f})"

                if confidence > CONF_THRESHOLD:

                    if label in ["stop", "person"]:
                        values["throttle"] = THROTTLE_STOPPED_TICKS

                    elif label == "go":
                        values["throttle"] = THROTTLE_FORWARD_TICKS

                    elif label == "slow":
                        values["throttle"] = THROTTLE_SLOW_TICKS

                    elif label == "Uturn":
                        values["steering"] = STEERING_RIGHT_TICKS

            # ===== OVERLAY INFO =====
            cv2.putText(frame, f"Mode: {MODE}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,255), 2)

            cv2.putText(frame,
                        f"Throttle: {values['throttle']}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

            cv2.putText(frame,
                        f"Steering: {values['steering']}",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

            if last_label:
                cv2.putText(frame, last_label,
                            (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255,255,0), 2)

            _, enc = cv2.imencode(".jpg", frame)
            jpg = enc.tobytes()

            with _latest_lock:
                _latest_jpeg = jpg
                _latest_seq += 1

# =========================
# MJPEG ROUTE
# =========================
@app.route("/mjpg")
def mjpg():
    def generate():
        boundary = b"--frame\r\n"
        last_seq = -1
        while True:
            with _latest_lock:
                seq = _latest_seq
                jpg = _latest_jpeg
            if jpg is None or seq == last_seq:
                time.sleep(0.01)
                continue
            last_seq = seq
            yield (boundary +
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   jpg + b"\r\n")

    return Response(generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

# =========================
# WEB UI
# =========================
@app.route("/")
def index():
    return render_template_string("""
    <html>
    <body style="background:black;color:white;text-align:center;">
    <h2>AI CAR CONTROL</h2>
    <img src="/mjpg" width="90%"><br><br>

    <button onclick="fetch('/mode')">Toggle Mode</button>
    <button onclick="fetch('/autopilot/start')">Autopilot Start</button>
    <button onclick="fetch('/autopilot/pause')">Autopilot Pause</button>
    <button style="background:red;color:white;"
            onclick="fetch('/estop')">E-STOP</button>

    </body>
    </html>
    """)

# =========================
# MODE ROUTES
# =========================
@app.route("/mode")
def toggle_mode():
    global MODE
    MODE = "AUTOPILOT" if MODE == "MANUAL" else "MANUAL"
    return "OK"

@app.route("/autopilot/start")
def auto_start():
    global AUTOPILOT_RUNNING
    AUTOPILOT_RUNNING = True
    return "OK"

@app.route("/autopilot/pause")
def auto_pause():
    global AUTOPILOT_RUNNING
    AUTOPILOT_RUNNING = False
    return "OK"

@app.route("/estop")
def estop():
    global E_STOP
    E_STOP = True
    values["throttle"] = THROTTLE_STOPPED_TICKS
    return "STOPPED"

# =========================
# CONTROL LOOP
# =========================
def control_loop():
    while True:
        if pwm is not None:
            pwm.set_pwm_12bit(THROTTLE_CHANNEL, values["throttle"])
            pwm.set_pwm_12bit(STEERING_CHANNEL, values["steering"])
        time.sleep(CONTROL_DT)

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    signal.signal(signal.SIGINT, lambda s,f: exit(0))

    print("Loading AI model...")
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    print("Model loaded")

    pwm = PCA9685(I2C_BUS, PCA9685_ADDR, PCA9685_FREQ)

    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=control_loop, daemon=True).start()

    print(f"Open: http://<board-ip>:{PORT}/")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
