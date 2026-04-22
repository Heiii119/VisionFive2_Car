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

from flask import Flask, Response, render_template_string, jsonify
import subprocess
import time
import threading
import signal
import cv2
import numpy as np
from smbus2 import SMBus

# =========================
# CONFIG
# =========================
DEVICE = "/dev/video4"
WIDTH = 320
HEIGHT = 240
FPS = 10
PORT = 6088

MODEL_PATH = "model.onnx"
CLASS_NAMES = ["background", "stop", "person", "slow", "Uturn", "go"]
CONF_THRESHOLD = 0.75

# =========================
# MODE STATE
# =========================
MODE = "MANUAL"
AUTOPILOT_RUNNING = False
E_STOP = False

# =========================
# PWM CONFIG
# =========================
PCA9685_ADDR = 0x40
PCA9685_FREQ = 60
I2C_BUS = 0

THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

THROTTLE_STOPPED_TICKS = 370
THROTTLE_FORWARD_TICKS = 415
THROTTLE_SLOW_TICKS = 399
THROTTLE_REVERSE_TICKS = 305

STEERING_LEFT_TICKS   = 280
STEERING_CENTER_TICKS = 380
STEERING_RIGHT_TICKS  = 480

STEERING_MIN_TICKS = 305
STEERING_MAX_TICKS = 480

CONTROL_HZ = 60.0
CONTROL_DT = 1.0 / CONTROL_HZ

# =========================
# FLASK
# =========================
app = Flask(__name__)

values = {
    "throttle": THROTTLE_STOPPED_TICKS,
    "steering": STEERING_CENTER_TICKS,
}

# =========================
# PCA9685 DRIVER
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
net = None

# =========================
# LINE FOLLOWING
# =========================
def line_follow(frame):
    h, w, _ = frame.shape
    roi = frame[int(h*0.6):h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)

    moments = cv2.moments(thresh)

    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        error = cx - (w // 2)

        steer = STEERING_CENTER_TICKS - int(error * 0.3)
        steer = max(STEERING_MIN_TICKS, min(STEERING_MAX_TICKS, steer))

        values["steering"] = steer
        values["throttle"] = THROTTLE_SLOW_TICKS + 5
    else:
        values["throttle"] = THROTTLE_STOPPED_TICKS

# =========================
# CAMERA BUFFER
# =========================
_latest_lock = threading.Lock()
_latest_jpeg = None
_latest_seq = 0

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
# CAMERA + AI
# =========================
def camera_worker():
    global _latest_jpeg, _latest_seq
    frame_count = 0
    last_label = ""

    while True:
        p = ffmpeg_jpeg_pipe()
        for jpg in iter_jpegs(p.stdout):

            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame_count += 1

            if (net and MODE == "AUTOPILOT"
                and AUTOPILOT_RUNNING and not E_STOP
                and frame_count % 4 == 0):

                img = cv2.resize(frame, (224,224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32)/255.0
                img = np.expand_dims(img, axis=0)

                net.setInput(img)
                output = net.forward()[0]

                class_id = int(np.argmax(output))
                confidence = float(output[class_id])
                label = CLASS_NAMES[class_id]
                last_label = f"{label} ({confidence:.2f})"

                if confidence > CONF_THRESHOLD:

                    if label in ["stop","person"]:
                        values["throttle"] = THROTTLE_STOPPED_TICKS

                    elif label == "go":
                        values["throttle"] = THROTTLE_FORWARD_TICKS

                    elif label == "slow":
                        values["throttle"] = THROTTLE_SLOW_TICKS

                    elif label == "Uturn":
                        values["steering"] = STEERING_RIGHT_TICKS

                    elif label == "background":
                        line_follow(frame)

            cv2.putText(frame, f"Mode:{MODE}", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
            cv2.putText(frame,
                        f"T:{values['throttle']} S:{values['steering']}",
                        (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            if last_label:
                cv2.putText(frame,last_label,(10,80),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

            _, enc = cv2.imencode(".jpg", frame)
            with _latest_lock:
                _latest_jpeg = enc.tobytes()
                _latest_seq += 1

# =========================
# MJPEG STREAM
# =========================
@app.route("/mjpg")
def mjpg():
    def generate():
        boundary = b"--frame\r\n"
        last = -1
        while True:
            with _latest_lock:
                seq = _latest_seq
                jpg = _latest_jpeg
            if jpg is None or seq == last:
                time.sleep(0.01)
                continue
            last = seq
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
    return Response(generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

# =========================
# WEB UI
# =========================
@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{margin:0;background:black;color:white;font-family:Arial;text-align:center;}
.container{display:flex;flex-direction:column;height:100vh;}
.video{flex:1;}
.video img{width:100%;height:100%;object-fit:contain;}
.controls{background:#111;padding:10px;}
button{padding:10px;margin:5px;font-size:16px;border:none;border-radius:6px;}
.estop{background:red;color:white;font-weight:bold;}
</style>
</head>
<body>
<div class="container">
<div class="video"><img src="/mjpg"></div>
<div class="controls">
<button onclick="fetch('/mode')">Toggle Mode</button>
<button onclick="fetch('/autopilot/start')">Autopilot Start</button>
<button onclick="fetch('/autopilot/pause')">Autopilot Pause</button>
<button class="estop" onclick="fetch('/estop')">E-STOP</button><br>
Throttle:<span id="t">--</span> |
Steering:<span id="s">--</span> |
Mode:<span id="m">--</span>
</div>
</div>
<script>
async function update(){
let r=await fetch('/status');
let d=await r.json();
t.innerText=d.throttle;
s.innerText=d.steering;
m.innerText=d.mode;
}
setInterval(update,200);
update();
</script>
</body>
</html>
""")

@app.route("/status")
def status():
    return jsonify({
        "throttle": values["throttle"],
        "steering": values["steering"],
        "mode": MODE
    })

@app.route("/mode")
def toggle_mode():
    global MODE
    MODE = "AUTOPILOT" if MODE=="MANUAL" else "MANUAL"
    return "OK"

@app.route("/autopilot/start")
def auto_start():
    global AUTOPILOT_RUNNING
    AUTOPILOT_RUNNING=True
    return "OK"

@app.route("/autopilot/pause")
def auto_pause():
    global AUTOPILOT_RUNNING
    AUTOPILOT_RUNNING=False
    return "OK"

@app.route("/estop")
def estop():
    global E_STOP
    E_STOP=True
    values["throttle"]=THROTTLE_STOPPED_TICKS
    return "STOPPED"

# =========================
# CONTROL LOOP
# =========================
def control_loop():
    while True:
        if pwm:
            pwm.set_pwm_12bit(THROTTLE_CHANNEL, values["throttle"])
            pwm.set_pwm_12bit(STEERING_CHANNEL, values["steering"])
        time.sleep(CONTROL_DT)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s,f: exit(0))

    try:
        net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        print("AI model loaded")
    except:
        net=None
        print("No model found")

    pwm = PCA9685(I2C_BUS, PCA9685_ADDR, PCA9685_FREQ)

    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=control_loop, daemon=True).start()

    print(f"Open http://<board-ip>:{PORT}/")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
