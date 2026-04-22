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

from flask import Flask, Response, render_template_string, jsonify, request
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

THROTTLE_STOPPED = 370
THROTTLE_FORWARD = 415
THROTTLE_SLOW = 405
THROTTLE_REVERSE = 305

STEERING_CENTER = 380
STEERING_MIN = 305
STEERING_MAX = 480

CONTROL_HZ = 60.0
CONTROL_DT = 1.0 / CONTROL_HZ

LINE_THRESHOLD = 100  # will be calibrated

app = Flask(__name__)

values = {
    "throttle": THROTTLE_STOPPED,
    "steering": STEERING_CENTER,
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

    def set_pwm_12bit(self, channel, value):
        value = max(0, min(4095, int(value)))
        base = 0x06 + 4 * channel
        self.write8(base + 0, 0)
        self.write8(base + 1, 0)
        self.write8(base + 2, value & 0xFF)
        self.write8(base + 3, (value >> 8) & 0xFF)

pwm = None
net = None

# =========================
# CAMERA BUFFER
# =========================
_latest_lock = threading.Lock()
_latest_frame = None
_latest_jpeg = None
_latest_seq = 0

# =========================
# LINE CALIBRATION
# =========================
def calibrate_line(frame):
    global LINE_THRESHOLD
    h, w, _ = frame.shape
    roi = frame[int(h*0.6):h, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    LINE_THRESHOLD = int(np.mean(gray) * 0.8)
    print("Line threshold calibrated:", LINE_THRESHOLD)

def line_follow(frame):
    h, w, _ = frame.shape
    roi = frame[int(h*0.6):h, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, LINE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    moments = cv2.moments(thresh)

    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        error = cx - (w // 2)
        steer = STEERING_CENTER - int(error * 0.3)
        steer = max(STEERING_MIN, min(STEERING_MAX, steer))
        values["steering"] = steer
        values["throttle"] = THROTTLE_SLOW + 5
    else:
        values["throttle"] = THROTTLE_STOPPED

# =========================
# CAMERA THREAD
# =========================
def camera_worker():
    global _latest_frame, _latest_jpeg, _latest_seq
    cap = cv2.VideoCapture(DEVICE)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        _latest_frame = frame.copy()

        if MODE == "AUTOPILOT" and AUTOPILOT_RUNNING and not E_STOP:

            img = cv2.resize(frame, (224,224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)/255.0
            img = np.expand_dims(img, axis=0)

            if net:
                net.setInput(img)
                output = net.forward()[0]
                class_id = int(np.argmax(output))
                confidence = float(output[class_id])
                label = CLASS_NAMES[class_id]

                if confidence > CONF_THRESHOLD:
                    if label in ["stop","person"]:
                        values["throttle"] = THROTTLE_STOPPED
                    elif label == "go":
                        values["throttle"] = THROTTLE_FORWARD
                    elif label == "slow":
                        values["throttle"] = THROTTLE_SLOW
                    elif label == "background":
                        line_follow(frame)
                    elif label == "Uturn":
                        print("U-TURN detected")

                        # Slow down first
                        values["throttle"] = THROTTLE_Slow
                        values["steering"] = STEERING_MAX
                        time.sleep(0.5)   # adjust duration
                        # forwards and turn left 
                        values["throttle"] = THROTTLE_FORWARD
                        values["steering"] = STEERING_MAX
                        time.sleep(3)   # adjust duration
                        # backwards and turn right 
                        values["throttle"] = THROTTLE_BACKWARD + 5
                        values["steering"] = STEERING_MIN
                        time.sleep(3)   # adjust duration
                        # Straighten out
                        values["steering"] = STEERING_CENTER
                        time.sleep(0.3)

        _, enc = cv2.imencode(".jpg", frame)
        with _latest_lock:
            _latest_jpeg = enc.tobytes()
            _latest_seq += 1

# =========================
# WEB ROUTES
# =========================
@app.route("/mjpg")
def mjpg():
    def gen():
        last = -1
        while True:
            with _latest_lock:
                seq = _latest_seq
                jpg = _latest_jpeg
            if jpg is None or seq == last:
                time.sleep(0.01)
                continue
            last = seq
            yield b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"+jpg+b"\r\n"
    return Response(gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

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
    AUTOPILOT_RUNNING = True
    if _latest_frame is not None:
        calibrate_line(_latest_frame)
    return "STARTED"

@app.route("/autopilot/pause")
def auto_pause():
    global AUTOPILOT_RUNNING
    AUTOPILOT_RUNNING = False
    return "PAUSED"

@app.route("/estop")
def estop():
    global E_STOP
    E_STOP = True
    values["throttle"] = THROTTLE_STOPPED
    return "STOPPED"

@app.route("/joystick", methods=["POST"])
def joystick():
    data = request.json
    x = float(data["x"])
    y = float(data["y"])

    steer = STEERING_CENTER + int(x * 80)
    steer = max(STEERING_MIN, min(STEERING_MAX, steer))

    throttle = THROTTLE_STOPPED - int(y * 60)

    values["steering"] = steer
    values["throttle"] = throttle

    return "OK"

# =========================
# UI
# =========================
@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{margin:0;background:black;color:white;text-align:center;}
.video{height:60vh;}
.video img{width:100%;height:100%;object-fit:contain;}
#joy{width:200px;height:200px;margin:auto;background:#222;border-radius:50%;position:relative;}
#stick{width:80px;height:80px;background:#555;border-radius:50%;position:absolute;top:60px;left:60px;}
button{padding:10px;margin:5px;}
</style>
</head>
<body>
<div class="video"><img src="/mjpg"></div>
<button onclick="fetch('/mode')">Toggle Mode</button>
<button onclick="fetch('/autopilot/start')">Autopilot Start</button>
<button onclick="fetch('/autopilot/pause')">Pause</button>
<button onclick="fetch('/estop')">E-STOP</button>
<div id="joy"><div id="stick"></div></div>
<script>
let joy=document.getElementById("joy");
let stick=document.getElementById("stick");

joy.addEventListener("touchmove",e=>{
let rect=joy.getBoundingClientRect();
let x=e.touches[0].clientX-rect.left-100;
let y=e.touches[0].clientY-rect.top-100;
x/=100; y/=100;
fetch("/joystick",{method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({x:x,y:y})});
});
</script>
</body>
</html>
""")

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
    try:
        net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    except:
        net = None

    pwm = PCA9685(I2C_BUS, PCA9685_ADDR, PCA9685_FREQ)

    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=control_loop, daemon=True).start()

    print("Open http://<board-ip>:6088/")
    app.run(host="0.0.0.0", port=PORT)
