#!/usr/bin/env python3

"""
AI Autonomous RC Car
Full System:
- Manual control
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

# =========================
# Vision deps
# =========================
import cv2
import numpy as np

# =========================
# CONFIG
# =========================
DEVICE = "/dev/video4"
WIDTH = 320
HEIGHT = 240
FPS = 6
PORT = 8180

# =========================
# PCA9685 CONFIG
# =========================
PCA9685_ADDR = 0x40
PCA9685_FREQ = 60
I2C_BUS = 0

THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

THROTTLE_STOPPED_TICKS = 370
THROTTLE_FORWARD_TICKS = 410
THROTTLE_REVERSE_TICKS = 310

STEERING_LEFT_TICKS = 280
STEERING_CENTER_TICKS = 380
STEERING_RIGHT_TICKS = 480

STEERING_MIN_TICKS = 305
STEERING_MAX_TICKS = 480

AUTO_THROTTLE_TICKS = 405
AUTO_SLOW_TICKS = 390

AUTO_HZ = 20
AUTO_DT = 1.0 / AUTO_HZ

ROI_Y_START = int(HEIGHT * 0.35)

LAB_TOL_L = 40
LAB_TOL_A = 20
LAB_TOL_B = 20

AUTO_STEER_GAIN = 1.25

# =========================
# Flask
# =========================
app = Flask(__name__)

# =========================
# Shared States
# =========================
auto_lock = threading.Lock()
sign_lock = threading.Lock()
pwm_lock = threading.Lock()

auto_state = {
    "enabled": False,
    "calibrated": False,
    "lab_lower": None,
    "lab_upper": None,
    "decision": "IDLE",
    "error": 0.0,
}

sign_state = {
    "label": "none",
    "confidence": 0.0,
    "last_seen": 0.0,
}

# Latching flags
stop_latched = False
person_latched = False
uturn_active = False
uturn_stage = 0
uturn_timer = 0.0

# PWM runtime values
values = {
    "throttle": THROTTLE_STOPPED_TICKS,
    "steering": STEERING_CENTER_TICKS,
}

# =========================
# PCA9685 (Minimal SMBus2)
# =========================
from smbus2 import SMBus

class PCA9685:
    MODE1 = 0x00
    PRESCALE = 0xFE
    LED0_ON_L = 0x06

    def __init__(self, bus, address=0x40, freq=60):
        self.bus = SMBus(bus)
        self.address = address
        self.set_pwm_freq(freq)

    def write8(self, reg, val):
        self.bus.write_byte_data(self.address, reg, val)

    def set_pwm_freq(self, freq):
        prescale = int(round(25000000.0 / (4096 * freq) - 1))
        self.write8(self.MODE1, 0x10)
        self.write8(self.PRESCALE, prescale)
        self.write8(self.MODE1, 0x80)

    def set_pwm(self, channel, value):
        reg = self.LED0_ON_L + 4 * channel
        self.write8(reg, 0)
        self.write8(reg + 1, 0)
        self.write8(reg + 2, value & 0xFF)
        self.write8(reg + 3, value >> 8)

pwm = PCA9685(I2C_BUS, PCA9685_ADDR, PCA9685_FREQ)

# =========================
# Utility
# =========================
def clamp_steer(v):
    return max(STEERING_MIN_TICKS, min(STEERING_MAX_TICKS, int(v)))

def get_latest_frame():
    cap = cv2.VideoCapture(DEVICE)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.resize(frame, (WIDTH, HEIGHT))

# =========================
# SIGN UPDATE FUNCTION
# =========================
def update_sign(label, confidence):
    global stop_latched, person_latched

    label = label.lower()

    with sign_lock:
        sign_state["label"] = label
        sign_state["confidence"] = confidence
        sign_state["last_seen"] = time.perf_counter()

    if label == "stop":
        stop_latched = True

    if label == "go":
        stop_latched = False

    if label == "person":
        person_latched = True
    else:
        person_latched = False

# =========================
# LINE DETECTION
# =========================
def compute_line_error(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lower = np.array(auto_state["lab_lower"], dtype=np.uint8)
    upper = np.array(auto_state["lab_upper"], dtype=np.uint8)
    mask = cv2.inRange(lab, lower, upper)

    roi = mask[ROI_Y_START:HEIGHT, :]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, 0

    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return False, 0

    cx = int(M["m10"] / M["m00"])
    error = (cx - WIDTH//2) / (WIDTH//2)
    return True, error

# =========================
# AUTOPILOT LOOP
# =========================
def autopilot_loop():
    global stop_latched, person_latched
    global uturn_active, uturn_stage, uturn_timer

    while True:
        time.sleep(AUTO_DT)

        with auto_lock:
            if not auto_state["enabled"] or not auto_state["calibrated"]:
                continue

        frame = get_latest_frame()
        if frame is None:
            continue

        with sign_lock:
            current_sign = sign_state["label"]

        # PERSON
        if person_latched:
            auto_state["decision"] = "PERSON STOP"
            pwm.set_pwm(THROTTLE_CHANNEL, THROTTLE_STOPPED_TICKS)
            continue

        # STOP
        if stop_latched:
            auto_state["decision"] = "STOPPED"
            pwm.set_pwm(THROTTLE_CHANNEL, THROTTLE_STOPPED_TICKS)
            continue

        # UTURN
        if current_sign == "uturn" and not uturn_active:
            uturn_active = True
            uturn_stage = 1
            uturn_timer = time.perf_counter()

        if uturn_active:
            now = time.perf_counter()

            if uturn_stage == 1:
                pwm.set_pwm(STEERING_CHANNEL, STEERING_RIGHT_TICKS)
                pwm.set_pwm(THROTTLE_CHANNEL, AUTO_THROTTLE_TICKS)
                auto_state["decision"] = "UTURN RIGHT"
                if now - uturn_timer > 5:
                    uturn_stage = 2
                    uturn_timer = now

            elif uturn_stage == 2:
                pwm.set_pwm(STEERING_CHANNEL, STEERING_LEFT_TICKS)
                auto_state["decision"] = "UTURN LEFT"
                if now - uturn_timer > 5:
                    uturn_stage = 3

            elif uturn_stage == 3:
                pwm.set_pwm(STEERING_CHANNEL, STEERING_CENTER_TICKS)
                uturn_active = False

            continue

        # SLOW
        throttle = AUTO_SLOW_TICKS if current_sign == "slow" else AUTO_THROTTLE_TICKS

        # LINE FOLLOW
        found, error = compute_line_error(frame)
        if not found:
            pwm.set_pwm(THROTTLE_CHANNEL, THROTTLE_STOPPED_TICKS)
            continue

        steer = STEERING_CENTER_TICKS - int(error * 80)
        steer = clamp_steer(steer)

        pwm.set_pwm(STEERING_CHANNEL, steer)
        pwm.set_pwm(THROTTLE_CHANNEL, throttle)

        auto_state["decision"] = "LINE FOLLOW"
        auto_state["error"] = error

# =========================
# ROUTES
# =========================
@app.route("/api/autopilot/start", methods=["POST"])
def start_auto():
    auto_state["enabled"] = True
    return jsonify(ok=True)

@app.route("/api/autopilot/stop", methods=["POST"])
def stop_auto():
    auto_state["enabled"] = False
    pwm.set_pwm(THROTTLE_CHANNEL, THROTTLE_STOPPED_TICKS)
    return jsonify(ok=True)

@app.route("/api/status")
def status():
    return jsonify(auto_state)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    threading.Thread(target=autopilot_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)
