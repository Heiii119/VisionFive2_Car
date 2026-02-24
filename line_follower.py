#!/usr/bin/env python3
import time
import sys
import signal
from collections import deque

import cv2
import numpy as np

# =========================
# Camera config
# =========================
DEVICE = "/dev/video4"
WIDTH = 1280
HEIGHT = 720
FPS = 30

# =========================
# PCA9685 config (as given)
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

START_THROTTLE_TICKS = THROTTLE_STOPPED_TICKS
START_STEERING_TICKS = STEERING_CENTER_TICKS

STOP_ON_EXIT = True

# =========================
# Line-follow tuning
# =========================
# Region-of-interest: use bottom part of image for tracking
ROI_Y_START = 0.55  # start at 55% down from top

# How wide is the calibration patch around image center (in ROI coords)
CAL_PATCH_W = 120
CAL_PATCH_H = 90

# HSV threshold margins around calibrated line color (supports many colors)
H_MARGIN = 12            # hue margin (0..179)
S_MIN = 60               # minimum saturation
V_MIN = 60               # minimum value/brightness

# Morphology / filtering
MORPH_K = 5
MIN_CONTOUR_AREA = 900   # pixels, tune if needed

# Control loop
CONTROL_HZ = 30.0
DT = 1.0 / CONTROL_HZ

# Steering control: simple P controller (error is normalized [-1..1])
KP_STEER = 120.0          # ticks per unit error (tune)
STEER_SMOOTH_ALPHA = 0.35 # 0..1 higher = more responsive, less smooth

# Throttle while following
FOLLOW_THROTTLE_TICKS = 382   # should be inside min/max; adjust for your car

# If line lost, what to do
LOST_LINE_BRAKE = True
LOST_LINE_TIMEOUT_SEC = 0.6   # brake/center if lost longer than this

# Optional: smooth centroid estimate a bit
CENTROID_HISTORY = 5

# =========================
# PCA9685 driver (same style as your Flask file)
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

def open_camera():
    # V4L2 backend usually best on Linux
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {DEVICE}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Try to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def roi_crop(frame_bgr):
    h, w = frame_bgr.shape[:2]
    y0 = int(h * ROI_Y_START)
    return frame_bgr[y0:h, 0:w], y0

def circular_hue_bounds(h_center, margin):
    # Hue range in OpenCV: 0..179
    lo = int(h_center - margin)
    hi = int(h_center + margin)
    if lo < 0:
        return [(0, hi), (180 + lo, 179)]
    if hi > 179:
        return [(0, hi - 180), (lo, 179)]
    return [(lo, hi)]

def build_line_mask(hsv_roi, h_center, s_center, v_center):
    # Keep S/V floors but allow bright/dim variability
    s_lo = max(S_MIN, int(0.5 * s_center))
    v_lo = max(V_MIN, int(0.5 * v_center))

    ranges = circular_hue_bounds(h_center, H_MARGIN)
    mask = None
    for (h_lo, h_hi) in ranges:
        lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        upper = np.array([h_hi, 255, 255], dtype=np.uint8)
        m = cv2.inRange(hsv_roi, lower, upper)
        mask = m if mask is None else cv2.bitwise_or(mask, m)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask

def find_line_centroid(mask):
    # Return (cx, area, contour) in mask coords, or None if not found
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(best))
    if area < MIN_CONTOUR_AREA:
        return None
    m = cv2.moments(best)
    if m["m00"] <= 1e-6:
        return None
    cx = m["m10"] / m["m00"]
    return (cx, area, best)

def calibrate_line_color(cap):
    """
    User places line in the middle of the camera.
    We sample a center patch in the ROI and take the dominant HSV.
    """
    print("\nCalibration: place the line under the center of the camera view.")
    print("Hold still… capturing samples (about 1.5s).")

    samples_h = []
    samples_s = []
    samples_v = []

    t_end = time.time() + 1.5
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            continue

        roi, _ = roi_crop(frame)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        rh, rw = hsv.shape[:2]

        cx = rw // 2
        cy = rh // 2
        x0 = max(0, cx - CAL_PATCH_W // 2)
        y0 = max(0, cy - CAL_PATCH_H // 2)
        x1 = min(rw, cx + CAL_PATCH_W // 2)
        y1 = min(rh, cy + CAL_PATCH_H // 2)

        patch = hsv[y0:y1, x0:x1]
        # Robust: take median (handles glare/noise)
        med = np.median(patch.reshape(-1, 3), axis=0)
        samples_h.append(med[0])
        samples_s.append(med[1])
        samples_v.append(med[2])

        time.sleep(0.03)

    h = int(np.median(np.array(samples_h)))
    s = int(np.median(np.array(samples_s)))
    v = int(np.median(np.array(samples_v)))

    print(f"Calibrated line HSV center: H={h}, S={s}, V={v}")
    print(f"Using Hue margin ±{H_MARGIN}, S/V floors dynamic (min S={S_MIN}, min V={V_MIN}).")
    return h, s, v

def prompt_start():
    ans = input("\nPress (y) to start line following: ").strip().lower()
    return ans == "y"

# =========================
# Main line-follow loop
# =========================
def run_line_following(cap, pwm, h_center, s_center, v_center):
    print("\nLine following started. Press Ctrl+C to stop.\n")

    # Initialize outputs
    steer_cmd = float(STEERING_CENTER_TICKS)
    throttle_cmd = float(clamp_throttle(FOLLOW_THROTTLE_TICKS))
    pwm.set_pwm_12bit(STEERING_CHANNEL, clamp_steering(steer_cmd))
    pwm.set_pwm_12bit(THROTTLE_CHANNEL, clamp_throttle(throttle_cmd))

    last_seen_line = time.time()
    cx_hist = deque(maxlen=CENTROID_HISTORY)

    next_t = time.perf_counter()

    while True:
        next_t += DT

        ok, frame = cap.read()
        if not ok:
            # camera hiccup: treat as lost line
            time.sleep(0.01)
            continue

        roi, roi_y0 = roi_crop(frame)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = build_line_mask(hsv, h_center, s_center, v_center)
        found = find_line_centroid(mask)

        rh, rw = mask.shape[:2]
        center_x = rw / 2.0

        if found is None:
            # line lost
            if (time.time() - last_seen_line) > LOST_LINE_TIMEOUT_SEC:
                if LOST_LINE_BRAKE:
                    pwm.set_pwm_12bit(THROTTLE_CHANNEL, clamp_throttle(THROTTLE_STOPPED_TICKS))
                pwm.set_pwm_12bit(STEERING_CHANNEL, clamp_steering(STEERING_CENTER_TICKS))
            # keep loop timing
        else:
            cx, area, contour = found
            last_seen_line = time.time()
            cx_hist.append(cx)
            cx_smooth = float(np.mean(cx_hist))

            # error: -1 (line left) .. +1 (line right)
            err = (cx_smooth - center_x) / center_x

            # P control -> steering ticks
            steer_target = STEERING_CENTER_TICKS + (KP_STEER * err)

            # Smooth steering command
            steer_cmd = (1.0 - STEER_SMOOTH_ALPHA) * steer_cmd + STEER_SMOOTH_ALPHA * steer_target

            # Apply outputs
            pwm.set_pwm_12bit(STEERING_CHANNEL, clamp_steering(steer_cmd))
            pwm.set_pwm_12bit(THROTTLE_CHANNEL, clamp_throttle(throttle_cmd))

        # Maintain control rate
        remaining = next_t - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        else:
            next_t = time.perf_counter()


def main():
    pwm = PCA9685Driver(
        address=PCA9685_ADDR,
        busnum=I2C_BUS,
        frequency=PCA9685_FREQ,
        prefer=DRIVER_PREFER,
    )

    cap = None

    def handle_exit(sig=None, frame=None):
        if STOP_ON_EXIT:
            safe_stop(pwm)
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            pwm.close()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Start safe
    safe_stop(pwm)

    cap = open_camera()

    # Calibrate using line in middle of camera
    h, s, v = calibrate_line_color(cap)

    # Ask user to start
    if not prompt_start():
        print("Not starting. Exiting.")
        handle_exit()

    # Go
    run_line_following(cap, pwm, h, s, v)


if __name__ == "__main__":
    main()
