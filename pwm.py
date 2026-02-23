#!/usr/bin/env python3
# pwm.py  (no command-line flags; all config inside the program)

import time
import curses
import signal
import sys

# =========================
# USER CONFIG (edit here)
# =========================
I2C_BUS = 0
PCA9685_ADDR = 0x40
DRIVER_PREFER = "smbus2"   # "smbus2", "legacy", or "auto"

# Channels on PCA9685
THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

# Default frequency (user can change in UI with 'f')
PCA9685_START_FREQ_HZ = 50

# Your calibrated PCA9685 12-bit PWM "ticks" (0..4095)
STEERING_LEFT_PWM    = 280
STEERING_RIGHT_PWM   = 480
THROTTLE_FORWARD_PWM = 385
THROTTLE_STOPPED_PWM = 370
THROTTLE_REVERSE_PWM = 330

# Startup values (ticks)
START_THROTTLE_TICKS = THROTTLE_STOPPED_PWM
START_STEERING_TICKS = (STEERING_LEFT_PWM + STEERING_RIGHT_PWM) // 2

# Safety
STOP_ON_EXIT = True  # force throttle to STOP ticks on quit / Ctrl-C

# Step sizes (ticks)
STEP = 5
BIG_STEP = 25

UI_FPS = 30.0


def ticks_to_us(ticks, freq):
    period_us = 1_000_000.0 / float(freq)
    return int(round((int(ticks) / 4095.0) * period_us))


def ticks_to_duty_pct(ticks):
    return (max(0, min(4095, int(ticks))) / 4095.0) * 100.0


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

        # Init
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

        if prefer in ("smbus2", "auto"):
            try:
                self._drv = PCA9685_SMBus2(busnum=busnum, address=address, frequency=frequency)
                self._mode = "smbus2"
                return
            except Exception as e:
                if prefer == "smbus2":
                    raise
                smbus2_err = e
        else:
            smbus2_err = None

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
        return self._drv.frequency if self._mode == "smbus2" else getattr(self._drv, "frequency", PCA9685_START_FREQ_HZ)

    def close(self):
        if hasattr(self._drv, "close"):
            self._drv.close()

    def set_pwm_freq(self, freq_hz):
        self._drv.set_pwm_freq(freq_hz)

    def set_pwm_12bit(self, channel, value_12bit):
        v = max(0, min(4095, int(value_12bit)))
        if self._mode == "legacy":
            self._drv.set_pwm(channel, 0, v)
        else:
            self._drv.set_pwm_12bit(channel, v)


def prompt_input(stdscr, row, col, prompt):
    stdscr.addstr(row, col, " " * 140)
    stdscr.addstr(row, col, prompt)
    stdscr.refresh()
    curses.echo()
    curses.curs_set(1)
    try:
        s = stdscr.getstr(row, col + len(prompt), 40)
        try:
            return s.decode().strip()
        except Exception:
            return ""
    finally:
        curses.noecho()
        curses.curs_set(0)


def draw_help(stdscr):
    stdscr.addstr(0, 0, "VF2 PWM Tuner (PCA9685)  [12-bit ticks mode: 0..4095]")
    stdscr.addstr(1, 0, "Controls:")
    stdscr.addstr(2, 2, f"Arrow Up/Down   = throttle +/- {STEP} ticks")
    stdscr.addstr(3, 2, f"Arrow Left/Right= steering +/- {STEP} ticks")
    stdscr.addstr(4, 2, f"W/S             = throttle +/- {STEP} ticks (alias)")
    stdscr.addstr(5, 2, f"Shift+W/Shift+S = throttle +/- {BIG_STEP} ticks")
    stdscr.addstr(6, 2, "Space           = throttle -> STOP preset")
    stdscr.addstr(7, 2, "c               = steering -> CENTER preset")
    stdscr.addstr(8, 2, "i               = set ticks (0..4095) for selected channel")
    stdscr.addstr(9, 2, "f               = change PCA9685 frequency (Hz)")
    stdscr.addstr(10,2, "TAB             = switch selected channel (Throttle/Steering)")
    stdscr.addstr(11,2, "Quick set presets:")
    stdscr.addstr(12,4, "1/2/3 = throttle REV/STOP/FWD")
    stdscr.addstr(13,4, "4/5/6 = steering LEFT/CENTER/RIGHT")
    stdscr.addstr(14,2, "q               = quit")
    stdscr.addstr(16,0, "Status:")
    stdscr.refresh()


def run(stdscr):
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    stdscr.nodelay(True)
    curses.curs_set(0)

    pwm = PCA9685Driver(
        address=PCA9685_ADDR,
        busnum=I2C_BUS,
        frequency=PCA9685_START_FREQ_HZ,
        prefer=DRIVER_PREFER,
    )

    values = {
        "throttle": int(START_THROTTLE_TICKS),
        "steering": int(START_STEERING_TICKS),
    }
    channels = {"throttle": THROTTLE_CHANNEL, "steering": STEERING_CHANNEL}

    items = ["throttle", "steering"]
    sel_idx = 0

    def clamp12(v):
        return max(0, min(4095, int(v)))

    def apply(key):
        values[key] = clamp12(values[key])
        pwm.set_pwm_12bit(channels[key], values[key])

    def redraw():
        freq = float(pwm.frequency)
        period_us = 1_000_000.0 / freq

        thr = values["throttle"]
        ste = values["steering"]

        thr_us = ticks_to_us(thr, freq)
        ste_us = ticks_to_us(ste, freq)
        thr_dc = ticks_to_duty_pct(thr)
        ste_dc = ticks_to_duty_pct(ste)

        stdscr.addstr(
            17, 0,
            f"Selected: {items[sel_idx].upper():9s}   PCA9685: addr=0x{PCA9685_ADDR:02X} bus=/dev/i2c-{I2C_BUS}   "
            f"Freq: {freq:7.1f} Hz (period ~ {period_us:7.1f} us)".ljust(140)
        )
        stdscr.addstr(
            19, 0,
            f"Throttle: ch{channels['throttle']}  ticks={thr:4d}  us~{thr_us:4d}  duty={thr_dc:6.2f}%".ljust(140)
        )
        stdscr.addstr(
            20, 0,
            f"Steering: ch{channels['steering']}  ticks={ste:4d}  us~{ste_us:4d}  duty={ste_dc:6.2f}%".ljust(140)
        )

        stdscr.addstr(
            22, 0,
            ("Presets (ticks): "
             f"thr REV/STOP/FWD={THROTTLE_REVERSE_PWM}/{THROTTLE_STOPPED_PWM}/{THROTTLE_FORWARD_PWM}   |   "
             f"ste L/C/R={STEERING_LEFT_PWM}/{(STEERING_LEFT_PWM + STEERING_RIGHT_PWM)//2}/{STEERING_RIGHT_PWM}"
            ).ljust(140)
        )

        stdscr.addstr(
            23, 0,
            ("Direct set: press 'i' to enter ticks for selected channel. "
             "Freq: press 'f' to change Hz.").ljust(140)
        )

        stdscr.refresh()

    def safe_exit():
        if STOP_ON_EXIT:
            try:
                values["throttle"] = clamp12(THROTTLE_STOPPED_PWM)
                pwm.set_pwm_12bit(channels["throttle"], values["throttle"])
            except Exception:
                pass
        try:
            pwm.close()
        except Exception:
            pass

    def sigint_handler(signum, frame):
        safe_exit()
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    # Apply startup outputs
    apply("throttle")
    apply("steering")

    draw_help(stdscr)
    last_ui = 0.0

    try:
        while True:
            ch = stdscr.getch()
            if ch != -1:
                if ch in (ord('q'), ord('Q')):
                    return

                elif ch in (curses.KEY_BTAB, 9):
                    sel_idx = (sel_idx + 1) % len(items)

                # step changes (ticks)
                elif ch in (curses.KEY_UP, ord('w')):
                    values["throttle"] = clamp12(values["throttle"] + STEP)
                    apply("throttle")
                elif ch in (curses.KEY_DOWN, ord('s')):
                    values["throttle"] = clamp12(values["throttle"] - STEP)
                    apply("throttle")
                elif ch == ord('W'):
                    values["throttle"] = clamp12(values["throttle"] + BIG_STEP)
                    apply("throttle")
                elif ch == ord('S'):
                    values["throttle"] = clamp12(values["throttle"] - BIG_STEP)
                    apply("throttle")

                elif ch == curses.KEY_RIGHT:
                    values["steering"] = clamp12(values["steering"] + STEP)
                    apply("steering")
                elif ch == curses.KEY_LEFT:
                    values["steering"] = clamp12(values["steering"] - STEP)
                    apply("steering")

                # direct set ticks
                elif ch in (ord('i'), ord('I')):
                    key = items[sel_idx]
                    s = prompt_input(stdscr, 25, 0, f"Enter 12-bit ticks (0..4095) for {key}: ")
                    try:
                        v = int(float(s))
                        values[key] = clamp12(v)
                        apply(key)
                    except Exception:
                        pass

                # frequency change
                elif ch in (ord('f'), ord('F')):
                    s = prompt_input(stdscr, 25, 0, f"Enter PCA9685 frequency in Hz (current {pwm.frequency:.1f}): ")
                    try:
                        hz = float(s)
                        hz = max(24.0, min(1526.0, hz))
                        pwm.set_pwm_freq(hz)
                        # re-apply current ticks
                        apply("throttle")
                        apply("steering")
                    except Exception:
                        pass

                # quick presets
                elif ch == ord('1'):
                    values["throttle"] = clamp12(THROTTLE_REVERSE_PWM)
                    apply("throttle")
                elif ch == ord('2'):
                    values["throttle"] = clamp12(THROTTLE_STOPPED_PWM)
                    apply("throttle")
                elif ch == ord('3'):
                    values["throttle"] = clamp12(THROTTLE_FORWARD_PWM)
                    apply("throttle")

                elif ch == ord('4'):
                    values["steering"] = clamp12(STEERING_LEFT_PWM)
                    apply("steering")
                elif ch == ord('5'):
                    values["steering"] = clamp12((STEERING_LEFT_PWM + STEERING_RIGHT_PWM) // 2)
                    apply("steering")
                elif ch == ord('6'):
                    values["steering"] = clamp12(STEERING_RIGHT_PWM)
                    apply("steering")

                # convenience keys
                elif ch == ord(' '):
                    values["throttle"] = clamp12(THROTTLE_STOPPED_PWM)
                    apply("throttle")
                elif ch in (ord('c'), ord('C')):
                    values["steering"] = clamp12((STEERING_LEFT_PWM + STEERING_RIGHT_PWM) // 2)
                    apply("steering")

            t = time.time()
            if t - last_ui >= (1.0 / UI_FPS):
                last_ui = t
                redraw()

            time.sleep(1.0 / UI_FPS)

    finally:
        safe_exit()
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()


def main():
    curses.wrapper(run)


if __name__ == "__main__":
    main()
