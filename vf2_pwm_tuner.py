#!/usr/bin/env python3
# vf2_pwm_tuner.py

import time
import curses
import signal
import sys
import argparse

PCA9685_DEFAULT_ADDR = 0x40
PCA9685_DEFAULT_FREQ = 60  # Hz
PCA9685_DEFAULT_BUS  = 1

THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

THROTTLE_STOPPED_US = 1500
STEERING_CENTER_US  = 1500  # set to 1600 if needed

STEP = 5
BIG_STEP = 25
UI_FPS = 30.0


def us_to_12bit(us, freq):
    period_us = 1_000_000.0 / float(freq)
    ticks = round((us / period_us) * 4095.0)
    return max(0, min(4095, int(ticks)))


def ticks_to_us(ticks, freq):
    period_us = 1_000_000.0 / float(freq)
    return int(round((ticks / 4095.0) * period_us))


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
        return self._drv.frequency if self._mode == "smbus2" else getattr(self._drv, "frequency", PCA9685_DEFAULT_FREQ)

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


def prompt_input(stdscr, row, col, prompt):
    stdscr.addstr(row, col, " " * 120)
    stdscr.addstr(row, col, prompt)
    stdscr.refresh()
    curses.echo()
    curses.curs_set(1)
    try:
        s = stdscr.getstr(row, col + len(prompt), 30)
        try:
            return s.decode().strip()
        except Exception:
            return ""
    finally:
        curses.noecho()
        curses.curs_set(0)


def draw_help(stdscr):
    stdscr.addstr(0, 0, "VF2 PWM Tuner (PCA9685)")
    stdscr.addstr(1, 0, "Controls:")
    stdscr.addstr(2, 2, "Arrow Up/Down   = throttle +/- step")
    stdscr.addstr(3, 2, "Arrow Left/Right= steering +/- step")
    stdscr.addstr(4, 2, "W/S             = throttle +/- step (alias)")
    stdscr.addstr(5, 2, "Shift+W/Shift+S = throttle +/- BIG step")
    stdscr.addstr(6, 2, "Space           = throttle -> stop (default 1500us)")
    stdscr.addstr(7, 2, "c               = steering -> center (default 1500us)")
    stdscr.addstr(8, 2, "i               = set ticks (0..4095) for selected channel")
    stdscr.addstr(9, 2, "u               = set microseconds for selected channel")
    stdscr.addstr(10,2, "f               = change PCA9685 frequency (Hz)")
    stdscr.addstr(11,2, "TAB             = switch selected channel (Throttle/Steering)")
    stdscr.addstr(12,2, "q               = quit")
    stdscr.addstr(14,0, "Status:")
    stdscr.refresh()


def run(stdscr, args):
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    stdscr.nodelay(True)
    curses.curs_set(0)

    pwm = PCA9685Driver(
        address=args.addr,
        busnum=args.bus,
        frequency=args.freq,
        prefer=args.prefer,
    )

    values = {
        "throttle": us_to_12bit(args.throttle_us, pwm.frequency),
        "steering": us_to_12bit(args.steering_us, pwm.frequency),
    }
    channels = {"throttle": args.throttle_ch, "steering": args.steering_ch}

    items = ["throttle", "steering"]
    sel_idx = 0

    def clamp12(v):
        return max(0, min(4095, int(v)))

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
            15, 0,
            f"Selected: {items[sel_idx].upper():9s}   PCA9685: addr=0x{args.addr:02X} bus=/dev/i2c-{args.bus}   "
            f"Freq: {int(freq):4d} Hz (period ~ {period_us:7.1f} us)          "
        )
        stdscr.addstr(17, 0, f"Throttle: ch{channels['throttle']}  ticks={thr:4d}  us~{thr_us:4d}  duty={thr_dc:6.2f}%                              ")
        stdscr.addstr(18, 0, f"Steering: ch{channels['steering']}  ticks={ste:4d}  us~{ste_us:4d}  duty={ste_dc:6.2f}%                              ")
        stdscr.addstr(20, 0, f"Targets: stop={args.throttle_stop_us}us (ticks {us_to_12bit(args.throttle_stop_us, freq)}), center={args.steering_center_us}us (ticks {us_to_12bit(args.steering_center_us, freq)})          ")
        stdscr.refresh()

    def safe_exit():
        if args.stop_on_exit:
            try:
                values["throttle"] = us_to_12bit(args.throttle_stop_us, pwm.frequency)
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
    pwm.set_pwm_12bit(channels["throttle"], values["throttle"])
    pwm.set_pwm_12bit(channels["steering"], values["steering"])

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

                elif ch in (curses.KEY_UP, ord('w')):
                    values["throttle"] = clamp12(values["throttle"] + STEP)
                    pwm.set_pwm_12bit(channels["throttle"], values["throttle"])
                elif ch in (curses.KEY_DOWN, ord('s')):
                    values["throttle"] = clamp12(values["throttle"] - STEP)
                    pwm.set_pwm_12bit(channels["throttle"], values["throttle"])
                elif ch == ord('W'):
                    values["throttle"] = clamp12(values["throttle"] + BIG_STEP)
                    pwm.set_pwm_12bit(channels["throttle"], values["throttle"])
                elif ch == ord('S'):
                    values["throttle"] = clamp12(values["throttle"] - BIG_STEP)
                    pwm.set_pwm_12bit(channels["throttle"], values["throttle"])

                elif ch == curses.KEY_RIGHT:
                    values["steering"] = clamp12(values["steering"] + STEP)
                    pwm.set_pwm_12bit(channels["steering"], values["steering"])
                elif ch == curses.KEY_LEFT:
                    values["steering"] = clamp12(values["steering"] - STEP)
                    pwm.set_pwm_12bit(channels["steering"], values["steering"])

                elif ch in (ord('i'), ord('I')):
                    key = items[sel_idx]
                    s = prompt_input(stdscr, 22, 0, f"Enter 12-bit ticks (0..4095) for {key}: ")
                    try:
                        v = int(s)
                        values[key] = clamp12(v)
                        pwm.set_pwm_12bit(channels[key], values[key])
                    except Exception:
                        pass

                elif ch in (ord('u'), ord('U')):
                    key = items[sel_idx]
                    s = prompt_input(stdscr, 22, 0, f"Enter microseconds for {key} (e.g., 1500): ")
                    try:
                        us = float(s)
                        values[key] = clamp12(us_to_12bit(us, pwm.frequency))
                        pwm.set_pwm_12bit(channels[key], values[key])
                    except Exception:
                        pass

                elif ch in (ord('f'), ord('F')):
                    s = prompt_input(stdscr, 22, 0, f"Enter PCA9685 frequency in Hz (current {int(pwm.frequency)}): ")
                    try:
                        hz = int(float(s))
                        hz = max(24, min(1526, hz))
                        pwm.set_pwm_freq(hz)
                        pwm.set_pwm_12bit(channels["throttle"], values["throttle"])
                        pwm.set_pwm_12bit(channels["steering"], values["steering"])
                    except Exception:
                        pass

                elif ch == ord(' '):
                    values["throttle"] = us_to_12bit(args.throttle_stop_us, pwm.frequency)
                    pwm.set_pwm_12bit(channels["throttle"], values["throttle"])
                elif ch in (ord('c'), ord('C')):
                    values["steering"] = us_to_12bit(args.steering_center_us, pwm.frequency)
                    pwm.set_pwm_12bit(channels["steering"], values["steering"])

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
    p = argparse.ArgumentParser(description="PCA9685 PWM tuner (VisionFive-friendly)")

    # Built-in defaults: --bus 1 --addr 0x40 --freq 60
    p.add_argument("--bus", type=int, default=PCA9685_DEFAULT_BUS, help="I2C bus number (uses /dev/i2c-<bus>)")
    p.add_argument("--addr", type=lambda x: int(x, 0), default=PCA9685_DEFAULT_ADDR, help="PCA9685 I2C address (e.g. 0x40)")
    p.add_argument("--freq", type=int, default=PCA9685_DEFAULT_FREQ, help="PCA9685 PWM frequency in Hz")

    p.add_argument("--prefer", choices=["auto", "smbus2", "legacy"], default="smbus2",
                   help="Driver backend preference")

    p.add_argument("--throttle-ch", type=int, default=THROTTLE_CHANNEL)
    p.add_argument("--steering-ch", type=int, default=STEERING_CHANNEL)

    p.add_argument("--throttle-us", type=int, default=THROTTLE_STOPPED_US, help="Startup throttle pulse width (us)")
    p.add_argument("--steering-us", type=int, default=STEERING_CENTER_US, help="Startup steering pulse width (us)")
    p.add_argument("--throttle-stop-us", type=int, default=THROTTLE_STOPPED_US, help="Stop pulse width (us)")
    p.add_argument("--steering-center-us", type=int, default=STEERING_CENTER_US, help="Center pulse width (us)")

    # Built-in default: --stop-on-exit enabled
    p.add_argument("--stop-on-exit", action="store_true", default=True,
                   help="Force throttle to stop on exit (default: enabled)")
    p.add_argument("--no-stop-on-exit", dest="stop_on_exit", action="store_false",
                   help="Disable forcing throttle to stop on exit")

    args = p.parse_args()
    curses.wrapper(lambda stdscr: run(stdscr, args))


if __name__ == "__main__":
    main()
