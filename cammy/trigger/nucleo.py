"""Acknowledged, newline-delimited control of trigger-nucleo firmware."""

import logging
import math
import time

import serial
import serial.tools.list_ports

logger = logging.getLogger(__name__)


class NucleoTriggerDevice:
    MODES = {"surface": 0, "fluorescence": 1, "alternate": 2}
    MAX_STEPS = 20
    MAX_INTERVAL_US = 2**31 - 1

    def __init__(
        self,
        com=None,
        baudrate=115200,
        mode="surface",
        pulse_widths=(0.03,),
        pulse_widths_low=(0.002,),
        surface_illumination_percent=100,
        fluorescence_illumination_percent=100,
        surface_voltage=0.7,
        fluorescence_voltage=0.7,
        light_settle_us=200,
        timeout=3.0,
        surface_polarity="low",
    ):
        if surface_polarity not in ("low", "high"):
            raise ValueError("Surface polarity must be low or high")
        if mode not in self.MODES:
            raise ValueError(f"Unknown light mode: {mode}")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Serial timeout must be finite and positive")
        if (
            not 0 <= light_settle_us <= self.MAX_INTERVAL_US
            or int(light_settle_us) != light_settle_us
        ):
            raise ValueError("Light settling time must be a nonnegative integer in microseconds")

        widths = self._microseconds(pulse_widths)
        lows = self._microseconds(pulse_widths_low)
        if not 1 <= len(widths) <= self.MAX_STEPS:
            raise ValueError("Specify between 1 and 20 exposure widths")
        if len(lows) == 1:
            lows *= len(widths)
        if len(lows) != len(widths):
            raise ValueError("Supply one low interval or one per exposure width")
        if any(width < 1 for width in widths):
            raise ValueError("Exposure widths must be at least one microsecond")
        if any(low < max(1, light_settle_us) for low in lows):
            raise ValueError("Low intervals must be positive and include the light settling time")
        if any(width + low > self.MAX_INTERVAL_US for width, low in zip(widths, lows)):
            raise ValueError("Exposure plus low interval exceeds the firmware timing range")

        params = [
            2 if surface_polarity == "high" else 1,
            self.MODES[mode],
            int(light_settle_us),
            self._scaled(surface_voltage, 0.7, 10, 1000, "Surface voltage"),
            self._scaled(fluorescence_voltage, 0.7, 10, 1000, "Fluorescence voltage"),
            self._scaled(
                surface_illumination_percent, 0, 100, 100, "Surface illumination percent"
            ),
            self._scaled(
                fluorescence_illumination_percent, 0, 100, 100,
                "Fluorescence illumination percent",
            ),
            len(widths),
        ]
        for width, low in zip(widths, lows):
            params.extend((width, low))
        if surface_polarity == "high":
            params.append(1)  # CONFIG,2 adds the polarity after the timing pairs.
        self._config = tuple(params)
        self.com = com
        self.baudrate = baudrate
        self.timeout = timeout
        self.dev = None

    @property
    def config_command(self):
        return "CONFIG," + ",".join(map(str, self._config))

    @property
    def mode(self):
        return ("surface", "fluorescence", "alternate")[self._config[1]]

    @property
    def steps(self):
        timings = self._config[8:8 + 2 * self._config[7]]
        return list(zip(timings[::2], timings[1::2]))

    def describe_sequence(self):
        _, _, settle, surface_mv, fluo_mv, surface_fill, fluo_fill, _ = self._config[:8]
        steps = self.steps
        if self.mode == "alternate" and len(steps) % 2:
            steps *= 2  # The banks toggle independently of the firmware's step index.
        period = sum(width + low for width, low in steps)
        fps = 1e6 * len(steps) / period
        rate = f"{fps:.3f} FPS total"
        if self.mode == "alternate":
            rate += f", {fps / 2:.3f} pairs/s"
        lines = [f"Nucleo {self.mode}, programmed cycle {period} µs, {rate}",
                 f"D2 surface active {'high' if self._config[0] == 2 else 'low'}, D3 fluorescence active high. Brightness setpoints "
                 f"{surface_mv / 1000:g} V surface, {fluo_mv / 1000:g} V fluorescence. "
                 f"Settling {settle} µs. Times from first D4 rise."]
        start = 0
        for i, (width, low) in enumerate(steps):
            fluo = self.mode == "fluorescence" or (self.mode == "alternate" and i % 2 == 1)
            bank, fill = ("fluorescence", fluo_fill) if fluo else ("surface", surface_fill)
            lit = (width * fill + 5000) // 10000  # Same integer rounding as prepare_light().
            if not lit:
                light = "off"
            elif self.mode != "alternate" and fill == 10000:
                light = f"continuous from {-settle} µs"
            else:
                light = f"on [{start - settle}, {start + lit}) µs"
            lines.append(f"  {i + 1}. {bank} light {light} ({fill / 100:g}% fill), D4 high [{start}, {start + width}) µs "
                         f"({width} µs exposure), then low {low} µs")
            start += width + low
        return "\n".join([*lines, self.config_command])

    @staticmethod
    def _scaled(value, minimum, maximum, scale, name):
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")
        return round(value * scale)

    @classmethod
    def _microseconds(cls, values):
        return [
            cls._scaled(value, 0, cls.MAX_INTERVAL_US / 1e6, 1e6, "Timing (seconds)")
            for value in values
        ]

    def _command(self, command, expected):
        self.dev.write((command + "\n").encode("ascii"))
        deadline = time.monotonic() + self.timeout
        pending = b""
        while time.monotonic() < deadline:
            pending += self.dev.readline()
            if not pending.endswith(b"\n"):
                continue
            response = pending.decode("ascii", errors="replace").strip()
            pending = b""
            if response == expected:
                return
            if response.startswith("ERR"):
                raise RuntimeError(f"Nucleo rejected {command.split(',')[0]}: {response}")
        raise TimeoutError(f"No {expected!r} response from Nucleo on {self.com}")

    def open(self):
        """Verify firmware identity and configure it with all outputs stopped."""
        if self.dev is not None:
            return
        if self.com is None:
            ports = [port.device for port in serial.tools.list_ports.comports()]
            if len(ports) != 1:
                raise RuntimeError(f"Specify --trigger-port; detected ports: {ports}")
            self.com = ports[0]
        self.dev = serial.Serial(
            self.com, self.baudrate, timeout=0.1, write_timeout=self.timeout
        )
        try:
            # ST-Link virtual COM does not require the Arduino DTR reset sequence.
            self.dev.reset_input_buffer()
            self._command("HELLO", "CAMMY_NUCLEO,1")
            self._command(self.config_command, "OK CONFIG")
            logger.info("%s", self.describe_sequence())
        except BaseException:
            # HELLO/CONFIG never starts outputs. Do not send our STOP protocol
            # to a device whose firmware identity could not be verified.
            self.dev.close()
            self.dev = None
            raise

    def start(self):
        """Start only after cameras have entered acquisition."""
        try:
            self.open()
            self._command("START", "OK START")
            self.started_at = time.perf_counter()
        except BaseException:
            try:
                self.close()
            except Exception:
                pass  # Preserve the startup error if emergency STOP also fails.
            raise

    def reconfigure(self, candidate):
        """Apply a validated configuration on the existing, stopped controller."""
        self._command(candidate.config_command, "OK CONFIG")
        self._config = candidate._config
        logger.info("%s", self.describe_sequence())

    def stop(self):
        if self.dev is not None:
            self._command("STOP", "OK STOP")

    def close(self):
        if self.dev is not None:
            try:
                self.stop()
            finally:
                self.dev.close()
                self.dev = None
