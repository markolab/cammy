"""Acknowledged, newline-delimited control of trigger-nucleo firmware."""

import math
import time

import serial
import serial.tools.list_ports


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
    ):
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
            1,
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
        self.config_command = "CONFIG," + ",".join(map(str, params))
        self.com = com
        self.baudrate = baudrate
        self.timeout = timeout
        self.dev = None

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
        except BaseException:
            try:
                self.close()
            except Exception:
                pass  # Preserve the startup error if emergency STOP also fails.
            raise

    def reconfigure(self, candidate):
        """Apply a validated configuration on the existing, stopped controller."""
        self._command(candidate.config_command, "OK CONFIG")
        self.config_command = candidate.config_command

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
