import serial
import serial.tools.list_ports
import click
import logging
import time
import numpy as np
from typing import Iterable, Optional


class TriggerDevice:
    def __init__(
        self,
        com: Optional[str] = None,
        baudrate: int = 115200,
        # frame_rate: float = 100.0,
        pins: Iterable[int] = [12, 13],
        duration: float = 0,
        alternate_mode: int = 0,
        pulse_widths: Iterable[float] = [.03],
        pulse_width_low: float = .002,
    ) -> None:
        if com is None:
            com = select_serial_port()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.com = com
        self.baudrate = baudrate
        self.dev = None
        # period = 1.0 / frame_rate  # period in seconds
        # max_pulses = (
        #     duration * 60.0
        # ) / period  # number of pulses to meet experiment duration
        self.command_params = {
            # "frame_rate": frame_rate,
            "pins": pins,
            # "max_pulses": max_pulses,
            "alternate_mode": alternate_mode,
            "pulse_widths": [np.round(_pulse_width * 1e6).astype("int") for _pulse_width in pulse_widths],
            "pulse_width_low": np.round(pulse_width_low * 1e6).astype("int")
        }

    def open(self):
        self.dev = serial.Serial(
            port=self.com, baudrate=self.baudrate, timeout=None, write_timeout=2
        )
        self.dev.setDTR(False)
        time.sleep(0.05)
        self.dev.reset_input_buffer()
        self.dev.setDTR(True)
        self.dev.setDTR(False)
        time.sleep(0.05)
        self.dev.reset_input_buffer()
        self.dev.setDTR(True)

        while True:
            if self.dev.in_waiting:
                time.sleep(2)
                self.logger.info(f"Arduino msg: {self.dev.read_all().decode()}")
                break
        # self.logger.info(f"Arduino msg: {self.dev.read_all().decode()}")

    def stop(self):
        command_list = (
            [len(self.command_params["pins"])]
            + self.command_params["pins"]
            + [
                # 0.0, # frame rate
                # 0.0, # max pulses
                -1.0, # alternate mode
                1, # n pulses
                0, # pulse width
                0, # pulse width low
            ]
        )
        command_string = ",".join(str(_) for _ in command_list)
        self.logger.info(f"Sending command string {command_string} to Arduino")
        # open the device if we haven't yet
        if self.dev is not None:
            self.dev.write(command_string.encode())
            while True:
                if self.dev.in_waiting:
                    time.sleep(2)
                    self.logger.info(f"Arduino msg: {self.dev.read_all().decode()}")
                    break

    def start(self):
        command_list = (
            [len(self.command_params["pins"])]
            + self.command_params["pins"]
            + [
                # self.command_params["frame_rate"],
                # self.command_params["max_pulses"],
                self.command_params["alternate_mode"],
                len(self.command_params["pulse_widths"]),
            ]
            + self.command_params["pulse_widths"]
            + [self.command_params["pulse_width_low"]]
        )
        command_string = ",".join(str(_) for _ in command_list)

        # open the device if we haven't yet
        if self.dev is None:
            self.open()

        if self.dev is None:
            raise RuntimeError("No serial device open...")
        self.logger.info(f"Sending command string {command_string} to Arduino")
        self.dev.write(command_string.encode())
        while True:
            if self.dev.in_waiting:
                time.sleep(2)
                self.logger.info(f"Arduino msg: {self.dev.read_all().decode()}")
                break


def select_serial_port():
    all_ports = serial.tools.list_ports.comports()

    if len(all_ports) > 1:
        print("Select serial port to connect to microcontroller:")
        print("-" * 10)
        for idx, port in enumerate(all_ports):
            print("[{}] {}".format(idx, port.device))
        print("-" * 10)

        selection = None
        while selection is None:
            selection = click.prompt("Enter a selection", type=int)
            if selection > len(all_ports) or selection < 0:
                selection = None
    else:
        selection = 0
        print("Using: {}".format(all_ports[0].device))

    return all_ports[selection].device
