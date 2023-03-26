import serial
import serial.tools.list_ports
import click
from typing import Iterable, Optional


class TriggerDevice:
    def __init__(
        self,
        com: Optional[str] = None,
        baudrate: int = 115200,
        frame_rate: float = 100.0,
        pins: Iterable[int] = [12, 13],
    ) -> None:
        if com is None:
            com = select_serial_port()

        self.com = com
        self.baudrate = baudrate
        self.dev = None
        self.command_params = {"frame_rate": frame_rate, "pins": pins}

    def open(self):
        self.dev = serial.Serial(port=self.com, baudrate=self.baudrate, timeout=0.1)

    def stop(self):
        command_list = (
            [len(self.command_params["pins"])]
            + self.command_params["pins"]
            + [0.] # frame_rate = 0 should led to all pins low
        )
        command_string = ",".join(str(_) for _ in command_list)
        
        # open the device if we haven't yet
        if self.dev is not None:
            self.dev.write(command_string.encode())


    def start(self):
        command_list = (
            [len(self.command_params["pins"])]
            + self.command_params["pins"]
            + [self.command_params["frame_rate"]]
        )
        command_string = ",".join(str(_) for _ in command_list)
        
        # open the device if we haven't yet
        if self.dev is None:
            self.open()

        self.dev.write(command_string.encode())


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
