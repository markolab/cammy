import serial.tools.list_ports
import click

# TODO:
# 1) simple function/class to initialize arduino and send command string
# 2) if no com port specified, punt to select_serial_port
def select_serial_port():
    all_ports = serial.tools.list_ports.comports()

    print('Select serial port to connect to microcontroller:')
    print('-'*10)
    for idx, port in enumerate(all_ports):
        print('[{}] {}'.format(idx, port.device))
    print('-'*10)

    selection = None
    while selection is None:
        selection = click.prompt('Enter a selection', type=int)
        if selection > len(all_ports) or selection < 0:
            selection = None

    return all_ports[selection].device