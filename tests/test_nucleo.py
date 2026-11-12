import unittest
from unittest.mock import Mock, patch

from cammy.trigger.nucleo import NucleoTriggerDevice


class NucleoTests(unittest.TestCase):
    def test_configuration_units_and_broadcast_low(self):
        dev = NucleoTriggerDevice(
            mode="alternate", pulse_widths=(.01, .02), pulse_widths_low=(.002,),
            surface_illumination_percent=80, fluorescence_illumination_percent=50,
            surface_voltage=.7, fluorescence_voltage=10,
        )
        self.assertEqual(dev.config_command,
                         "CONFIG,1,2,200,700,10000,8000,5000,2,10000,2000,20000,2000")

    def test_invalid_configuration_rejected_before_opening_serial(self):
        for kwargs in (
            {"mode": "both"}, {"pulse_widths": ()}, {"pulse_widths": (0,)},
            {"pulse_widths": (float("nan"),)}, {"pulse_widths": (3000,)},
            {"pulse_widths": (.001,) * 21}, {"pulse_widths_low": ()},
            {"pulse_widths_low": (.001, .002)}, {"pulse_widths_low": (.0001,)},
            {"surface_illumination_percent": 101}, {"fluorescence_illumination_percent": -1},
            {"surface_voltage": .69}, {"fluorescence_voltage": 10.01},
            {"light_settle_us": -1}, {"light_settle_us": 1.5}, {"timeout": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NucleoTriggerDevice(**kwargs)

    @patch("cammy.trigger.nucleo.serial.Serial")
    def test_configure_before_start_and_stop_before_close(self, serial_class):
        port = serial_class.return_value
        port.readline.side_effect = [b"CAMMY_NUCLEO,1\n", b"OK CONFIG\n", b"OK START\n", b"OK STOP\n"]
        device = NucleoTriggerDevice(com="test")
        device.open()
        self.assertEqual(port.write.call_count, 2)  # open() never starts timing.
        device.start()
        device.close()
        self.assertEqual([call.args[0] for call in port.write.call_args_list],
                         [b"HELLO\n", (device.config_command + "\n").encode(), b"START\n", b"STOP\n"])
        port.close.assert_called_once()
        self.assertIsNone(device.dev)

    @patch("cammy.trigger.nucleo.serial.Serial")
    def test_configuration_rejection_closes_port(self, serial_class):
        port = serial_class.return_value
        port.readline.side_effect = [b"CAMMY_NUCLEO,1\n", b"ERR CONFIG\n"]
        with self.assertRaisesRegex(RuntimeError, "rejected CONFIG"):
            NucleoTriggerDevice(com="test").open()
        port.close.assert_called_once()

    @patch("cammy.trigger.nucleo.serial.Serial")
    def test_startup_instructions_do_not_interfere_with_handoff(self, serial_class):
        port = serial_class.return_value
        port.readline.side_effect = [
            b"# Cammy NUCLEO-G474RE light/camera controller\n",
            b"# Serial monitor: 115200 baud, Newline or Both NL & CR\n",
            b"# CONFIG,1,2,200,700,700,8000,5000,1,10000,2000\n",
            b"CAMMY_NUCLEO,1\n", b"OK CONFIG\n", b"OK START\n", b"OK STOP\n",
        ]
        device = NucleoTriggerDevice(com="test")
        device.start()
        device.close()
        port.close.assert_called_once()

    def test_missing_acknowledgement_times_out(self):
        device = NucleoTriggerDevice(com="test", timeout=.001)
        device.dev = Mock()
        device.dev.readline.return_value = b""
        with self.assertRaises(TimeoutError):
            device.stop()

    def test_fragmented_acknowledgement(self):
        device = NucleoTriggerDevice(com="test")
        device.dev = Mock()
        device.dev.readline.side_effect = [b"OK ", b"STOP\n"]
        device.stop()

    @patch("cammy.trigger.nucleo.serial.Serial")
    def test_unrecognized_firmware_gets_no_config_or_stop(self, serial_class):
        port = serial_class.return_value
        port.readline.return_value = b"Legacy firmware\n"
        with self.assertRaises(TimeoutError):
            NucleoTriggerDevice(com="test", timeout=.001).open()
        port.write.assert_called_once_with(b"HELLO\n")
        port.close.assert_called_once()

    @patch("cammy.trigger.nucleo.serial.Serial")
    def test_start_failure_sends_stop_and_closes_port(self, serial_class):
        port = serial_class.return_value
        port.readline.side_effect = [b"CAMMY_NUCLEO,1\n", b"OK CONFIG\n", b"ERR COMMAND\n", b"OK STOP\n"]
        with self.assertRaisesRegex(RuntimeError, "rejected START"):
            NucleoTriggerDevice(com="test").start()
        self.assertEqual(port.write.call_args.args[0], b"STOP\n")
        port.close.assert_called_once()

    def test_stop_failure_still_closes_port(self):
        device = NucleoTriggerDevice(com="test")
        port = device.dev = Mock()
        port.readline.return_value = b"ERR COMMAND\n"
        with self.assertRaises(RuntimeError):
            device.close()
        port.close.assert_called_once()
        self.assertIsNone(device.dev)

    @patch("cammy.trigger.nucleo.serial.tools.list_ports.comports", return_value=[])
    def test_no_serial_ports(self, _ports):
        with self.assertRaisesRegex(RuntimeError, "--trigger-port"):
            NucleoTriggerDevice().open()


if __name__ == "__main__":
    unittest.main()
