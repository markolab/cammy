from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, call, patch

from click import ClickException
from click.testing import CliRunner
import numpy as np

from cammy.cli import cli
from cammy.tune import TuneSession, read_settings, save_settings


class TuneTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.settings = root / "cammy-settings.sh"
        self.settings.write_text(f'DATA_DIR="{root}"\nCAMERA_OPTIONS="camera rig.toml"\n'
                                 'GAIN_DB=""  # Use the camera baseline\nFPS=100\n'
                                 'SURFACE_EXPOSURE_US=1000\nFLUORESCENCE_EXPOSURE_US=9000\n'
                                 'SURFACE_VOLTAGE=3\nFLUORESCENCE_VOLTAGE=2\n'
                                 'SURFACE_ILLUMINATION_PERCENT=100\nFLUORESCENCE_ILLUMINATION_PERCENT=50\n'
                                 'LIGHT_SETTLE_US=100\n')
        (root / "camera rig.toml").touch()

    def test_invalid_edits_leave_live_preview_alone(self):
        session = TuneSession(self.settings, "surface", "/dev/test")
        camera = Mock()
        camera.camera.get_gain_bounds.return_value = (0, 12)
        with patch.object(session.trigger, "stop") as stop:
            for gain, width, volts in [("20", 9000, 2), ("nan", 9000, 2), ("", 10000, 2),
                                       ("", 9000, 11), ("", 0, 2)]:
                with self.subTest(gain=gain, width=width, volts=volts), self.assertRaises(ValueError):
                    session.apply({"cam": camera}, "fluorescence", gain, width, volts)
            for fps in [0, -1, float("nan"), float("inf"), 1e-320, 120]:
                with self.subTest(fps=fps), self.assertRaises(ValueError):
                    session.apply({"cam": camera}, "fluorescence", "", 9000, 2, fps=fps)
            for underfill in [-1, 101, float("nan")]:
                with self.subTest(underfill=underfill), self.assertRaisesRegex(ValueError, "Underfill"):
                    session.apply({"cam": camera}, "surface", "", 1000, 3, underfill=underfill)
            stop.assert_not_called()
        self.assertEqual(session.mode, "surface")
        self.assertEqual(session.saved, session.values)

    def test_gain_hint_preserves_blank_override_and_handles_camera_readback(self):
        session = TuneSession(self.settings, "surface", "/dev/test")
        dpg = MagicMock()
        camera = Mock()
        cameras = {"A": camera, "B": camera}
        for readings, override, hint in [
            ([0, 0], "", "0 dB"), ([2.5, 2.5], "", "2.5 dB"),
            ([1, 3], "", "Mixed"), ([2, 2], "4", "2 dB"),
            ([None, RuntimeError("no Gain feature")], "", "Unavailable"),
        ]:
            with self.subTest(readings=readings, override=override):
                session.values["GAIN_DB"] = override
                camera.get_feature.side_effect = readings
                session.build_panel(dpg, cameras)
                field = dpg.add_input_text.call_args.kwargs
                self.assertEqual((field["default_value"], field["hint"]), (override, hint))
                camera.set_feature.assert_not_called()

    def test_underfill_endpoints_only_change_the_selected_bank(self):
        session = TuneSession(self.settings, "surface", "/dev/test")
        camera = Mock()
        port = session.trigger.dev = Mock()
        for bank, dark, exposure, expected_fill in [
            ("surface", 100, 1000, ["0", "5000"]),
            ("fluorescence", 0, 9000, ["0", "10000"]),
        ]:
            with self.subTest(bank=bank):
                port.readline.side_effect = [b"OK STOP\n", b"OK CONFIG\n", b"OK START\n"]
                session.apply({"cam": camera}, bank, "", exposure, 3, underfill=dark)
                command = session.trigger.config_command.split(",")
                self.assertEqual(command[6:8], expected_fill)
                self.assertEqual(int(command[9]), exposure)
        camera.stop_acquisition.assert_not_called()
        camera.set_feature.assert_not_called()
        self.assertEqual(session.values["GAIN_DB"], "")

    def test_save_preserves_shell_expressions_and_detects_external_edits(self):
        raw = b'# header\nGAIN_DB="1" # gain\nGAIN_DB=2; KEEP=yes\nexport SURFACE_VOLTAGE=4  # V\n'
        self.settings.write_bytes(raw)
        new = save_settings(self.settings, raw, {"GAIN_DB": "", "SURFACE_VOLTAGE": "3.5"})
        self.assertIn(b'GAIN_DB=2; KEEP=yes', new)
        self.assertIn(b'export SURFACE_VOLTAGE=3.5  # V', new)
        self.assertEqual(read_settings(self.settings)["GAIN_DB"], "")
        with self.assertRaisesRegex(ValueError, "changed on disk"):
            save_settings(self.settings, raw, {"GAIN_DB": "5"})
        self.assertEqual(self.settings.read_bytes(), new)

    def test_rejected_configuration_does_not_restart_or_commit_the_edit(self):
        session = TuneSession(self.settings, "surface", "/dev/test")
        camera = Mock()
        port = session.trigger.dev = Mock()
        port.readline.side_effect = [b"OK STOP\n", b"ERR CONFIG\n"]
        with self.assertRaisesRegex(ClickException, "ERR CONFIG"):
            session.apply({"A": camera}, "fluorescence", "", 8000, 3)
        commands = [c.args[0].split(b",", 1)[0].strip() for c in port.write.call_args_list]
        self.assertEqual(commands, [b"STOP", b"CONFIG"])
        camera.stop_acquisition.assert_not_called()
        self.assertEqual(session.mode, "surface")
        self.assertEqual(session.values, session.saved)

    def test_tune_preview_switch_apply_and_save(self):
        # Run the real Click command, preview loop and bulk camera/serial control, without a rig.
        dpg = MagicMock()
        buttons = {}
        def add_button(label, callback, **attrs):
            buttons[label] = (callback, attrs.get("user_data"))
        dpg.add_button.side_effect = add_button
        original = self.settings.read_bytes()
        def interact(*_):
            apply, bank = buttons["Apply fluorescence"]
            apply(None, None, bank)
            self.assertEqual(self.settings.read_bytes(), original)
            buttons["Save applied settings"][0]()
        dpg.run_callbacks.side_effect = interact
        dpg.get_value.side_effect = {"tune_gain": "2.5", "tune_fps": 80.5, "fluorescence_us": 8000,
                                     "fluorescence_volts": 3.75, "fluorescence_underfill": 25.25}.__getitem__
        dpg.is_dearpygui_running.side_effect = [True, False]
        frame = np.zeros((12, 12), dtype=np.uint8)
        frame[:3] = 255
        cameras = {}
        for key, initial_gain in [("A", 1), ("B", 3)]:
            cam = Mock(id=key, _width=12, _height=12, frame_count=1, total_frames=1, missed_frames=0,
                       fps=None, display_frame=(frame, {}))
            cam.get_all_features.return_value = dict(PixelFormat="Mono8", ExposureMode="TriggerWidth",
                TriggerSelector="FrameStart", TriggerMode="On", TriggerActivation="RisingEdge", TriggerSource="Line1")
            cam.camera.get_gain_bounds.return_value = (0, 24)
            cam.get_feature.side_effect = {"GainAuto": "Off", "Gain": initial_gain}.__getitem__
            cameras[key] = cam
        serial = Mock()
        port = serial.return_value
        port.readline.side_effect = [b"CAMMY_NUCLEO,1\n", b"OK CONFIG\n", b"OK START\n", b"OK STOP\n",
                                     b"OK CONFIG\n", b"OK START\n", b"OK STOP\n"]
        with ExitStack() as stack:
            stack.enter_context(patch.dict("sys.modules", {"dearpygui": Mock(dearpygui=dpg), "dearpygui.dearpygui": dpg}))
            for name, value in [("cammy.cli.get_all_camera_ids", {k: "aravis" for k in cameras}),
                                ("cammy.cli.initialize_cameras", cameras),
                                ("cammy.cli.get_cpu_affinity", (None, [None], [])),
                                ("cammy.cli.time.sleep", None), ("cammy.cli.mpl_to_cv2_colormap", 2),
                                ("cammy.camera.aravis.acquisition_loop", None)]:
                stack.enter_context(patch(name, return_value=value))
            stack.enter_context(patch("cammy.trigger.nucleo.serial.Serial", serial))
            result = CliRunner().invoke(cli, ["tune", "--settings", str(self.settings), "--trigger-port", "/dev/test"],
                                        catch_exceptions=False)
        self.assertEqual(result.exit_code, 0, result.output)
        serial.assert_called_once()
        commands = [c.args[0].decode().strip() for c in port.write.call_args_list]
        self.assertEqual([c.split(",")[0] for c in commands], ["HELLO", "CONFIG", "START", "STOP", "CONFIG", "START", "STOP"])
        self.assertEqual(commands[4], "CONFIG,1,1,100,3000,3750,10000,7475,1,8000,4422")
        port.close.assert_called_once()
        for key, gain in [("A", 1), ("B", 3)]:
            self.assertEqual(cameras[key].set_feature.call_args_list,
                             [call("GainAuto", "Off"), call("Gain", "2.5"), call("GainAuto", "Off"), call("Gain", str(gain))])
            dpg.set_value.assert_any_call(f"saturation_{key}", "Saturated pixels: 25.000%")
        self.assertIn('GAIN_DB=2.5  # Use the camera baseline', self.settings.read_text())
        saved = read_settings(self.settings)
        self.assertEqual(saved["FLUORESCENCE_EXPOSURE_US"], "8000")
        self.assertEqual(float(saved["FLUORESCENCE_ILLUMINATION_PERCENT"]), 74.75)
        self.assertEqual(saved["FPS"], "80.5")
        dpg.destroy_context.assert_called_once()
