from contextlib import ExitStack
from pathlib import Path
import tempfile
import time
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

    def test_rejected_crop_restores_every_touched_camera_before_stream_setup(self):
        cameras = {}
        for key in ("A", "B"):
            cam = Mock()
            native = cam.camera
            native.get_region.return_value = (0, 0, 128, 128)
            native.get_width_bounds.return_value = (1, 128)
            native.get_height_bounds.return_value = (1, 128)
            native.is_feature_available.side_effect = lambda name: name == "CenterY"
            native.get_boolean.return_value = True
            native.get_height_increment.return_value = native.get_y_offset_increment.return_value = 1
            def set_roi(*region, n=native, name=key):
                n.get_region.return_value = tuple(old if new == -1 else new for old, new in zip(n.get_region(), region))
                if name == "B" and region[3] == 64:
                    raise RuntimeError("camera rejected ROI")
            native.set_region.side_effect = set_roi
            cameras[key] = cam
        with ExitStack() as stack:
            dpg = Mock()
            stack.enter_context(patch.dict("sys.modules", {"dearpygui": Mock(dearpygui=dpg), "dearpygui.dearpygui": dpg}))
            stack.enter_context(patch("cammy.cli.initialize_cameras", return_value=cameras))
            stack.enter_context(patch("cammy.cli.get_all_camera_ids", return_value={k: "aravis" for k in cameras}))
            stack.enter_context(patch("cammy.cli.time.sleep"))
            stack.enter_context(patch("cammy.cli.mpl_to_cv2_colormap"))
            result = CliRunner().invoke(cli, ["run", "--roi-height", "64"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Could not crop B", result.output)
        for cam in cameras.values():
            self.assertEqual(cam.camera.get_region(), (0, 0, 128, 128))
            self.assertEqual(cam.camera.set_boolean.call_args_list, [call("CenterY", False), call("CenterY", True)])
            cam.initialize_acquisition_stream.assert_not_called()

    def test_tune_preview_switch_crop_suggest_and_save(self):
        # Run the real Click command, preview loop and bulk camera/serial control, without a rig.
        dpg = MagicMock()
        buttons = {}
        def add_button(label, callback, **attrs):
            buttons[label] = (callback, attrs.get("user_data"))
        dpg.add_button.side_effect = add_button
        with self.settings.open("a") as settings:
            settings.write("ROI_WIDTH=24\nSURFACE_TO_FLUORESCENCE_GAP_US=1250\n")
        (self.settings.parent / "camera rig.toml").write_text("[genicam.A.a]\nWidth=12\nOffsetX=6\n")
        original = self.settings.read_bytes()
        fields = {"tune_gain": "2.5", "tune_fps": 80.5, "surface_us": 1000, "surface_underfill": 0,
                  "surface_volts": 3, "fluorescence_us": 8000, "fluorescence_volts": 3.75,
                  "fluorescence_underfill": 25.25, "margin_ms": 1.5, "surface_gap_us": 1250,
                  "roi_width": 12, "roi_height": 0, "roi_horizontal": "right", "roi_vertical": "center"}
        dpg.get_value.side_effect = fields.__getitem__
        dpg.set_value.side_effect = fields.__setitem__
        def add_texture(width, height, canvas, *, tag, **kwargs):
            fields[tag] = canvas
        dpg.add_raw_texture.side_effect = add_texture
        def interact(*_):
            self.assertEqual(self.settings.read_bytes(), original)
            tick = dpg.run_callbacks.call_count
            if tick == 1:
                self.assertIn("Slowest sensor readout 1.50 ms", fields["timing_advice"])
                self.assertIn("gap 1250 µs → fluorescence", fields["timing_advice"])
                gap_widget = next(c.kwargs for c in dpg.add_input_int.call_args_list if c.kwargs["tag"] == "surface_gap_us")
                self.assertEqual(gap_widget["default_value"], 1250)
                fields["surface_gap_us"] = 0
                buttons["Suggest alternating timing"][0]()
                self.assertIn("gap 100 µs → fluorescence", fields["timing_advice"])
                apply, bank = buttons["Apply fluorescence"]
                apply(None, None, bank)
                fields["fluorescence_us"] = 8500  # Draft survives crop and is used only for advice.
            elif tick in (2, 3):
                # Two input ticks after the A/B batch, including one before another image update is due.
                self.assertIn("saturation_A", fields)
                self.assertIn("saturation_B", fields)
                self.assertNotIn("saturation_C", fields)
            elif tick == 4:
                self.assertIn("saturation_C", fields)
                buttons["Apply crop"][0]()
            elif tick == 5:
                image = fields["texture_A"]
                self.assertEqual(image.shape, (6, 6, 4))
                self.assertTrue(np.any(image[:, 4:, :3]))
                self.assertFalse(np.any(image[:, :4, :3]))
                self.assertEqual(fields["saturation_A"], "Saturated pixels: 25.000%")
                self.assertIn("gap 100 µs → fluorescence 8500", fields["timing_advice"])
                self.assertIn("Slowest sensor readout 1.00 ms", fields["timing_advice"])
                fields["roi_width"] = 0
                buttons["Apply crop"][0]()
            elif tick == 7:
                self.assertIn("Slowest sensor readout 2.00 ms", fields["timing_advice"])
                self.assertTrue(np.any(fields["texture_A"][:, :2, :3]))
                fields["roi_width"] = 18
                buttons["Apply crop"][0]()
                self.assertIn("Could not crop B", fields["tune_status"])
                self.assertEqual(cameras["A"].camera.get_region(), (0, 0, 36, 36))
                self.assertEqual(cameras["B"].camera.get_region(), (0, 0, 36, 36))
            elif tick == 8:
                fields["roi_width"] = 12
                buttons["Apply crop"][0]()
                buttons["Save applied settings"][0]()
        dpg.run_callbacks.side_effect = interact
        dpg.is_dearpygui_running.side_effect = [True] * 8 + [False]
        frame = np.zeros((36, 36), dtype=np.uint8)
        frame[:9] = 255
        cameras = {}
        initial_gains = {"A": 1, "B": 3, "C": 2}
        for key, initial_gain in initial_gains.items():
            cam = Mock(id=key, _width=36, _height=36, _spoof_cameras=[], frame_count=1, total_frames=1, missed_frames=0,
                       fps=None, display_frame=(frame, {}))
            cam.get_all_features.return_value = dict(PixelFormat="Mono8", SensorReadoutTime=2000, ExposureMode="TriggerWidth",
                TriggerSelector="FrameStart", TriggerMode="On", TriggerActivation="RisingEdge", TriggerSource="Line1")
            cam.camera.get_region.return_value = (0, 0, 36, 36)
            def set_roi(*region, n=cam.camera):
                n.get_region.return_value = tuple(old if new == -1 else new for old, new in zip(n.get_region(), region))
            cam.camera.set_region.side_effect = set_roi
            cam.camera.get_width_bounds.return_value = (1, 36)
            cam.camera.get_height_bounds.return_value = (1, 36)
            cam.camera.get_width_increment.return_value = cam.camera.get_x_offset_increment.return_value = 1
            def init_stream(c=cam):
                _, _, c._width, c._height = c.camera.get_region()
                c._payload = c._width * c._height
                c.get_all_features.return_value["SensorReadoutTime"] = {36: 2000, 24: 1500, 12: 1000}.get(c._width)
                c.display_frame = (frame[:, :c._width], {})
            cam.initialize_acquisition_stream.side_effect = init_stream
            def change_region(region, c=cam, refresh=init_stream):
                c.camera.set_region(*region)
                if c.id == "B" and region[2] == 18:
                    raise RuntimeError("camera rejected crop")
                refresh()
            cam.change_region.side_effect = change_region
            cam.camera.get_gain_bounds.return_value = (0, 24)
            cam.get_feature.side_effect = lambda name, c=cam, g=initial_gain: {
                "GainAuto": "Off", "Gain": g, "SensorReadoutTime": 2000 if c._width == 36 else 1000}[name]
            cameras[key] = cam
        serial = Mock()
        port = serial.return_value
        port.readline.side_effect = [b"CAMMY_NUCLEO,1\n", b"OK CONFIG\n", b"OK START\n", b"OK STOP\n",
                                     b"OK CONFIG\n", b"OK START\n"] + [b"OK STOP\n", b"OK START\n"] * 4 + [b"OK STOP\n"]
        with ExitStack() as stack:
            stack.enter_context(patch.dict("sys.modules", {"dearpygui": Mock(dearpygui=dpg), "dearpygui.dearpygui": dpg}))
            clock = stack.enter_context(patch("cammy.cli.time", wraps=time))
            clock.monotonic.side_effect = [0, .01, .05, .1, .15, .2, .25, .3]
            clock.sleep.return_value = None
            for name, value in [("cammy.cli.get_all_camera_ids", {k: "aravis" for k in cameras}),
                                ("cammy.cli.initialize_cameras", cameras),
                                ("cammy.cli.get_cpu_affinity", (None, [None], [])),
                                ("cammy.cli.mpl_to_cv2_colormap", 2),
                                ("cammy.camera.aravis.acquisition_loop", None)]:
                stack.enter_context(patch(name, return_value=value))
            stack.enter_context(patch("cammy.trigger.nucleo.serial.Serial", serial))
            result = CliRunner().invoke(cli, ["tune", "--settings", str(self.settings), "--trigger-port", "/dev/test"],
                                        catch_exceptions=False)
        self.assertEqual(result.exit_code, 0, result.output)
        serial.assert_called_once()
        commands = [c.args[0].decode().strip() for c in port.write.call_args_list]
        self.assertEqual([c.split(",")[0] for c in commands],
                         ["HELLO", "CONFIG", "START", "STOP", "CONFIG", "START"] + ["STOP", "START"] * 4 + ["STOP"])
        self.assertEqual(commands[4], "CONFIG,1,1,100,3000,3750,10000,7475,1,8000,4422")
        port.close.assert_called_once()
        for key, gain in initial_gains.items():
            self.assertEqual(cameras[key].set_feature.call_args_list,
                             [call("GainAuto", "Off"), call("Gain", "2.5"), call("GainAuto", "Off"), call("Gain", str(gain))])
            cameras[key].initialize_acquisition_stream.assert_called_once()
            self.assertEqual(cameras[key].camera.get_region(), (0, 0, 36, 36))
            self.assertEqual(cameras[key]._payload, 432)
            dpg.set_value.assert_any_call(f"saturation_{key}", "Saturated pixels: 25.000%")
        self.assertIn('GAIN_DB=2.5  # Use the camera baseline', self.settings.read_text())
        saved = read_settings(self.settings)
        self.assertEqual(saved["FLUORESCENCE_EXPOSURE_US"], "8000")
        self.assertEqual(float(saved["FLUORESCENCE_ILLUMINATION_PERCENT"]), 74.75)
        self.assertEqual(saved["FPS"], "80.5")
        self.assertEqual(saved["SURFACE_TO_FLUORESCENCE_GAP_US"], "1250")
        self.assertEqual(saved["ROI_WIDTH"], "12")
        self.assertEqual(saved["ROI_HORIZONTAL"], "right")
        dpg.create_viewport.assert_called_once()
        self.assertEqual(dpg.add_raw_texture.call_count, len(cameras))
        dpg.destroy_context.assert_called_once()
