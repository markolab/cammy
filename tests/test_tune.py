from contextlib import ExitStack
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, Mock, call, patch

from click import ClickException
from click.testing import CliRunner
import numpy as np

from cammy.camera.aravis import AravisCamera
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
        session.initial_trigger({})
        camera = Mock()
        camera.camera.get_gain_bounds.return_value = (0, 12)
        with patch.object(session.trigger, "stop") as stop:
            for gain, width, volts in [("20", 9000, 2), ("nan", 9000, 2), ("", 10000, 2),
                                       ("", 9000, 11), ("", 0, 2)]:
                with self.subTest(gain=gain, width=width, volts=volts), self.assertRaises(ValueError):
                    session.apply({"cam": camera}, "fluorescence", gain, {"fluorescence": (width, volts, 50)})
            for fps in [0, -1, float("nan"), float("inf"), 1e-320, 120]:
                with self.subTest(fps=fps), self.assertRaises(ValueError):
                    session.apply({"cam": camera}, "fluorescence", "", {"fluorescence": (9000, 2, 50)}, fps=fps)
            for underfill in [-1, 101, float("nan")]:
                with self.subTest(underfill=underfill), self.assertRaisesRegex(ValueError, "Underfill"):
                    session.apply({"cam": camera}, "surface", "", {"surface": (1000, 3, underfill)})
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
        session.initial_trigger({})
        camera = Mock()
        port = session.trigger.dev = Mock()
        for bank, dark, exposure, expected_fill in [
            ("surface", 100, 1000, ["0", "5000"]),
            ("fluorescence", 0, 9000, ["0", "10000"]),
        ]:
            with self.subTest(bank=bank):
                port.readline.side_effect = [b"OK STOP\n", b"OK CONFIG\n", b"OK START\n"]
                session.apply({"cam": camera}, bank, "", {bank: (exposure, 3, dark)})
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
        session.initial_trigger({})
        camera = Mock()
        port = session.trigger.dev = Mock()
        port.readline.side_effect = [b"OK STOP\n", b"ERR CONFIG\n"]
        with self.assertRaisesRegex(ClickException, "ERR CONFIG"):
            session.apply({"A": camera}, "fluorescence", "", {"fluorescence": (8000, 3, 50)})
        commands = [c.args[0].split(b",", 1)[0].strip() for c in port.write.call_args_list]
        self.assertEqual(commands, [b"STOP", b"CONFIG"])
        camera.stop_acquisition.assert_not_called()
        self.assertEqual(session.mode, "surface")
        self.assertEqual(session.values, session.saved)

    def test_shortest_alternate_gap_at_startup_uses_roi_readout(self):
        with self.settings.open("a") as settings:
            settings.write("SURFACE_TO_FLUORESCENCE_GAP_US=0\n")
        session = TuneSession(self.settings, "alternate", "/dev/test")
        metadata = {"cam": {"SensorReadoutTime": 4000}}
        self.assertEqual(session.initial_trigger(metadata).steps, [(1000, 100), (9000, 9900)])
        edits = {"surface": (1000, 3, 0), "fluorescence": (9000, 2, 50)}
        with patch.object(session.trigger, "stop") as stop:
            with self.assertRaisesRegex(ValueError, "shortest estimated gaps"):
                session.apply({}, "alternate", "", edits, gap=0, fps=150, metadata=metadata)
            stop.assert_not_called()
        self.assertEqual(session.values["FPS"], "100")
        self.assertEqual(session.values["SURFACE_TO_FLUORESCENCE_GAP_US"], "100")

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

    def test_live_preview_stride_pauses_images_and_keeps_frame_counters(self):
        cameras = {k: Mock(id=k, _width=36, _height=36, _spoof_cameras=[], missed_frames=0, fps=100,
                           display_frame=(None, None), frame_count=0, total_frames=0) for k in ("A", "B")}
        for camera in cameras.values():
            camera.__class__ = AravisCamera
            camera.get_all_features.return_value = {"PixelFormat": "Mono8"}
        cam = cameras["A"]
        dpg = MagicMock()
        dpg.get_value.side_effect = lambda tag: 0 if tag.endswith("_min") else 255
        dpg.add_stage.return_value = "parked"
        dpg.texture_registry.return_value.__enter__.return_value = "live"
        moves = {"texture_A": [], "texture_B": []}
        def move_texture(tag, *, parent):
            moves[tag].append((dpg.is_dearpygui_running.call_count, parent))
        dpg.move_item.side_effect = move_texture
        frame = np.zeros((36, 36), dtype=np.uint8)
        drawn = []
        def next_frame():
            number = dpg.is_dearpygui_running.call_count
            if number == 8:  # GUI refresh without a new frame from either camera.
                return True
            if number > 8:
                return False
            field = dpg.add_input_int.call_args.kwargs
            self.assertEqual(field["default_value"], 2)
            if number in (4, 6):
                field["callback"](None, 0 if number == 4 else 1)
            cam.frame_count = cam.total_frames = number
            cam.display_frame = (frame, {})
            if number in (2, 5, 7):
                cameras["B"].frame_count += 1
                cameras["B"].total_frames += 1
                cameras["B"].display_frame = (frame, {})
            return True

        dpg.is_dearpygui_running.side_effect = next_frame
        def update(tag, value):
            if tag == "texture_A":
                drawn.append(cam.frame_count)
        dpg.set_value.side_effect = update
        with patch.dict("sys.modules", {"dearpygui": Mock(dearpygui=dpg), "dearpygui.dearpygui": dpg}), \
                patch("cammy.cli.get_all_camera_ids", return_value=dict.fromkeys(cameras, "aravis")), \
                patch("cammy.cli.initialize_cameras", return_value=cameras), \
                patch("cammy.cli.get_cpu_affinity", return_value=(None, [None], [])), \
                patch("cammy.cli.mpl_to_cv2_colormap", return_value=2), \
                patch("cammy.cli.time.perf_counter", side_effect=lambda: dpg.is_dearpygui_running.call_count), \
                patch("cammy.cli.time.sleep"), patch("cammy.camera.aravis.acquisition_loop"):
            result = CliRunner().invoke(cli, ["run", "--display-time-downsample", "2",
                                              "--stream-stats", "--stream-stats-interval", "5"], catch_exceptions=False)
        self.assertEqual(result.exit_code, 0, result.output)
        for camera in cameras.values():
            self.assertEqual(camera.log_stream_statistics.call_count, 2)  # GUI ticks 1 and 6.
        self.assertEqual(drawn, [1, 3, 6, 7])
        self.assertEqual(moves["texture_A"], [(2, "parked"), (3, "live"), (4, "parked"), (6, "live"), (8, "parked")])
        self.assertEqual(moves["texture_B"], [(3, "parked"), (7, "live"), (8, "parked")])
        dpg.set_value.assert_any_call(dpg.add_text.return_value, "0 missed / 5 total (0.0% missed)")
        cam.start_acquisition.assert_called_once()

    def test_tune_stream_stats_follows_file_and_command_line(self):
        for setting, flags, expected in [(None, [], True), ("off", [], False),
                                         ("off", ["--stream-stats"], True), ("on", ["--no-stream-stats"], False)]:
            with self.subTest(setting=setting, flags=flags):
                if setting is not None:
                    with self.settings.open("a") as f:
                        f.write(f"STREAM_STATS={setting}\n")
                with patch.object(cli.commands["run"], "callback") as preview:
                    result = CliRunner().invoke(cli, ["tune", "--settings", str(self.settings), *flags])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(preview.call_args.kwargs["stream_stats"], expected)

    def test_tune_stats_interval_setting_and_override(self):
        with self.settings.open("a") as f:
            f.write("STREAM_STATS_INTERVAL=5\n")
        for flags, seconds in [([], 5), (["--stream-stats-interval", "10"], 10)]:
            with patch.object(cli.commands["run"], "callback") as preview:
                result = CliRunner().invoke(cli, ["tune", "--settings", str(self.settings), *flags])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(preview.call_args.kwargs["stream_stats_interval"], seconds)

    def test_aravis_buffer_size_from_settings(self):
        for value, expected in [(None, 512), ("192", 192), ("0", None), ("bad", None)]:
            with self.subTest(value=value):
                if value is not None:
                    with self.settings.open("a") as f:
                        f.write(f"ARAVIS_BUFFER_SIZE={value}\n")
                with patch.object(cli.commands["run"], "callback") as preview:
                    result = CliRunner().invoke(cli, ["tune", "--settings", str(self.settings)])
                if expected is None:
                    self.assertNotEqual(result.exit_code, 0)
                    self.assertIn("ARAVIS_BUFFER_SIZE", result.output)
                    preview.assert_not_called()
                else:
                    self.assertEqual(result.exit_code, 0, result.output)
                    self.assertEqual(preview.call_args.kwargs["buffer_size"], expected)

    def test_tune_display_startup_setting_and_override(self):
        for setting, stride, flags, enabled, effective_stride in [
            ("", 0, [], False, 0), ("off", 4, [], False, 4),
            ("on", 0, [], True, 1), ("off", 4, ["--display", "ON"], True, 4),
        ]:
            with self.subTest(setting=setting, stride=stride, flags=flags):
                with self.settings.open("a") as f:
                    f.write(f"TUNE_DISPLAY={setting}\nPREVIEW_TIME_DOWNSAMPLE={stride}\n")
                with patch.object(cli.commands["run"], "callback") as preview:
                    result = CliRunner().invoke(cli, ["tune", "--settings", str(self.settings), *flags])
                self.assertEqual(result.exit_code, 0, result.output)
                params = preview.call_args.kwargs
                self.assertEqual(params["tuning"].display_on, enabled)
                self.assertEqual(params["display_time_downsample"], effective_stride)

    def test_tune_preview_switch_crop_suggest_and_save(self):
        # Run the real Click command, preview loop and bulk camera/serial control, without a rig.
        dpg = MagicMock()
        buttons = {}
        def add_button(label, callback, **attrs):
            buttons[label] = (callback, attrs.get("user_data"))
        dpg.add_button.side_effect = add_button
        with self.settings.open("a") as settings:
            settings.write("ROI_WIDTH=24\nSURFACE_TO_FLUORESCENCE_GAP_US=1250\nPREVIEW_TIME_DOWNSAMPLE=3\n")
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
                display = dpg.add_checkbox.call_args.kwargs
                self.assertFalse(display["default_value"])
                for key in cameras:
                    dpg.group.assert_any_call(tag=f"tune_images_{key}", show=False)
                for texture in dpg.add_raw_texture.call_args_list:
                    self.assertEqual(texture.kwargs["parent"], dpg.add_stage.return_value)
                display["callback"](None, True)
                stride = next(c.kwargs for c in dpg.add_input_int.call_args_list if c.kwargs["tag"] == "preview_time_downsample")
                self.assertEqual(stride["default_value"], 3)
                stride["callback"](None, 1)
                self.assertIn("Slowest sensor readout 1.50 ms", fields["timing_advice"])
                self.assertIn("gap 1250 µs → fluorescence", fields["timing_advice"])
                gap_widget = next(c.kwargs for c in dpg.add_input_int.call_args_list if c.kwargs["tag"] == "surface_gap_us")
                self.assertEqual(gap_widget["default_value"], 1250)
                fields["surface_gap_us"] = 0
                buttons["Suggest alternating timing"][0]()
                self.assertIn("gap 100 µs → fluorescence", fields["timing_advice"])
                apply, bank = buttons["Apply fluorescence"]
                apply(None, None, bank)
                fields["fluorescence_us"] = 8500  # Draft survives crop until Apply alternate.
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
                apply, mode = buttons["Apply alternate"]
                apply(None, None, mode)
                self.assertIn("Applied surface-to-fluorescence gap 100 µs", fields["tune_status"])
                for cam in cameras.values():
                    cam.frame_count, cam.display_frame = 0, (None, None)
            elif tick == 9:
                clock.perf_counter.return_value = 200
                for cam in cameras.values():
                    cam.frame_count, cam.missed_frames = 120, 2
                dpg.add_checkbox.call_args.kwargs["callback"](None, False)
                for key, cam in cameras.items():
                    dpg.configure_item.assert_any_call(f"tune_images_{key}", show=False)
                    dpg.move_item.assert_any_call(f"texture_{key}", parent=dpg.add_stage.return_value)
                    cam.frame_count, cam.missed_frames = 128, 3
                    cam.display_frame = (object(), {})  # Raises if used as an image while off.
            elif tick == 10:
                dpg.set_value.assert_any_call(dpg.add_text.return_value, "Frame IDs: 1 missed / 9 total (11.1% missed)")
                clock.perf_counter.return_value = 201
            elif tick == 11:
                dpg.set_value.assert_any_call(dpg.add_text.return_value, "8.0 received / 80.5 target FPS")
                dpg.add_checkbox.call_args.kwargs["callback"](None, True)
                for key, cam in cameras.items():
                    dpg.move_item.assert_any_call(f"texture_{key}", parent=dpg.texture_registry.return_value.__enter__.return_value)
                    cam.frame_count, cam.display_frame = 132, (frame[:, :cam._width], {})
            elif tick == 12:
                dpg.set_value.assert_any_call(dpg.add_text.return_value, "Frame IDs: 0 missed / 4 total (0.0% missed)")
                dpg.add_checkbox.call_args.kwargs["callback"](None, False)
                buttons["Save applied settings"][0]()
        dpg.run_callbacks.side_effect = interact
        dpg.is_dearpygui_running.side_effect = [True] * 12 + [False]
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
                                     b"OK CONFIG\n", b"OK START\n"] + [b"OK STOP\n", b"OK START\n"] * 4 + [
                                     b"OK STOP\n", b"OK CONFIG\n", b"OK START\n", b"OK STOP\n"]
        with ExitStack() as stack:
            stack.enter_context(patch.dict("sys.modules", {"dearpygui": Mock(dearpygui=dpg), "dearpygui.dearpygui": dpg}))
            clock = stack.enter_context(patch("cammy.cli.time", wraps=time))
            clock.monotonic.side_effect = [0, .01, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5]
            clock.sleep.return_value = None
            for name, value in [("cammy.cli.get_all_camera_ids", {k: "aravis" for k in cameras}),
                                ("cammy.cli.initialize_cameras", cameras),
                                ("cammy.cli.get_cpu_affinity", (None, [None], [])),
                                ("cammy.cli.mpl_to_cv2_colormap", 2),
                                ("cammy.camera.aravis.acquisition_loop", None)]:
                stack.enter_context(patch(name, return_value=value))
            stack.enter_context(patch("cammy.trigger.nucleo.serial.Serial", serial))
            result = CliRunner().invoke(cli, ["tune", "--settings", str(self.settings), "--trigger-port", "/dev/test",
                                              "--mode", "alternate", "--stream-stats", "--display", "off"],
                                        catch_exceptions=False)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(any(c.args[1] == "0.0 received / 80.5 target FPS" for c in dpg.set_value.call_args_list
                            if isinstance(c.args[1], str)))
        serial.assert_called_once()
        commands = [c.args[0].decode().strip() for c in port.write.call_args_list]
        self.assertEqual([c.split(",")[0] for c in commands],
                         ["HELLO", "CONFIG", "START", "STOP", "CONFIG", "START"] + ["STOP", "START"] * 4
                         + ["STOP", "CONFIG", "START", "STOP"])
        self.assertEqual(commands[1], "CONFIG,1,2,100,3000,2000,10000,5000,2,1000,1250,9000,8750")
        self.assertEqual(commands[4], "CONFIG,1,1,100,3000,3750,10000,7475,1,8000,4422")
        self.assertEqual(commands[-3], "CONFIG,1,2,100,3000,3750,10000,7475,2,1000,100,8500,15245")
        port.close.assert_called_once()
        for key, gain in initial_gains.items():
            self.assertEqual(cameras[key].set_feature.call_args_list,
                             [call("GainAuto", "Off"), call("Gain", "2.5"), call("GainAuto", "Off"), call("Gain", str(gain))])
            cameras[key].initialize_acquisition_stream.assert_called_once()
            self.assertEqual(cameras[key].reset_tune_trial.call_args_list, [call(2)] + [call(1)] * 5 + [call(2)])
            self.assertEqual(cameras[key].camera.get_region(), (0, 0, 36, 36))
            self.assertEqual(cameras[key]._payload, 432)
            dpg.set_value.assert_any_call(f"saturation_{key}", "Saturated pixels: 25.000%")
        self.assertIn('GAIN_DB=2.5  # Use the camera baseline', self.settings.read_text())
        saved = read_settings(self.settings)
        self.assertEqual(saved["FLUORESCENCE_EXPOSURE_US"], "8500")
        self.assertEqual(float(saved["FLUORESCENCE_ILLUMINATION_PERCENT"]), 74.75)
        self.assertEqual(saved["FPS"], "80.5")
        self.assertEqual(saved["PREVIEW_TIME_DOWNSAMPLE"], "1")
        self.assertEqual(saved["TUNE_DISPLAY"], "off")
        self.assertEqual(saved["SURFACE_TO_FLUORESCENCE_GAP_US"], "100")
        self.assertEqual(saved["ROI_WIDTH"], "12")
        self.assertEqual(saved["ROI_HORIZONTAL"], "right")
        dpg.create_viewport.assert_called_once()
        self.assertEqual(dpg.add_raw_texture.call_count, len(cameras))
        dpg.destroy_context.assert_called_once()
