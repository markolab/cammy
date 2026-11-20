import unittest
from contextlib import nullcontext
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, Mock, PropertyMock, call, patch

import toml
from click.testing import CliRunner

from cammy.camera.aravis import Aravis, AravisCamera
from cammy.cli import cli
from cammy.record.video import RawVideoRecorder
from cammy.roi import centered_region, crop_region, set_region, timing_advice


class RoiTests(unittest.TestCase):
    def test_recording_metadata_uses_hardware_roi_after_cli_override(self):
        runner = CliRunner()
        dpg = MagicMock()
        dpg.get_value.side_effect = lambda tag: 0 if tag.endswith("_min") else 255
        written = Event()
        write_batch = RawVideoRecorder.write_batch

        def write_and_notify(recorder, frames, timestamps):
            write_batch(recorder, frames, timestamps)
            written.set()

        def finish_preview():
            self.assertTrue(written.wait(3), "Fake camera did not record a frame")
            return dpg.render_dearpygui_frame.call_count == 0

        def gui_sleep(seconds):
            if seconds == 0.1:
                written.clear()
                self.assertTrue(written.wait(3), "Writer stalled while the status GUI was asleep")

        dpg.is_dearpygui_running.side_effect = finish_preview
        native = Aravis.Camera.new_with_device(Aravis.FakeDevice.new("roi-test"))
        native.set_region(20, 0, 256, 256)
        with runner.isolated_filesystem(), \
                patch.dict("sys.modules", {"dearpygui": Mock(dearpygui=dpg), "dearpygui.dearpygui": dpg}), \
                patch("cammy.cli.get_all_camera_ids", return_value={"Fake": "aravis"}), \
                patch.object(Aravis.Camera, "new", return_value=native), \
                patch("cammy.cli.get_cpu_affinity", return_value=({0}, [1], [2])), \
                patch("os.sched_getaffinity", return_value={0, 1, 2}, create=True), \
                patch("os.sched_setaffinity", create=True), \
                patch("cammy.cli.mpl_to_cv2_colormap", return_value=2), \
                patch("cammy.cli.time.sleep", side_effect=gui_sleep) as sleep, patch("os.nice", create=True), \
                patch.object(RawVideoRecorder, "write_batch", write_and_notify), \
                patch("cammy.camera.aravis.AravisCamera.get_all_features", return_value={"PixelFormat": "Mono8"}):
            options = Path("camera_options.toml")
            config = "[genicam.Fake.a]\nWidth = 128\nOffsetX = 32\nHeight = 96\nOffsetY = 24\n"
            for args, expected, requested_width, display_stride in [
                ([], (32, 24, 128, 96), None, 1),
                (["--roi-width", "512", "--roi-horizontal", "right"], (1536, 24, 512, 96), 512, 0),
                (["--roi-width", "0", "--roi-height", "192", "--roi-vertical", "bottom"], (0, 1856, 2048, 192), 0, 0),
                (["--roi-horizontal", "center"], (960, 24, 128, 96), None, 1),
            ]:
                with self.subTest(args=args):
                    # Even an unusable TOML offset must be ignored on the overridden axis.
                    options.write_text(config.replace("OffsetX = 32", 'OffsetX = "invalid"') if "right" in args else config)
                    written.clear()
                    dpg.reset_mock()
                    sleep.reset_mock()
                    # Acquisition can publish frames, but the status-only GUI must never read them.
                    pixels = (nullcontext() if display_stride else
                              patch.object(AravisCamera, "display_frame", new_callable=PropertyMock, create=True))
                    with pixels as display_frame, self.assertLogs(level="INFO") as logs:
                        result = runner.invoke(cli, ["run", "--record", "--frame-writer-batch-size", "1",
                                                     "--display-time-downsample", str(display_stride), *args],
                                               catch_exceptions=False)
                    self.assertEqual(result.exit_code, 0, result.output)
                    self.assertEqual(sleep.call_args_list.count(call(0.1)), int(display_stride == 0))
                    if not display_stride:
                        self.assertNotIn(call(), display_frame.call_args_list)
                    self.assertEqual(dpg.add_raw_texture.call_count, int(display_stride > 0))
                    self.assertTrue(any(isinstance(c.args[1], str) and "Writer:" in c.args[1]
                                        for c in dpg.set_value.call_args_list))
                    messages = [record.getMessage() for record in logs.records]
                    self.assertRegex("\n".join(messages), r"Fake acquisition TID \d+ pinned to CPU 1")
                    self.assertRegex("\n".join(messages), r"Fake\.dat writer TID \d+ pinned to CPU 2")
                    if args:
                        self.assertIn("Requested ROI: " + " ".join(args), messages)
                    x, y, w, h = expected
                    self.assertIn(f"Fake ROI readback: Width={w} Height={h} OffsetX={x} OffsetY={y} (pixels)", messages)
                    session = sorted(Path.cwd().glob("session_*"))[-1]
                    metadata = toml.load(session / "metadata.toml")
                    features = metadata["camera_metadata"]["Fake"]
                    self.assertEqual(tuple(features[k] for k in ("OffsetX", "OffsetY", "Width", "Height")), expected)
                    self.assertEqual(metadata["cli_parameters"].get("roi_width"), requested_width)
                    frame_count = len((session / "Fake.txt").read_text().splitlines()) - 1
                    self.assertGreater(frame_count, 0)
                    self.assertEqual((session / "Fake.dat").stat().st_size, frame_count * features["Width"] * features["Height"])
            self.assertEqual(tuple(native.get_region()), (32, 24, 128, 96))

    def test_camera_payload_tracks_the_centered_crop_and_restore(self):
        camera = Aravis.Camera.new_with_device(Aravis.FakeDevice.new("test-roi"))
        camera.set_region(20, 40, 256, 192)
        original = tuple(camera.get_region())
        for width, height, expected in [(128, 0, (84, 40, 128, 192)), (0, 64, (20, 104, 256, 64)),
                                        (128, 64, (84, 104, 128, 64)), (0, 0, original)]:
            set_region(camera, centered_region(camera, original, width, height))
            self.assertEqual(tuple(camera.get_region()), expected)
            self.assertEqual(camera.get_payload(), expected[2] * expected[3])

        for horizontal, x in (("left", 20), ("center", 84), ("right", 148)):
            for vertical, y in (("top", 40), ("center", 104), ("bottom", 168)):
                region = crop_region(camera, original, 128, 64, horizontal, vertical)
                set_region(camera, region)
                self.assertEqual(tuple(camera.get_region()), (x, y, 128, 64))
        self.assertEqual(crop_region(camera, original, 0, 0, "right", "bottom"), original)

    def test_size_and_center_respect_camera_increments(self):
        camera = Mock()
        camera.get_width_bounds.return_value = (8, 256)
        camera.get_width_increment.return_value = 4
        camera.get_x_offset_increment.return_value = 8
        base = (16, 0, 256, 192)
        self.assertEqual(centered_region(camera, base, 132, 0), (72, 0, 132, 192))
        self.assertEqual(crop_region(camera, base, 132, 0, "right"), (136, 0, 132, 192))
        for width in (-1, 4, 131, 260):
            with self.assertRaisesRegex(ValueError, 'Width'):
                centered_region(camera, base, width, 0)
        camera.set_region.assert_not_called()

    def test_alternating_budget_and_underfill_use_the_slowest_camera(self):
        meta = {"A": {"ReadoutTimeAbs": 700}, "B": {"SensorReadoutTime": 1600}}
        for dark, lit in [(0, 10000), (75, 2500), (100, 0)]:
            tip = timing_advice(meta, 1000, 10000, dark, 80, 100, 1.5)
            self.assertIn("Slowest sensor readout 1.60 ms", tip)
            self.assertIn(f"gap 100 µs → fluorescence 10000 µs ({lit} µs lit) → wait 13900 µs (2100 µs min)", tip)
            self.assertIn("151.51 FPS total", tip)
            self.assertIn("up to 21800 µs", tip)

    def test_overlapping_exposures_budget_both_falling_edges(self):
        meta = {"A": {"SensorReadoutTime": 6000}}
        tip = timing_advice(meta, 1000, 10000, 80, 100, 100, 1.5)
        self.assertIn("gap 100 µs → fluorescence 10000 µs (2000 µs lit) → wait 8900 µs (6500 µs min)", tip)
        self.assertIn("up to 12400 µs", tip)
        meta["A"]["ExposureOverlapTimeMax"] = 800
        self.assertIn("gap 6700 µs", timing_advice(meta, 1000, 10000, 80, 100, 100, 1.5))
        meta["A"]["ExposureOverlapTimeMax"] = 1500
        with self.assertRaisesRegex(ValueError, "ExposureOverlapTimeMax"):
            timing_advice(meta, 1000, 10000, 80, 100, 100, 1.5)
        meta["A"]["OverlapMode"] = "Off"
        self.assertIn("gap 7500 µs", timing_advice(meta, 1000, 10000, 80, 100, 100, 1.5))

    def test_requested_gap_reserves_only_the_surface_to_fluorescence_interval(self):
        meta = {"A": {"SensorReadoutTime": 6000}}
        tip = timing_advice(meta, 1000, 10000, 50, 100, 100, 1.5, requested_gap=3000)
        self.assertIn("Shortest surface-to-fluorescence gap 100 µs", tip)
        self.assertIn("gap 3000 µs → fluorescence 10000 µs (5000 µs lit) → wait 6500 µs", tip)
        self.assertIn("97.56 FPS total", tip)
        self.assertIn("exceeds the target period by 500 µs", tip)
        self.assertIn("up to 9500 µs", tip)

        # A short draft exposure needs extra readout time. A longer candidate can reclaim it.
        tip = timing_advice(meta, 1000, 5000, 50, 100, 100, 1.5, requested_gap=500)
        self.assertIn("gap 2500 µs → fluorescence 5000 µs", tip)
        self.assertIn("up to 12000 µs", tip)
        for invalid in (-1, .5):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "gap"):
                timing_advice(meta, 1000, 5000, 50, 100, 100, 1.5, requested_gap=invalid)

    def test_unknown_sensor_readout_identifies_camera_even_when_link_data_is_available(self):
        for bad in (None, 0, -1, float("nan"), float("inf")):
            meta = {"A": {"ReadoutTimeAbs": 100}, "B": {"SensorReadoutTime": bad, "GevSCBWA": 5e8}}
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "B did not report a valid sensor readout"):
                timing_advice(meta, 1000, 10000, 50, 100, 100, 1.5)
