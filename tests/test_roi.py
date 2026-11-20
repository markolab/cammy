import unittest
from unittest.mock import Mock

from cammy.camera.aravis import Aravis
from cammy.roi import centered_region, crop_region, set_region, timing_advice


class RoiTests(unittest.TestCase):
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
