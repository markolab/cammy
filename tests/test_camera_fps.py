import unittest
from unittest.mock import Mock, patch

import numpy as np

from cammy.camera.aravis import Aravis, AravisCamera


class CameraFpsTests(unittest.TestCase):
    def make_camera(self, fps_stride=1):
        with patch("cammy.camera.aravis.Aravis") as aravis, patch("builtins.print"):
            device = aravis.Camera.new.return_value
            device.is_uv_device.return_value = False
            device.is_gv_device.return_value = False
            camera = AravisCamera("test", fps_stride=fps_stride)
        camera.stream = Mock()
        camera._array_from_buffer_address = Mock(return_value=np.zeros((2, 2), dtype=np.uint16))
        return camera

    def pop_frame(self, camera, frame_id, timestamp_ms):
        buffer = Mock()
        buffer.get_status.return_value = Aravis.BufferStatus.SUCCESS
        buffer.get_frame_id.return_value = frame_id
        buffer.get_timestamp.return_value = round(timestamp_ms * 1e6)
        buffer.get_system_timestamp.return_value = round(timestamp_ms * 1e6)
        camera.stream.timeout_pop_buffer.return_value = buffer
        camera.try_pop_frame()

    def test_single_channel_uses_adjacent_frames(self):
        camera = self.make_camera()
        self.pop_frame(camera, 1, 0)
        self.assertTrue(np.isnan(camera.fps))
        for frame_id, timestamp in [(2, 10), (3, 20), (4, 30)]:
            self.pop_frame(camera, frame_id, timestamp)
            self.assertAlmostEqual(camera.fps, 100)

    def test_alternate_uses_same_bank_with_unequal_exposure_intervals(self):
        camera = self.make_camera(fps_stride=2)
        self.pop_frame(camera, 1, 0)
        self.pop_frame(camera, 2, 6)
        self.assertTrue(np.isnan(camera.fps))
        for frame_id, timestamp in [(3, 20), (4, 26), (5, 40)]:
            self.pop_frame(camera, frame_id, timestamp)
            self.assertAlmostEqual(camera.fps, 50)

    def test_missing_frame_keeps_light_banks_aligned(self):
        camera = self.make_camera(fps_stride=2)
        self.pop_frame(camera, 1, 0)
        # Frame 2 is missing; frame 3 still compares with the same bank in frame 1.
        for frame_id, timestamp in [(3, 20), (4, 26), (5, 40)]:
            self.pop_frame(camera, frame_id, timestamp)
            self.assertAlmostEqual(camera.fps, 50)
        self.assertEqual(camera.missed_frames, 1)


if __name__ == "__main__":
    unittest.main()
