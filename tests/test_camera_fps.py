import unittest
from unittest.mock import Mock, patch

import numpy as np

from cammy.camera.aravis import Aravis, AravisCamera


class CameraFpsTests(unittest.TestCase):
    def make_camera(self, fps_stride=1, gige=False, extended_ids=False):
        with patch("cammy.camera.aravis.Aravis") as aravis, patch("builtins.print"), \
                patch.object(AravisCamera, "get_feature", return_value="On" if extended_ids else "Off"):
            device = aravis.Camera.new.return_value
            device.is_uv_device.return_value = False
            device.is_gv_device.return_value = gige
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

    def test_session_counts_start_at_first_received_frame(self):
        for first_id in [0, 48000]:
            with self.subTest(first_id=first_id):
                camera = self.make_camera()
                self.pop_frame(camera, first_id, 0)
                self.pop_frame(camera, first_id + 1, 10)
                self.assertEqual(camera.total_frames, 2)
                self.assertEqual(camera.missed_frames, 0)

    def test_new_tune_trial_drops_old_buffers_and_resets_counts_and_bank_phase(self):
        camera = self.make_camera()
        self.pop_frame(camera, 12, 0)
        self.pop_frame(camera, 15, 30)
        stale_buffer = Mock()
        camera.stream.try_pop_buffer.side_effect = [stale_buffer, None]
        camera.reset_tune_trial(2)
        camera.stream.push_buffer.assert_called_with(stale_buffer)
        self.assertEqual((camera.frame_count, camera.missed_frames, camera.total_frames), (0, 0, 0))
        self.assertTrue(np.isnan(camera.fps))
        for frame_id, timestamp in [(40, 100), (41, 106), (42, 120)]:
            self.pop_frame(camera, frame_id, timestamp)
        self.assertEqual((camera.missed_frames, camera.total_frames, camera.fps), (0, 3, 50))

    def test_legacy_rollover_keeps_counts_and_alternate_fps_aligned(self):
        camera = self.make_camera(fps_stride=2, gige=True)
        for frame_id, timestamp in [(65534, 0), (65535, 6), (1, 20), (2, 26)]:
            self.pop_frame(camera, frame_id, timestamp)
        self.assertEqual(camera.frame_count, 4)
        self.assertEqual(camera.total_frames, 4)
        self.assertEqual(camera.missed_frames, 0)
        self.assertAlmostEqual(camera.fps, 50)

    def test_missing_frames_across_legacy_rollover_are_counted(self):
        camera = self.make_camera(gige=True)
        for frame_id, timestamp in [(65533, 0), (65535, 20), (2, 40)]:
            self.pop_frame(camera, frame_id, timestamp)
        self.assertEqual(camera.frame_count, 3)
        self.assertEqual(camera.total_frames, 5)
        self.assertEqual(camera.missed_frames, 2)

    def test_extended_ids_continue_past_65535(self):
        camera = self.make_camera(fps_stride=2, gige=True, extended_ids=True)
        for frame_id, timestamp in [(65534, 0), (65535, 6), (65536, 20), (65537, 26)]:
            self.pop_frame(camera, frame_id, timestamp)
        self.assertEqual(camera.total_frames, 4)
        self.assertEqual(camera.missed_frames, 0)
        self.assertAlmostEqual(camera.fps, 50)

    def test_duplicates_and_backward_ids_do_not_subtract_or_add_large_gaps(self):
        for gige in [False, True]:
            with self.subTest(gige=gige):
                camera = self.make_camera(gige=gige)
                for frame_id, timestamp in [(100, 0), (101, 10), (101, 10), (1, 20), (2, 30)]:
                    self.pop_frame(camera, frame_id, timestamp)
                self.assertEqual(camera.total_frames, 5)
                self.assertEqual(camera.missed_frames, 0)
                self.assertAlmostEqual(camera.fps, 100)


if __name__ == "__main__":
    unittest.main()
