import unittest
from unittest.mock import Mock, call, patch

from cammy.util import initialize_camera, initialize_cameras


class CameraInitializationTests(unittest.TestCase):
    @patch("cammy.util.initialize_camera")
    def test_failed_camera_is_logged_and_successful_camera_is_kept(self, initialize):
        camera = Mock()
        initialize.side_effect = [RuntimeError("ExposureOverlapTimeMax is unavailable"), camera]
        ids = {"Basler-bad": "aravis", "Basler-good": "aravis"}
        configs = {"genicam": {"Basler": {"a": {"ExposureOverlapTimeMax": "5000"}}}}

        with self.assertLogs("cammy.util", level="ERROR") as logs:
            cameras = initialize_cameras(ids, configs, buffer_size=5)

        self.assertEqual(cameras, {"Basler-good": camera})
        self.assertIn("Basler-bad", logs.output[0])
        self.assertIn("ExposureOverlapTimeMax is unavailable", logs.output[0])
        self.assertEqual(initialize.call_count, 2)

    @patch("cammy.util.initialize_camera", side_effect=ValueError("Invalid camera option"))
    def test_all_failures_are_logged(self, initialize):
        ids = {"Basler-one": "aravis", "Basler-two": "aravis"}
        with self.assertLogs("cammy.util", level="ERROR") as logs:
            cameras = initialize_cameras(ids, {})
        self.assertEqual(cameras, {})
        self.assertEqual(len(logs.output), 2)

    @patch("cammy.util.initialize_camera", side_effect=KeyboardInterrupt)
    def test_keyboard_interrupt_is_not_swallowed(self, initialize):
        with self.assertRaises(KeyboardInterrupt):
            initialize_cameras({"Basler-one": "aravis"}, {})

    @patch("cammy.camera.aravis.AravisCamera")
    def test_feature_error_identifies_camera_setting_and_value(self, camera_class):
        original_error = ValueError("Feature is not writable")
        camera_class.return_value.set_feature.side_effect = [None, original_error]
        config = {
            "ExposureTime": "1600",
            "ExposureOverlapTimeMax": "5000",
            "ExposureMode": "TriggerWidth",
        }

        with self.assertRaisesRegex(
            RuntimeError, "Camera Basler-one: failed to set ExposureOverlapTimeMax='5000'"
        ) as caught:
            initialize_camera("Basler-one", "aravis", config)

        self.assertIs(caught.exception.__cause__, original_error)
        self.assertEqual(camera_class.return_value.set_feature.call_args_list, [
            call("ExposureTime", "1600"), call("ExposureOverlapTimeMax", "5000")
        ])


if __name__ == "__main__":
    unittest.main()
