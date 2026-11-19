import logging
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from cammy.camera.aravis import Aravis, AravisCamera
from cammy.cli import cli


class StreamStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.camera = AravisCamera.__new__(AravisCamera)
        self.camera.id = "test-camera"
        self.camera.logger = logging.getLogger("test-stream-statistics")
        self.camera.frame_count = 1492
        self.camera.missed_frames = 18

    def test_gige_logs_frame_and_packet_counters_from_their_respective_apis(self):
        self.camera.stream = Mock(spec=Aravis.GvStream)
        self.camera.stream.get_n_buffers.return_value = (3, 508)
        self.camera.recorder = SimpleNamespace(frames_written=1400)
        self.camera.recording_drops = 12

        with patch.object(Aravis.Stream, "get_statistics", return_value=(2000, 7, 11)) as frames, \
                patch.object(Aravis.GvStream, "get_statistics", return_value=(17, 23)) as packets, \
                self.assertLogs("test-stream-statistics", level="INFO") as logs:
            self.camera.log_stream_statistics()

        self.camera.stream.get_n_buffers.assert_called_once_with()
        frames.assert_called_once_with(self.camera.stream)
        packets.assert_called_once_with(self.camera.stream)
        # An unqualified call would use GvStream's two packet counters as frame counters.
        self.camera.stream.get_statistics.assert_not_called()
        self.assertIn(
            "test-camera stream: free_buffers=3 queued_buffers=508 completed=2000 failures=7 "
            "underruns=11 acquired=1492 missed=18 written=1400 recording_drops=12 "
            "resent_packets=17 missing_packets=23",
            logs.output[0],
        )
        self.camera.stream.pop_buffer.assert_not_called()
        self.camera.stream.push_buffer.assert_not_called()

    def test_non_gige_preview_logs_without_packet_counters_or_a_recorder(self):
        self.camera.stream = Mock(spec=Aravis.Stream)
        self.camera.stream.get_n_buffers.return_value = (512, 0)
        self.camera.frame_count = 0
        self.camera.missed_frames = 0

        with patch.object(Aravis.Stream, "get_statistics", return_value=(0, 0, 0)), \
                patch.object(Aravis.GvStream, "get_statistics") as packets, \
                self.assertLogs("test-stream-statistics", level="INFO") as logs:
            self.camera.log_stream_statistics()

        packets.assert_not_called()
        self.assertIn(
            "test-camera stream: free_buffers=512 queued_buffers=0 completed=0 failures=0 "
            "underruns=0 acquired=0 missed=0 written=0 recording_drops=0",
            logs.output[0],
        )
        self.assertNotIn("missing_packets", logs.output[0])

    def test_cli_statistics_are_opt_in(self):
        for args, expected in [([], False), (["--stream-stats"], True)]:
            with self.subTest(args=args), cli.commands["run"].make_context("run", args) as context:
                self.assertEqual(context.params["stream_stats"], expected)


if __name__ == "__main__":
    unittest.main()
