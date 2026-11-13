import errno
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from cammy.record.base import BaseRecord
from cammy.record.video import RawVideoRecorder


class WritevTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.video_path = Path(self.directory.name) / "video.dat"
        self.recorder = RawVideoRecorder(
            filename=str(self.video_path), timestamp_fields=["frame_id"]
        )
        self.recorder.open_writer()
        self.frames = [b"abcd", b"EFGHI", b"jklmnop"]
        self.timestamps = [{"frame_id": i} for i in range(3)]

    def tearDown(self):
        self.recorder.close_writer()
        self.directory.cleanup()

    def assert_output(self, expected, timestamps="frame_id\t\n0\t\n1\t\n2\t\n"):
        self.recorder._tstamp_file.flush()
        self.assertEqual(self.video_path.read_bytes(), expected)
        self.assertEqual(self.video_path.with_suffix(".txt").read_text(), timestamps)

    def test_complete_batch_matches_concatenation(self):
        self.recorder.write_batch(self.frames, self.timestamps)
        self.assert_output(b"".join(self.frames))

    def test_partial_writes_inside_frames_and_at_boundaries(self):
        limits = iter([6, 3, 2, 100])

        def short_write(fd, buffers):
            return os.write(fd, b"".join(buffers)[:next(limits)])

        with patch("cammy.record.video.os.writev", side_effect=short_write) as writev:
            self.recorder.write_batch(self.frames, self.timestamps)
        self.assertEqual(writev.call_count, 4)
        self.assert_output(b"".join(self.frames))

    def test_vector_count_limit(self):
        self.recorder._iov_max = 2
        real_writev = os.writev

        def limited_write(fd, buffers):
            self.assertLessEqual(len(buffers), 2)
            return real_writev(fd, buffers)

        with patch("cammy.record.video.os.writev", side_effect=limited_write) as writev:
            self.recorder.write_batch(self.frames, self.timestamps)
        self.assertEqual(writev.call_count, 2)
        self.assert_output(b"".join(self.frames))

    def test_more_buffers_than_actual_system_limit(self):
        frames = [b"x"] * (self.recorder._iov_max + 1)
        self.recorder.write_batch(frames, None)
        self.assert_output(b"".join(frames), "frame_id\t\n")

    def test_empty_buffers(self):
        with patch("cammy.record.video.os.writev") as writev:
            self.recorder.write_batch([], [])
            self.recorder.write_batch([b"", memoryview(b"")], None)
        writev.assert_not_called()
        self.recorder.write_batch([b"", *self.frames, b""], self.timestamps)
        self.assert_output(b"".join(self.frames))

    def test_interruption_after_partial_progress(self):
        calls = 0
        real_writev = os.writev

        def interrupted_write(fd, buffers):
            nonlocal calls
            calls += 1
            if calls == 1:
                return os.write(fd, b"".join(buffers)[:3])
            if calls == 2:
                raise InterruptedError()
            return real_writev(fd, buffers)

        with patch("cammy.record.video.os.writev", side_effect=interrupted_write):
            self.recorder.write_batch(self.frames, self.timestamps)
        self.assertEqual(calls, 3)
        self.assert_output(b"".join(self.frames))

    def test_zero_progress_and_disk_error_do_not_write_timestamps(self):
        for failure in (0, OSError(errno.ENOSPC, "No space left on device")):
            with self.subTest(failure=failure):
                self.recorder._video_file.seek(0)
                self.recorder._video_file.truncate()
                calls = 0

                def failing_write(fd, buffers):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        return os.write(fd, b"".join(buffers)[:3])
                    if isinstance(failure, OSError):
                        raise failure
                    return failure

                with patch("cammy.record.video.os.writev", side_effect=failing_write):
                    with self.assertRaises(OSError):
                        self.recorder.write_batch(self.frames, self.timestamps)
                self.assertEqual(calls, 2)
                self.assert_output(b"abc", "frame_id\t\n")

    def test_direct_write_data_uses_partial_write_handling(self):
        def short_write(fd, buffers):
            return os.write(fd, b"".join(buffers)[:2])

        with patch("cammy.record.video.os.writev", side_effect=short_write):
            self.recorder.write_data(b"abcdef", [{"frame_id": 0}])
        self.assert_output(b"abcdef", "frame_id\t\n0\t\n")

    def test_contiguous_array_buffer_preserves_pixel_bytes(self):
        frame = np.arange(12, dtype=np.uint16).reshape(3, 4)
        self.recorder.write_batch([frame, memoryview(frame)], None)
        self.assert_output(frame.tobytes() * 2, "frame_id\t\n")

    def test_base_batch_hook_preserves_existing_write_data_interface(self):
        recorder = Mock(spec=BaseRecord)
        BaseRecord.write_batch(recorder, self.frames, self.timestamps)
        recorder.write_data.assert_called_once_with(b"".join(self.frames), self.timestamps)

    def test_run_does_not_replay_batch_after_keyboard_interrupt(self):
        self.recorder.batch_size = 1
        socket = Mock()
        socket.recv_json.return_value = {"timestamps": self.timestamps[0]}
        socket.recv.return_value = self.frames[0]
        context = Mock()
        context.socket.return_value = socket
        calls = 0

        def interrupted_write(fd, buffers):
            nonlocal calls
            calls += 1
            if calls == 1:
                return os.write(fd, b"".join(buffers)[:2])
            raise KeyboardInterrupt()

        with patch("cammy.record.base.zmq.Context", return_value=context), \
                patch("cammy.record.base.os.nice"), \
                patch.object(self.recorder, "open_writer"), \
                patch("cammy.record.video.os.writev", side_effect=interrupted_write):
            with self.assertRaises(KeyboardInterrupt):
                self.recorder.run()
        self.assertEqual(calls, 2)
        self.assertEqual(self.recorder.frame_batch, [self.frames[0]])
        self.assert_output(b"ab", "frame_id\t\n")


if __name__ == "__main__":
    unittest.main()
