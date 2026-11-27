import errno
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import zmq

from cammy.record.video import RawVideoRecorder


class RecordShutdownTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.video_path = Path(self.directory.name) / "video.dat"
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUSH)
        self.publisher.setsockopt(zmq.SNDTIMEO, 2000)
        self.publisher.bind("inproc://shutdown-test")
        self.recorder = RawVideoRecorder(
            filename=str(self.video_path), timestamp_fields=["frame_id"],
            batch_size=2, zmq_address="inproc://shutdown-test",
        )
        # A broken shutdown should fail the test without hanging the test runner.
        self.recorder.daemon = True
        for patcher in (
            patch("cammy.record.base.zmq.Context.instance", return_value=self.context),
            patch("cammy.record.base.os.nice"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.recorder.stop()
        if self.recorder.ident is not None:
            self.recorder.join(timeout=3)
        self.publisher.close(linger=0)
        if not self.recorder.is_alive():
            self.context.term()
        self.directory.cleanup()
        self.assertFalse(self.recorder.is_alive(), "Writer did not finish shutdown")

    def test_stop_before_start_is_not_overwritten(self):
        self.recorder.stop()
        self.recorder.start()
        self.recorder.join(timeout=3)
        self.assertFalse(self.recorder.is_alive())
        self.assertIsNone(self.recorder.error)
        self.assertTrue(self.recorder.ready.is_set())
        self.assertEqual(self.recorder.frames_written, 0)
        self.assertTrue(self.recorder._video_file.closed)
        self.assertTrue(self.recorder._tstamp_file.closed)

    def test_idle_writer_stops_without_another_frame(self):
        self.recorder.start()
        self.assertTrue(self.recorder.ready.wait(2))
        self.recorder.stop()
        self.recorder.join(timeout=3)
        self.assertFalse(self.recorder.is_alive())
        self.assertIsNone(self.recorder.error)
        self.assertTrue(self.recorder.zmq_socket.closed)
        self.assertFalse(self.context.closed)
        self.assertFalse(self.publisher.closed)

    def test_stop_during_write_drains_backlog_and_final_partial_batch(self):
        writing = threading.Event()
        resume = threading.Event()
        batch_sizes = []
        writer_threads = []
        write_batch = self.recorder.write_batch

        def slow_write(frames, timestamps):
            batch_sizes.append(len(frames))
            writer_threads.append(threading.get_ident())
            writing.set()
            if not resume.wait(3):
                raise TimeoutError("Test did not release the writer")
            write_batch(frames, timestamps)

        with patch.object(self.recorder, "write_batch", side_effect=slow_write):
            try:
                self.recorder.start()
                self.assertTrue(self.recorder.ready.wait(2))
                for frame_id in range(2):
                    self.publisher.send_json({"timestamps": {"frame_id": frame_id}}, zmq.SNDMORE)
                    self.publisher.send(bytes([frame_id]))
                self.assertTrue(writing.wait(2))
                for frame_id in range(2, 7):
                    self.publisher.send_json({"timestamps": {"frame_id": frame_id}}, zmq.SNDMORE)
                    self.publisher.send(bytes([frame_id]))
                # No producer sends after stop; five frames are still queued.
                self.recorder.stop()
            finally:
                resume.set()
                self.recorder.join(timeout=3)
        self.assertFalse(self.recorder.is_alive())
        self.assertIsNone(self.recorder.error)
        self.assertEqual(batch_sizes, [2, 2, 2, 1])
        self.assertEqual(writer_threads, [self.recorder.ident] * 4)
        self.assertEqual(self.recorder.frames_written, 7)
        self.assertEqual(self.video_path.read_bytes(), bytes(range(7)))
        self.assertEqual(self.video_path.with_suffix(".txt").read_text(),
                         "frame_id\t\n" + "".join(f"{i}\t\n" for i in range(7)))
        self.assertTrue(self.recorder._video_file.closed)
        self.assertTrue(self.recorder._tstamp_file.closed)
        self.assertTrue(self.recorder.zmq_socket.closed)

    def test_open_failure_is_reported_and_closes_partially_opened_files(self):
        # Opening the timestamp file fails after the pixel file has opened.
        self.recorder.filenames["timestamps"] = str(Path(self.directory.name) / "missing" / "video.txt")
        with self.assertRaises(FileNotFoundError):
            self.recorder.run()
        self.assertIsInstance(self.recorder.error, FileNotFoundError)
        self.assertTrue(self.recorder.ready.is_set())
        self.assertTrue(self.recorder._video_file.closed)
        self.assertTrue(self.recorder.zmq_socket.closed)

    def test_sync_failure_is_reported_and_both_files_still_close(self):
        error = OSError(errno.EIO, "Simulated sync failure")
        self.recorder.stop()
        with patch("cammy.record.video.os.fsync", side_effect=error) as fsync:
            with self.assertRaisesRegex(OSError, "Simulated sync failure"):
                self.recorder.run()
            self.assertEqual(fsync.call_count, 2)
            self.recorder.close_writer()  # Closing again must be harmless.
            self.assertEqual(fsync.call_count, 2)
        self.assertIs(self.recorder.error, error)
        self.assertTrue(self.recorder.ready.is_set())
        self.assertTrue(self.recorder._video_file.closed)
        self.assertTrue(self.recorder._tstamp_file.closed)
        self.assertTrue(self.recorder.zmq_socket.closed)


if __name__ == "__main__":
    unittest.main()
