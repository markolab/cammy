import errno
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import zmq

from cammy.record.buffered import BufferedRawVideoRecorder


class BufferedRecordTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "cam.dat"
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUSH)
        self.publisher.sndhwm = 1
        self.publisher.sndtimeo = 2000
        self.publisher.bind("inproc://buffered")
        self.recorder = BufferedRawVideoRecorder(
            filename=str(self.path), timestamp_fields=["frame_id", "device_timestamp"],
            zmq_address="inproc://buffered", queue_frames=1, frame_bytes=4,
            buffer_bytes=20, batch_size=2,
        )
        self.recorder.daemon = True
        instance = patch("cammy.record.base.zmq.Context.instance", return_value=self.context)
        instance.start()
        self.addCleanup(instance.stop)
        self.resume = threading.Event()

    def tearDown(self):
        self.resume.set()
        self.recorder.stop()
        if self.recorder.ident is not None:
            self.recorder.join(3)
        self.publisher.close(linger=0)
        self.assertFalse(self.recorder.is_alive(), "Recorder did not stop")
        self.context.term()

    def send(self, index, flags=0):
        timestamps = {"frame_id": index, "device_timestamp": index * 10}
        self.publisher.send_json({"timestamps": timestamps}, flags | zmq.SNDMORE)
        self.publisher.send(index.to_bytes(4, "little"), flags)

    def fill_buffer(self):
        self.recorder.start()
        self.assertTrue(self.recorder.ready.wait(2))
        # Five pool slots and two inproc queue slots. The first write is stalled.
        for index in range(7):
            self.send(index)
        deadline = time.monotonic() + 2
        while "writer_full_waits=0 " in self.recorder.buffer_statistics() and time.monotonic() < deadline:
            time.sleep(0.005)
        stats = self.recorder.buffer_statistics()
        self.assertIn("writer_buffer_frames=5/5 writer_buffer_peak=5", stats)
        self.assertIn("writer_full_waits=1 ", stats)
        self.assertEqual(self.recorder.frames_written, 0)
        with self.assertRaises(zmq.Again):
            self.send(7, zmq.NOBLOCK)

    def test_stop_with_full_buffer_drains_in_order_including_partial_batch(self):
        batches = []
        disk_threads = set()
        write_batch = self.recorder.write_batch

        def stalled_write(frames, timestamps):
            batches.append(len(frames))
            disk_threads.add(threading.get_ident())
            if not self.resume.wait(5):
                raise TimeoutError("Test writer was never released")
            write_batch(frames, timestamps)

        with patch.object(self.recorder, "write_batch", side_effect=stalled_write):
            self.fill_buffer()
            self.recorder.stop()
            self.resume.set()
            self.recorder.join(3)
        self.assertFalse(self.recorder.is_alive())
        self.assertIsNone(self.recorder.error)
        self.assertEqual(len(disk_threads), 1)
        self.assertNotIn(self.recorder.ident, disk_threads)
        self.assertEqual(batches, [2, 2, 2, 1])
        self.assertEqual(self.recorder.frames_written, 7)
        self.assertEqual(self.path.read_bytes(), b"".join(i.to_bytes(4, "little") for i in range(7)))
        self.assertEqual(self.path.with_suffix(".txt").read_text(),
                         "frame_id\tdevice_timestamp\t\n" + "".join(f"{i}\t{i * 10}\t\n" for i in range(7)))
        self.assertIn("writer_buffer_frames=0/5", self.recorder.buffer_statistics())
        self.assertGreater(self.recorder._max_write_ms, 0)
        self.assertTrue(self.recorder._video_file.closed)
        self.assertTrue(self.recorder._tstamp_file.closed)
        self.assertTrue(self.recorder.zmq_socket.closed)

    def test_disk_error_wakes_full_receiver_without_replaying_short_write(self):
        calls = []
        error = OSError(errno.ENOSPC, "disk full")

        def failing_write(fd, buffers):
            calls.append(len(buffers))
            if len(calls) == 1:
                if not self.resume.wait(5):
                    raise TimeoutError("Test writer was never released")
                return os.write(fd, buffers[0][:3])
            raise error

        with patch("cammy.record.video.os.writev", side_effect=failing_write), patch("threading.excepthook") as hook:
            self.fill_buffer()
            self.resume.set()
            self.recorder.join(3)
        self.assertFalse(self.recorder.is_alive())
        self.assertIs(self.recorder.error, error)
        self.assertIs(hook.call_args.args[0].exc_value, error)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.recorder.frames_written, 0)
        self.assertEqual(self.path.read_bytes(), bytes(3))
        self.assertEqual(self.path.with_suffix(".txt").read_text(), "frame_id\tdevice_timestamp\t\n")

    def test_stop_before_first_frame_syncs_both_empty_outputs(self):
        self.recorder.stop()
        with patch("cammy.record.video.os.fsync", wraps=os.fsync) as sync:
            self.recorder.run()
        self.assertTrue(self.recorder.ready.is_set())
        self.assertEqual(sync.call_count, 2)
        self.assertEqual(self.recorder.frames_written, 0)
        self.assertTrue(self.recorder.zmq_socket.closed)

    def test_batch_must_fit_in_pool(self):
        with self.assertRaisesRegex(ValueError, "3-frame recording buffer"):
            BufferedRawVideoRecorder(frame_bytes=4, buffer_bytes=15, batch_size=4)
