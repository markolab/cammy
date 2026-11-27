import logging
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import zmq

from cammy.camera.aravis import acquisition_loop
from cammy.camera.spoof import SpoofCamera


class RecordingBackpressureTests(unittest.TestCase):
    def test_full_queue_drops_one_frame_and_resumes_without_mixing_metadata(self):
        context = zmq.Context()
        self.addCleanup(context.term)
        publisher = context.socket(zmq.PUSH)
        receiver = context.socket(zmq.PULL)
        for socket in (publisher, receiver):
            socket.setsockopt(zmq.LINGER, 0)
            self.addCleanup(socket.close)
        publisher.setsockopt(zmq.SNDHWM, 1)
        receiver.setsockopt(zmq.RCVHWM, 1)
        # Bound failures if a send regresses to blocking or a message is lost.
        publisher.setsockopt(zmq.SNDTIMEO, 2000)
        receiver.setsockopt(zmq.RCVTIMEO, 2000)
        publisher.bind("inproc://recording-backpressure")
        receiver.connect("inproc://recording-backpressure")
        self.assertTrue(publisher.poll(2000, zmq.POLLOUT))

        shutdown = threading.Event()
        frames = [np.full((2, 2), i, dtype=np.uint16) for i in range(4)]
        frame_ids = iter(range(4))
        received = []

        def receive_frame():
            return receiver.recv_json(), receiver.recv()

        def pop_frame():
            frame_id = next(frame_ids)
            if frame_id == 3:
                # Two queued frames fill the inproc pipe; frame 2 was dropped.
                received.extend([receive_frame(), receive_frame()])
                self.assertTrue(publisher.poll(2000, zmq.POLLOUT))
                shutdown.set()
            return frames[frame_id], {"frame_id": frame_id}

        camera = SimpleNamespace(
            logger=logging.getLogger(__name__), zmq_publisher=publisher,
            try_pop_frame=Mock(side_effect=pop_frame),
        )
        start = time.monotonic()
        with patch("cammy.camera.aravis.os.nice"):
            acquisition_loop(camera, shutdown)
        self.assertLess(time.monotonic() - start, 1)
        received.append(receive_frame())

        self.assertEqual(camera.recording_drops, 1)
        self.assertEqual(camera.try_pop_frame.call_count, 4)
        self.assertEqual(camera.display_frame[1], {"frame_id": 3})
        self.assertEqual([metadata["timestamps"]["frame_id"] for metadata, _ in received], [0, 1, 3])
        for (metadata, pixels), frame_id in zip(received, [0, 1, 3]):
            self.assertEqual(metadata["shape"], [2, 2])
            self.assertEqual(metadata["dtype"], "uint16")
            self.assertEqual(pixels, frames[frame_id].tobytes())
        self.assertFalse(receiver.poll(0, zmq.POLLIN))

    def test_preview_without_recording_still_updates(self):
        shutdown = threading.Event()
        frame = np.zeros((2, 2), dtype=np.uint16)

        def pop_frame():
            shutdown.set()
            return frame, {"frame_id": 0}

        camera = SimpleNamespace(
            logger=logging.getLogger(__name__), zmq_publisher=None,
            try_pop_frame=pop_frame,
        )
        with patch("cammy.camera.aravis.os.nice"):
            acquisition_loop(camera, shutdown)
        self.assertIs(camera.display_frame[0], frame)
        self.assertEqual(camera.recording_drops, 0)

    def test_spoof_camera_uses_common_acquisition_without_storage_queue(self):
        camera = SpoofCamera("spoof")
        frame = np.zeros((2, 2), dtype=np.uint8)
        timestamps = {"frame_id": 1}
        camera.recv_queue.put((frame, timestamps))
        shutdown = threading.Event()
        pop_frame = camera.try_pop_frame

        def pop_and_stop():
            shutdown.set()
            return pop_frame()

        with patch("cammy.camera.aravis.os.nice"), \
                patch.object(camera, "try_pop_frame", side_effect=pop_and_stop):
            acquisition_loop(camera, shutdown)
        self.assertIs(camera.display_frame[0], frame)
        self.assertEqual(camera.display_frame[1], timestamps)
        self.assertEqual(camera.frame_count, 1)
        self.assertTrue(camera.recv_queue.empty())
        self.assertIsNone(camera.save_queue)


if __name__ == "__main__":
    unittest.main()
