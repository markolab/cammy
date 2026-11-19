import unittest
from unittest.mock import patch

from click.testing import CliRunner
import zmq

from cammy.cli import cli
from cammy.record.video import RawVideoRecorder


class RecordQueueTests(unittest.TestCase):
    def test_cli_default_and_custom_queue_size(self):
        for args, expected in [([], 64), (["--record-queue-frames", "256"], 256)]:
            with self.subTest(args=args), cli.commands["run"].make_context("run", args) as context:
                self.assertEqual(context.params["record_queue_frames"], expected)

    def test_cli_rejects_unbounded_negative_and_noninteger_sizes(self):
        for value in ["0", "-1", "1.5"]:
            with self.subTest(value=value):
                result = CliRunner().invoke(cli, ["run", "--record-queue-frames", value])
                self.assertEqual(result.exit_code, 2)
                self.assertIn("--record-queue-frames", result.output)

    def test_custom_queue_bounds_backlog_and_recovers_after_draining(self):
        context = zmq.Context()
        self.addCleanup(context.term)
        publisher = context.socket(zmq.PUSH)
        self.addCleanup(publisher.close, linger=0)
        publisher.setsockopt(zmq.SNDHWM, 3)
        publisher.bind("inproc://custom-record-queue")
        recorder = RawVideoRecorder(queue_frames=3, zmq_address="inproc://custom-record-queue")

        def check_queue_before_recording():
            receiver = recorder.zmq_socket
            receiver.setsockopt(zmq.RCVTIMEO, 2000)
            self.assertTrue(publisher.poll(2000, zmq.POLLOUT))

            # inproc combines the send and receive limits: three frames on each side.
            for frame_id in range(6):
                publisher.send_json({"timestamps": {"frame_id": frame_id}}, zmq.SNDMORE | zmq.NOBLOCK)
                publisher.send(bytes([frame_id]), flags=zmq.NOBLOCK)
            with self.assertRaises(zmq.Again):
                publisher.send_json({"timestamps": {"frame_id": 6}}, zmq.SNDMORE | zmq.NOBLOCK)

            for frame_id in range(6):
                self.assertEqual(receiver.recv_json()["timestamps"]["frame_id"], frame_id)
                self.assertEqual(receiver.recv(), bytes([frame_id]))
            self.assertTrue(publisher.poll(2000, zmq.POLLOUT))
            publisher.send_json({"timestamps": {"frame_id": 6}}, zmq.SNDMORE | zmq.NOBLOCK)
            publisher.send(b"\x06", flags=zmq.NOBLOCK)
            self.assertEqual(receiver.recv_json()["timestamps"]["frame_id"], 6)
            self.assertEqual(receiver.recv(), b"\x06")
            recorder.stop()

        with patch("cammy.record.base.zmq.Context.instance", return_value=context), \
                patch("cammy.record.base.os.nice"), \
                patch.object(recorder, "open_writer", side_effect=check_queue_before_recording), \
                patch.object(recorder, "close_writer"):
            recorder.run()
        self.assertTrue(recorder.zmq_socket.closed)


if __name__ == "__main__":
    unittest.main()
