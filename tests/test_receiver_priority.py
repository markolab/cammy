from contextlib import ExitStack
import errno
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

from click.testing import CliRunner

from cammy.camera.aravis import set_receiver_realtime_priority
from cammy.cli import cli


class ReceiverPriorityTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tasks = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.task_path = self.stack.enter_context(
            patch("cammy.camera.aravis.Path", return_value=self.tasks)
        )
        self.stack.enter_context(patch("cammy.camera.aravis.sys.platform", "linux"))
        self.stack.enter_context(patch("cammy.camera.aravis.os.SCHED_RR", 2, create=True))
        self.scheduler = self.stack.enter_context(
            patch("cammy.camera.aravis.os.sched_setscheduler", create=True)
        )
        self.params = self.stack.enter_context(
            patch("cammy.camera.aravis.os.sched_param", create=True)
        )

    def add_thread(self, tid, name):
        task = self.tasks / str(tid)
        task.mkdir()
        (task / "comm").write_text(name + "\n")

    def test_disabled_does_not_scan_or_change_threads(self):
        set_receiver_realtime_priority(0)
        self.task_path.assert_not_called()
        self.scheduler.assert_not_called()

    def test_only_native_gige_receivers_are_promoted(self):
        for tid, name in [(101, "cammy"), (102, "arv_gv_stream"),
                          (103, "ZMQbg/IO/0"), (104, "arv_uv_stream"),
                          (105, "arv_gv_stream")]:
            self.add_thread(tid, name)
        (self.tasks / "106").mkdir()  # Simulate a thread exiting before comm is read.
        with self.assertLogs("cammy.camera.aravis", level="INFO") as logs:
            set_receiver_realtime_priority(10)
        self.task_path.assert_called_once_with("/proc/self/task")
        self.assertEqual(self.scheduler.call_args_list, [
            call(102, 2, self.params.return_value),
            call(105, 2, self.params.return_value),
        ])
        self.assertEqual(self.params.call_args_list, [call(10), call(10)])
        self.assertIn("TID 102: SCHED_RR priority 10", logs.output[0])
        self.assertIn("TID 105: SCHED_RR priority 10", logs.output[1])

    def test_permission_failure_is_logged_and_other_receivers_are_attempted(self):
        self.add_thread(102, "arv_gv_stream")
        self.add_thread(105, "arv_gv_stream")
        self.scheduler.side_effect = [PermissionError(errno.EPERM, "Operation not permitted"), None]
        with self.assertLogs("cammy.camera.aravis", level="INFO") as logs:
            set_receiver_realtime_priority(10)
        self.assertEqual(self.scheduler.call_count, 2)
        self.assertIn("Could not set Aravis receiver TID 102", logs.output[0])
        self.assertIn("Operation not permitted", logs.output[0])
        self.assertIn("keeping existing scheduling", logs.output[0])
        self.assertIn("TID 105: SCHED_RR priority 10", logs.output[1])

    def test_unsupported_platform_is_logged_without_scanning(self):
        with patch("cammy.camera.aravis.sys.platform", "darwin"), \
                self.assertLogs("cammy.camera.aravis", level="WARNING") as logs:
            set_receiver_realtime_priority(10)
        self.assertIn("requires Linux", logs.output[0])
        self.task_path.assert_not_called()
        self.scheduler.assert_not_called()

    def test_missing_proc_directory_is_logged(self):
        self.task_path.return_value = self.tasks / "missing"
        with self.assertLogs("cammy.camera.aravis", level="WARNING") as logs:
            set_receiver_realtime_priority(10)
        self.assertIn("Could not find Aravis receiver threads", logs.output[0])
        self.scheduler.assert_not_called()

    def test_no_receivers_is_logged(self):
        self.add_thread(101, "cammy")
        with self.assertLogs("cammy.camera.aravis", level="WARNING") as logs:
            set_receiver_realtime_priority(10)
        self.assertIn("No arv_gv_stream receiver threads found", logs.output[0])
        self.scheduler.assert_not_called()


class ReceiverPriorityCliTests(unittest.TestCase):
    def test_default_and_valid_priorities(self):
        for args, expected in [([], 0), (["--receiver-realtime-priority", "10"], 10),
                               (["--receiver-realtime-priority", "99"], 99)]:
            with self.subTest(args=args), cli.commands["run"].make_context("run", args) as context:
                self.assertEqual(context.params["receiver_realtime_priority"], expected)

    def test_invalid_priorities_are_rejected(self):
        for value in ["-1", "100", "1.5"]:
            with self.subTest(value=value):
                result = CliRunner().invoke(cli, ["run", "--receiver-realtime-priority", value])
                self.assertEqual(result.exit_code, 2)
                self.assertIn("--receiver-realtime-priority", result.output)


if __name__ == "__main__":
    unittest.main()
