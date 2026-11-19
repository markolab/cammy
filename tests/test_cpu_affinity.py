import io
import unittest
from unittest.mock import patch

from click.testing import CliRunner

from cammy.cli import cli
from cammy.util import get_cpu_affinity, get_physical_core_map


class CpuAffinityTests(unittest.TestCase):
    def test_topology_keeps_siblings_together_and_respects_allowed_cpus(self):
        # Sibling lists also distinguish cores with the same ID on different sockets.
        siblings = {0: "0,4\n", 1: "1,5\n", 2: "2,6\n", 4: "0,4\n", 6: "2,6\n"}

        def read_topology(path):
            cpu = int(path.split("/cpu")[-1].split("/")[0])
            return io.StringIO(siblings[cpu])

        with patch("cammy.util.os.sched_getaffinity", return_value=set(siblings), create=True), \
                patch("builtins.open", side_effect=read_topology):
            cores = get_physical_core_map()

        self.assertEqual(dict(cores), {"0,4": [0, 4], "1,5": [1], "2,6": [2, 6]})

    def allocate(self, cores, record=True, writer_cores=None):
        with patch("platform.system", return_value="Linux"), \
                patch("cammy.util.get_physical_core_map", return_value=cores):
            return get_cpu_affinity(record, writer_cores=writer_cores)

    def assert_separate_physical_cores(self, cores, pools):
        for siblings in cores.values():
            occupied_pools = sum(bool(set(siblings) & set(pool)) for pool in pools)
            self.assertEqual(occupied_pools, 1)
        assigned = [cpu for pool in pools for cpu in pool]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), {cpu for siblings in cores.values() for cpu in siblings})

    def test_recording_separates_gui_acquisition_and_writers_with_smt(self):
        cores = {core: [core, core + 10] for core in range(10)}
        gui, acquire, write = self.allocate(cores)

        self.assertEqual(gui, {0, 10})
        self.assertEqual(acquire, [1, 2, 3, 4, 5, 11, 12, 13, 14, 15])
        self.assertEqual(write, [6, 7, 8, 9, 16, 17, 18, 19])
        self.assert_separate_physical_cores(cores, [gui, acquire, write])

    def test_recording_without_smt_or_consecutive_cpu_ids(self):
        cores = {"a": [2], "b": [5], "c": [8], "d": [11]}
        gui, acquire, write = self.allocate(cores)

        self.assertEqual((gui, acquire, write), ({2}, [5, 8], [11]))
        self.assert_separate_physical_cores(cores, [gui, acquire, write])

    def test_two_writer_cores_leave_seven_acquisition_cores_with_smt(self):
        cores = {core: [core, core + 10] for core in range(10)}
        gui, acquire, write = self.allocate(cores, writer_cores=2)

        self.assertEqual(gui, {0, 10})
        self.assertEqual(acquire, [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17])
        self.assertEqual(write, [8, 9, 18, 19])
        self.assert_separate_physical_cores(cores, [gui, acquire, write])

    def test_custom_writer_count_without_smt_or_consecutive_cpu_ids(self):
        cores = {"a": [2], "b": [5], "c": [8], "d": [11], "e": [14]}
        gui, acquire, write = self.allocate(cores, writer_cores=1)

        self.assertEqual((gui, acquire, write), ({2}, [5, 8, 11], [14]))
        self.assert_separate_physical_cores(cores, [gui, acquire, write])

    def test_writer_count_cannot_empty_either_worker_pool(self):
        cores = {core: [core] for core in range(4)}
        for count in [0, -1, 3, 4]:
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "between 1 and 2"):
                self.allocate(cores, writer_cores=count)

    def test_all_hardware_threads_stay_in_their_pool_with_four_way_smt(self):
        cores = {core: [core, core + 3, core + 6, core + 9] for core in range(3)}
        gui, acquire, write = self.allocate(cores)

        self.assertEqual((gui, acquire, write), ({0, 3, 6, 9}, [1, 4, 7, 10], [2, 5, 8, 11]))
        self.assert_separate_physical_cores(cores, [gui, acquire, write])

    def test_preview_uses_remaining_cores_without_reserving_writers(self):
        cores = {core: [core, core + 3] for core in range(3)}
        gui, acquire, write = self.allocate(cores, record=False)

        self.assertEqual((gui, acquire, write), ({0, 3}, [1, 2, 4, 5], []))
        self.assert_separate_physical_cores(cores, [gui, acquire])

    def test_two_physical_cores_support_preview_but_cannot_isolate_recording(self):
        cores = {0: [0, 2], 1: [1, 3]}
        self.assertEqual(self.allocate(cores, record=False), ({0, 2}, [1, 3], []))
        with self.assertRaisesRegex(ValueError, "at least 3.*found 2"):
            self.allocate(cores)

    def test_non_linux_leaves_affinity_unset(self):
        with patch("platform.system", return_value="Darwin"), \
                patch("cammy.util.get_physical_core_map") as topology:
            self.assertEqual(get_cpu_affinity(record=True), (None, [None], [None]))
        topology.assert_not_called()

    def test_non_linux_logs_that_custom_writer_count_is_ignored(self):
        with patch("platform.system", return_value="Darwin"), \
                patch("cammy.util.get_physical_core_map") as topology, \
                self.assertLogs("cammy.util", level="WARNING") as logs:
            self.assertEqual(get_cpu_affinity(record=True, writer_cores=2), (None, [None], [None]))
        self.assertIn("--writer-cores only applies on Linux", logs.output[0])
        topology.assert_not_called()


class WriterCoresCliTests(unittest.TestCase):
    def test_default_and_custom_writer_counts(self):
        for args, expected in [([], None), (["--record", "--writer-cores", "2"], 2)]:
            with self.subTest(args=args), cli.commands["run"].make_context("run", args) as context:
                self.assertEqual(context.params["writer_cores"], expected)

    def test_nonpositive_and_noninteger_counts_are_rejected(self):
        for value in ["0", "-1", "1.5"]:
            with self.subTest(value=value):
                result = CliRunner().invoke(cli, ["run", "--record", "--writer-cores", value])
                self.assertEqual(result.exit_code, 2)
                self.assertIn("--writer-cores", result.output)

    def test_writer_cores_requires_recording(self):
        result = CliRunner().invoke(cli, ["run", "--writer-cores", "2"])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("--writer-cores requires --record", result.output)


if __name__ == "__main__":
    unittest.main()
