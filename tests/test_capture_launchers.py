from pathlib import Path
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

from cammy.cli import simple_preview
from cammy.trigger.nucleo import NucleoTriggerDevice


class CaptureLauncherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        source = Path(__file__).resolve().parents[1] / "launchers"
        self.scripts = self.root / "Cammy scripts"
        shutil.copytree(source, self.scripts)
        data = self.root / "camera data"
        data.mkdir()
        (data / "basler_new_8bit.toml").touch()
        self.settings = self.scripts / "settings.sh"
        with self.settings.open("a") as settings:
            settings.write(f'\nDATA_DIR="{data}"\n')
        self.cammy = self.root / "cammy"
        self.cammy.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$@"\n')
        self.cammy.chmod(0o755)

    def launch(self, mode, action):
        return subprocess.run(
            ["bash", self.scripts / "capture.sh", mode, action, self.cammy],
            cwd=self.root, text=True, capture_output=True,
        )

    def install(self, desktop, **environment):
        env = {"PATH": f"{self.root}:{os.defpath}", "HOME": str(self.root), **environment}
        subprocess.run(
            [sys.executable, self.scripts / "install.py", desktop], check=True,
            env=env, capture_output=True,
        )

    def run_shortcut(self, shortcut):
        fields = dict(line.split("=", 1) for line in shortcut.read_text().splitlines() if "=" in line)
        return subprocess.run(shlex.split(fields["Exec"]), cwd=fields["Path"], text=True, capture_output=True,
                              env={"PATH": os.defpath, "HOME": str(self.root)})

    def assert_printed_command_matches(self, result):
        command, = [line for line in result.stderr.splitlines() if line.startswith("cd -- ")]
        replay = subprocess.run(["bash", "-c", command], cwd=self.root, text=True, capture_output=True, check=True)
        self.assertEqual(replay.stdout, result.stdout)

    def test_desktop_launch_recreates_the_installing_shells_aravis_environment(self):
        desktop = self.root / "Desktop"
        prefix = str(self.root / "conda envs" / "aravis")
        conda = self.root / "conda executable"
        conda.write_text('#!/bin/sh\n'
                         '[ "$1 $2 $3" = "run --no-capture-output --prefix" ] || exit 1\n'
                         'export CONDA_PREFIX="$4"\nshift 4\nexec "$@"\n')
        conda.chmod(0o755)
        typelibs = "/opt/camera SDK/girepository-1.0:/usr/local/lib/girepository-1.0"
        libraries = "/opt/camera SDK/lib"
        self.cammy.write_text('#!/bin/sh\nprintf "%s\\n" "$CONDA_PREFIX" "$GI_TYPELIB_PATH" "$LD_LIBRARY_PATH"\n')
        self.install(desktop, CONDA_PREFIX=prefix, CONDA_EXE=str(conda),
                     GI_TYPELIB_PATH=typelibs, LD_LIBRARY_PATH=libraries)
        for name in ("cammy-surface-preview.desktop", "cammy-tune.desktop"):
            with self.subTest(name=name):
                result = self.run_shortcut(desktop / name)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), [prefix, typelibs, libraries])

    def test_compression_shortcut_recovers_from_failure_and_skips_finished_avis(self):
        desktop = self.root / "test desktop"
        conda = self.root / "conda"
        conda.write_text('#!/bin/sh\n'
                         'if [ "$1 $2 $3 $4" = "env update --name markolab-cli" ]; then test -f "$6"; exit; fi\n'
                         '[ "$1 $2 $3 $4" = "run --no-capture-output --name markolab-cli" ] || exit 1\n'
                         'shift 4\nexport PATH="$(dirname "$0"):$PATH"\nexec "$@"\n')
        conda.chmod(0o755)
        (self.root / "python").symlink_to(sys.executable)
        converter = self.root / "markolab-cli"
        converter.write_text('#!/bin/sh\n[ "$1" = convert-dat-to-avi ] || exit 1\n'
                             '[ "$2" = --help ] && exit 0\n'
                             'echo "$2" >> "$HOME/calls"\nprintf avi > "$4"\n'
                             '[ -s "$2" ]\n')
        converter.chmod(0o755)
        subprocess.run([sys.executable, self.scripts / "install.py", "--compression", desktop],
                       check=True, capture_output=True, env={"PATH": os.defpath, "CONDA_EXE": str(conda)})
        data = self.root / "camera data"
        sessions = [data.resolve() / name for name in ("session_01 (rig)", "session_02", "other")]
        for session in sessions:
            session.mkdir()
            (session / "cam A.dat").write_bytes(b"raw")
        bad = sessions[0] / "cam B.dat"
        bad.touch()
        shortcut = desktop / "cammy-compress.desktop"
        result = self.run_shortcut(shortcut)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("1 failed", result.stdout)
        self.assertFalse(bad.with_suffix(".avi").exists())
        self.assertEqual((sessions[0] / "cam A.avi").read_bytes(), b"avi")
        self.assertEqual((sessions[1] / "cam A.dat").read_bytes(), b"raw")
        self.assertFalse((sessions[2] / "cam A.avi").exists())
        bad.write_bytes(b"fixed")
        result = self.run_shortcut(shortcut)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 converted, 2 skipped, 0 failed", result.stdout)
        self.assertEqual([Path(line) for line in (self.root / "calls").read_text().splitlines()],
                         [sessions[0] / "cam A.dat", bad, sessions[1] / "cam A.dat", bad])

    def test_six_shortcuts_use_expected_trigger_sequence_and_recording_options(self):
        desktop = self.root / "test desktop"
        self.install(desktop)
        self.assertEqual(len(list(desktop.glob("*.desktop"))), 7)
        settings = desktop / "cammy-settings.sh"
        self.assertEqual(settings.read_bytes(), self.settings.read_bytes())
        with settings.open("a") as f:
            f.write("STREAM_STATS=off\nGAIN_DB=2.5\nROI_WIDTH=1024\nROI_HEIGHT=768\nROI_HORIZONTAL=right\nROI_VERTICAL=bottom\n"
                    "PREVIEW_TIME_DOWNSAMPLE=4\nSURFACE_TO_FLUORESCENCE_GAP_US=1250\nLIGHT_SETTLE_US=200\n"
                    "WRITER_CORES=5\nACQUISITION_CORES=3\nSHARED_GUI_CORES=2\n"
                    "ARAVIS_BUFFER_SIZE=192\nRECORD_QUEUE_FRAMES=96\nWRITER_BUFFER_MIB=512\nFRAME_WRITER_BATCH_SIZE=16\n"
                    "SURFACE_POLARITY=high\nSURFACE_VOLTAGE=3.5\nFLUORESCENCE_VOLTAGE=1.25\n"
                    "SURFACE_ILLUMINATION_PERCENT=0\nFLUORESCENCE_ILLUMINATION_PERCENT=62.5\n")
        self.install(desktop)
        expected_steps = {"surface": "1,1000,9750", "fluorescence": "1,10000,750",
                          "alternate": "2,1000,1250,10000,9250"}
        for mode, steps in expected_steps.items():
            for action in ("preview", "record"):
                with self.subTest(mode=mode, action=action):
                    shortcut = desktop / f"cammy-{mode}-{action}.desktop"
                    self.assertTrue(os.access(shortcut, os.X_OK))
                    result = self.run_shortcut(shortcut)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assert_printed_command_matches(result)
                    data, command, *args = result.stdout.splitlines()
                    self.assertEqual(Path(data).resolve(), (self.root / "camera data").resolve())
                    self.assertEqual(command, "run")
                    p = simple_preview.make_context("run", args).params
                    self.assertEqual(p["light_mode"], mode)
                    self.assertEqual(p["record"], action == "record")
                    self.assertEqual(p["buffer_size"], 192)
                    if action == "record":
                        self.assertEqual(p["record_queue_frames"], 96)
                        self.assertEqual((p["writer_buffer_mib"], p["frame_writer_batch_size"]), (512, 16))
                    else:
                        self.assertNotIn("--record-queue-frames", args)
                        self.assertNotIn("--writer-buffer-mib", args)
                    self.assertFalse(p["stream_stats"])
                    self.assertEqual(p["display_time_downsample"], 0 if action == "record" else 4)
                    self.assertEqual(p["shared_gui_cores"], 2)
                    self.assertEqual(p["acquisition_cores"], 3)
                    self.assertEqual(p["writer_cores"], 5 if action == "record" else None)
                    self.assertEqual(p["gain_db"], 2.5)
                    self.assertEqual(p["surface_polarity"], "high")
                    self.assertEqual((p["roi_width"], p["roi_height"]), (1024, 768))
                    self.assertEqual((p["roi_horizontal"], p["roi_vertical"]), ("right", "bottom"))
                    dev = NucleoTriggerDevice(
                        mode=mode, pulse_widths=p["hw_trigger_pulse_width"],
                        pulse_widths_low=p["hw_trigger_pulse_low"], light_settle_us=p["light_settle_us"],
                        surface_voltage=p["surface_voltage"], fluorescence_voltage=p["fluorescence_voltage"],
                        surface_polarity=p["surface_polarity"],
                        surface_illumination_percent=p["surface_illumination_percent"],
                        fluorescence_illumination_percent=p["fluorescence_illumination_percent"],
                    )
                    self.assertEqual(dev.config_command, f"CONFIG,2,{dev.MODES[mode]},200,3500,1250,0,6250,{steps},1")

    def test_tune_shortcut_uses_installed_settings_and_direct_launch_uses_cli_default(self):
        desktop = self.root / "custom desktop"
        self.install(desktop)
        shortcut = desktop / "cammy-tune.desktop"
        self.assertTrue(os.access(shortcut, os.X_OK))
        result = self.run_shortcut(shortcut)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[1:], ["tune", "--settings", str(desktop.resolve() / "cammy-settings.sh")])
        self.assert_printed_command_matches(result)
        direct = subprocess.run(["bash", self.scripts / "capture.sh", "tune"], text=True, capture_output=True,
                                env={"PATH": f"{self.root}:{os.defpath}"}, cwd=self.root)
        self.assertEqual(direct.returncode, 0, direct.stderr)
        self.assertEqual(direct.stdout.splitlines()[1:], ["tune"])

    def test_old_desktop_settings_use_the_current_rig_defaults(self):
        desktop = self.root / "Desktop"
        desktop.mkdir()
        settings = desktop / "cammy-settings.sh"
        original = f'DATA_DIR="{self.root / "camera data"}"\n'
        settings.write_text(original)
        self.install(desktop)
        upgraded = settings.read_text()
        self.assertTrue(upgraded.startswith(original))
        self.assertIn("# SURFACE_VOLTAGE=", upgraded)
        self.assertIn("# STREAM_STATS=on", upgraded)
        self.assertIn("# STREAM_STATS_INTERVAL=1", upgraded)
        self.assertIn("# ARAVIS_BUFFER_SIZE=512", upgraded)
        self.assertIn("# RECORD_QUEUE_FRAMES=512", upgraded)
        self.install(desktop)
        self.assertEqual(settings.read_text(), upgraded)
        result = self.run_shortcut(desktop / "cammy-alternate-preview.desktop")
        self.assertEqual(result.returncode, 0, result.stderr)
        direct = self.launch("alternate", "preview")
        self.assertEqual(direct.returncode, 0, direct.stderr)
        self.assertEqual(result.stdout, direct.stdout)
        params = simple_preview.make_context("run", direct.stdout.splitlines()[2:]).params
        self.assertEqual(params["display_time_downsample"], 1)
        self.assertTrue(params["stream_stats"])
        self.assertEqual(params["shared_gui_cores"], 0)
        with settings.open("a") as f:
            f.write("STREAM_STATS_INTERVAL=5\n")
        result = self.run_shortcut(desktop / "cammy-alternate-record.desktop")
        self.assertEqual(result.returncode, 0, result.stderr)
        params = simple_preview.make_context("run", result.stdout.splitlines()[2:]).params
        self.assertEqual(params["stream_stats_interval"], 5)
        self.assertEqual((params["shared_gui_cores"], params["writer_cores"], params["acquisition_cores"]), (1, 7, None))
        self.assertEqual((params["buffer_size"], params["record_queue_frames"]), (512, 512))

    def test_tight_alternation_reserves_time_for_both_lights_to_settle(self):
        with self.settings.open("a") as settings:
            settings.write("FPS=100\nFLUORESCENCE_EXPOSURE_US=18500\nLIGHT_SETTLE_US=200\n")
        result = self.launch("alternate", "record")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("300 µs", result.stderr)
        p = simple_preview.make_context("run", result.stdout.splitlines()[2:]).params
        self.assertEqual(p["hw_trigger_pulse_low"], (.0003, .0002))
        with self.settings.open("a") as settings:
            settings.write("SURFACE_TO_FLUORESCENCE_GAP_US=100\n")
        rejected = self.launch("alternate", "preview")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("SURFACE_TO_FLUORESCENCE_GAP_US", rejected.stderr)

    def test_100_fps_requires_settling_and_uses_only_active_exposure(self):
        with self.settings.open("a") as settings:
            settings.write("FPS=100\nGAIN_DB=0\n")
        rejected = self.launch("fluorescence", "preview")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("shorten the exposure", rejected.stderr)
        self.assertEqual(rejected.stdout, "")
        with self.settings.open("a") as settings:
            settings.write("FLUORESCENCE_EXPOSURE_US=9900\n")
        result = self.launch("fluorescence", "preview")
        self.assertEqual(result.returncode, 0, result.stderr)
        p = simple_preview.make_context("run", result.stdout.splitlines()[2:]).params
        self.assertEqual(p["gain_db"], 0)
        self.assertEqual(p["hw_trigger_pulse_low"], (0.0001,))
