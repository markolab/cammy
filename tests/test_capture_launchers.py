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

    def install(self, desktop):
        subprocess.run(
            [sys.executable, self.scripts / "install.py", desktop], check=True,
            env={**os.environ, "PATH": f"{self.root}:{os.environ['PATH']}"}, capture_output=True,
        )

    def run_shortcut(self, shortcut):
        fields = dict(line.split("=", 1) for line in shortcut.read_text().splitlines() if "=" in line)
        return subprocess.run(shlex.split(fields["Exec"]), cwd=fields["Path"], text=True, capture_output=True)

    def test_six_shortcuts_use_expected_trigger_sequence_and_recording_options(self):
        desktop = self.root / "test desktop"
        self.install(desktop)
        self.assertEqual(len(list(desktop.glob("*.desktop"))), 6)
        settings = desktop / "cammy-settings.sh"
        self.assertEqual(settings.read_bytes(), self.settings.read_bytes())
        with settings.open("a") as f:
            f.write("GAIN_DB=2.5\nSURFACE_TO_FLUORESCENCE_GAP_US=1250\nLIGHT_SETTLE_US=200\n"
                    "SURFACE_VOLTAGE=3.5\nFLUORESCENCE_VOLTAGE=1.25\n"
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
                    data, command, *args = result.stdout.splitlines()
                    self.assertEqual(Path(data).resolve(), (self.root / "camera data").resolve())
                    self.assertEqual(command, "run")
                    p = simple_preview.make_context("run", args).params
                    self.assertEqual(p["light_mode"], mode)
                    self.assertEqual(p["record"], action == "record")
                    self.assertEqual(p["display_time_downsample"], 0 if action == "record" else 1)
                    self.assertEqual(p["share_gui_core"], action == "record")
                    self.assertEqual(p["writer_cores"], 7 if action == "record" else None)
                    self.assertEqual(p["gain_db"], 2.5)
                    dev = NucleoTriggerDevice(
                        mode=mode, pulse_widths=p["hw_trigger_pulse_width"],
                        pulse_widths_low=p["hw_trigger_pulse_low"], light_settle_us=p["light_settle_us"],
                        surface_voltage=p["surface_voltage"], fluorescence_voltage=p["fluorescence_voltage"],
                        surface_illumination_percent=p["surface_illumination_percent"],
                        fluorescence_illumination_percent=p["fluorescence_illumination_percent"],
                    )
                    self.assertEqual(dev.config_command, f"CONFIG,1,{dev.MODES[mode]},200,3500,1250,0,6250,{steps}")

    def test_old_desktop_settings_use_the_current_rig_defaults(self):
        desktop = self.root / "Desktop"
        desktop.mkdir()
        settings = desktop / "cammy-settings.sh"
        settings.write_text("FPS=93.0233\n")
        self.install(desktop)
        upgraded = settings.read_text()
        self.assertTrue(upgraded.startswith("FPS=93.0233\n"))
        self.assertIn("# SURFACE_VOLTAGE=5.0", upgraded)
        self.install(desktop)
        self.assertEqual(settings.read_text(), upgraded)
        result = self.run_shortcut(desktop / "cammy-alternate-preview.desktop")
        self.assertEqual(result.returncode, 0, result.stderr)
        p = simple_preview.make_context("run", result.stdout.splitlines()[2:]).params
        self.assertEqual(p["hw_trigger_pulse_low"], (.0005, .010))
        self.assertEqual(p["light_settle_us"], 100)
        self.assertEqual((p["surface_voltage"], p["fluorescence_voltage"]), (5, 2))
        self.assertEqual((p["surface_illumination_percent"], p["fluorescence_illumination_percent"]), (100, 50))

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
