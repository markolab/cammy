"""Live single-bank tuning for the Nucleo preview."""
import fcntl
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import click

from cammy.roi import centered_region, timing_advice
from cammy.trigger.nucleo import NucleoTriggerDevice
from cammy.util import shared_asset


BANKS = ("surface", "fluorescence")
SETTINGS = ("DATA_DIR", "CAMERA_OPTIONS", "FPS", "GAIN_DB", "LIGHT_SETTLE_US", "ROI_WIDTH", "ROI_HEIGHT",
            "SURFACE_EXPOSURE_US", "FLUORESCENCE_EXPOSURE_US", "SURFACE_VOLTAGE",
            "FLUORESCENCE_VOLTAGE", "SURFACE_ILLUMINATION_PERCENT", "FLUORESCENCE_ILLUMINATION_PERCENT")


def desktop_settings():
    desktop = Path.home() / "Desktop"
    if shutil.which("xdg-user-dir"):
        result = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            desktop = Path(result.stdout.strip())
    return desktop / "cammy-settings.sh"


def read_settings(path):
    defaults = shared_asset("launchers/settings.sh")
    # Source the same trusted Bash files as capture.sh, including quoted paths and expansions.
    script = r'''
    set -eu
    source "$1" >&2
    if [[ "$1" != "$2" ]]; then source "$2" >&2; fi
    shift 2
    for key do printf '%s\0' "${!key}"; done
    '''
    proc = subprocess.run(["bash", "-c", script, "cammy tune", defaults, path, *SETTINGS],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode:
        raise ValueError(f"Could not load {path}: {os.fsdecode(proc.stderr).strip()}")
    return dict(zip(SETTINGS, os.fsdecode(proc.stdout).split("\0")))


def save_settings(path, original, changes):
    """Edit the last active assignments and preserve comments and unrelated settings."""
    text = original.decode()
    for key, value in changes.items():
        literal = value or '""'
        assignment = f"{key}={literal}"
        matches = list(re.finditer(rf"^([ \t]*(?:export[ \t]+)?){key}=(.*)$", text, re.M))
        if matches:
            last = matches[-1]
            # Preserve shell expressions or other commands on the same line.
            simple = re.fullmatch(r"""["']?[+\-\d.eE]*["']?([ \t]*(?:#.*)?)""", last[2])
            if simple:
                text = text[:last.start()] + last[1] + assignment + simple[1] + text[last.end():]
                continue
        text += ("" if text.endswith("\n") else "\n") + assignment + "\n"

    updated = text.encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
        tmp = Path(out.name)
        try:
            os.fchmod(out.fileno(), path.stat().st_mode & 0o777)
            out.write(updated)
            out.flush()
            os.fsync(out.fileno())
            with path.open("rb") as current:
                fcntl.flock(current, fcntl.LOCK_EX)
                if path.read_bytes() != original:
                    raise ValueError(f"{path.name} changed on disk. Reopen tune to load those edits.")
                os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    return updated


class TuneSession:
    PANEL_WIDTH = 920
    PANEL_HEIGHT = 300

    def __init__(self, path, mode, port):
        self.path = path.resolve()
        self.original = self.path.read_bytes()
        self.values = read_settings(self.path)
        for key in ("ROI_WIDTH", "ROI_HEIGHT"):
            if not self.values[key].isascii() or not self.values[key].isdigit():
                raise ValueError(f"{key} must be a nonnegative whole number of pixels.")
        self.mode = mode
        self.trigger = self.controller(mode, self.values, port)
        self.saved = self.values.copy()
        self.original_gains = {}
        self.gains_changed = False
        self.original_regions = {}

    @staticmethod
    def controller(mode, values, port):
        fps = float(values["FPS"])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be finite and positive.")
        width = int(values[f"{mode.upper()}_EXPOSURE_US"])
        settle = int(values["LIGHT_SETTLE_US"])
        period_us = 1_000_000 / fps
        if period_us > NucleoTriggerDevice.MAX_INTERVAL_US:
            raise ValueError("FPS is too low for the Nucleo timing range.")
        low = round(period_us) - width
        if low < max(1, settle):
            raise ValueError(f"{mode.title()} exposure leaves less than {max(1, settle)} µs between frames at {fps:g} FPS.")
        gain = values["GAIN_DB"]
        if gain and (not math.isfinite(float(gain)) or float(gain) < 0):
            raise ValueError("Gain must be blank or a finite, nonnegative number in dB.")
        return NucleoTriggerDevice(
            com=port, mode=mode, pulse_widths=(width / 1e6,), pulse_widths_low=(low / 1e6,),
            light_settle_us=settle, surface_voltage=float(values["SURFACE_VOLTAGE"]),
            fluorescence_voltage=float(values["FLUORESCENCE_VOLTAGE"]),
            surface_illumination_percent=float(values["SURFACE_ILLUMINATION_PERCENT"]),
            fluorescence_illumination_percent=float(values["FLUORESCENCE_ILLUMINATION_PERCENT"]),
        )

    def set_gain(self, cameras, gain):
        # Called with triggers stopped and cameras disarmed, including on shutdown.
        if not self.original_gains:
            for key, camera in cameras.items():
                auto, value = camera.get_feature("GainAuto"), camera.get_feature("Gain")
                if auto is None or value is None:
                    raise ValueError(f"Could not read gain on {key}")
                self.original_gains[key] = (auto, value)
        self.gains_changed = True
        for key, camera in cameras.items():
            auto, value = ("Off", float(gain)) if gain else self.original_gains[key]
            camera.set_feature("GainAuto", "Off")
            camera.set_feature("Gain", str(value))
            if auto != "Off":
                camera.set_feature("GainAuto", auto)

    def apply(self, cameras, mode, gain, exposure, voltage, *, fps=None, underfill=None):
        # Canonical numeric strings are also safe to write as shell assignments.
        requested_gain = float(gain) if gain.strip() else None
        changes = {"GAIN_DB": str(requested_gain) if requested_gain is not None else "",
                   f"{mode.upper()}_EXPOSURE_US": str(int(exposure)),
                   f"{mode.upper()}_VOLTAGE": str(float(voltage))}
        if fps is not None and float(fps) != float(self.values["FPS"]):
            changes["FPS"] = str(float(fps))
        if underfill is not None:
            if not 0 <= underfill <= 100:
                raise ValueError("Underfill must be between 0 and 100% of the exposure.")
            key = f"{mode.upper()}_ILLUMINATION_PERCENT"
            lit_percent = round(100 - underfill, 2)
            if lit_percent != float(self.values[key]):
                changes[key] = str(lit_percent)
        values = {**self.values, **changes}
        candidate = self.controller(mode, values, self.trigger.com)
        previous_gain = float(self.values["GAIN_DB"]) if self.values["GAIN_DB"] else None
        gain_changed = requested_gain != previous_gain
        if gain_changed and requested_gain is not None:
            for key, camera in cameras.items():
                try:
                    minimum, maximum = camera.camera.get_gain_bounds()
                except Exception as exc:
                    raise ValueError(f"Could not read gain limits on {key}: {exc}") from exc
                if not minimum <= requested_gain <= maximum:
                    raise ValueError(f"{key} supports {minimum:g}–{maximum:g} dB gain.")
        try:
            self.trigger.stop()
            if gain_changed:
                for camera in cameras.values():
                    camera.stop_acquisition()
                self.set_gain(cameras, values["GAIN_DB"])
                for camera in cameras.values():
                    camera.start_acquisition()
            self.trigger.reconfigure(candidate)
            self.trigger.start()
        except Exception as exc:
            # Exit through preview's ordered hardware shutdown after any partial application.
            raise click.ClickException(f"Could not apply {mode} settings. Closing preview: {exc}") from exc
        self.values = values
        self.mode = mode

    def build_panel(self, dpg, cameras, metadata=None, apply_regions=None):
        # A hint shows the startup reading while an empty input preserves per-camera gain.
        gains = {}
        for key, camera in cameras.items():
            try:
                value = float(camera.get_feature("Gain"))
                gains[key] = f"{value:g} dB" if math.isfinite(value) else "Unavailable"
            except Exception:
                gains[key] = "Unavailable"
        distinct = set(gains.values())
        gain_hint = "Mixed" if len(distinct) > 1 else next(iter(distinct), "Unavailable")

        def suggest():
            try:
                advice = timing_advice(
                    metadata, dpg.get_value("surface_us"), dpg.get_value("fluorescence_us"),
                    dpg.get_value("fluorescence_underfill"), dpg.get_value("tune_fps"),
                    int(self.values["LIGHT_SETTLE_US"]), dpg.get_value("margin_ms"))
            except ValueError as exc:
                advice = str(exc)
            dpg.set_value("timing_advice", advice)
            click.echo(advice)

        def crop():
            width, height = (dpg.get_value(tag) for tag in ("roi_width", "roi_height"))
            changes = {}
            try:
                for key, camera in cameras.items():
                    region = centered_region(camera.camera, self.original_regions[key], width, height)
                    if region != tuple(camera.camera.get_region()):
                        changes[key] = region
            except Exception as exc:
                dpg.set_value("tune_status", f"Could not crop {key}. {exc}")
                return
            if changes:
                try:
                    apply_regions(changes)
                except ValueError as exc:
                    dpg.set_value("tune_status", str(exc))
                    return
            self.values.update(ROI_WIDTH=str(width), ROI_HEIGHT=str(height))
            dpg.set_value("roi_size", current_size())
            dpg.set_value("tune_status", f"Previewing {self.mode}. Target {self.values['FPS']} FPS.")
            suggest()

        def current_size():
            sizes = sorted({f"{c._width} × {c._height}" for c in cameras.values()})
            return "Current: " + ", ".join(sizes) + " px. 0 keeps original axis."

        def apply_bank(sender, app_data, bank):
            try:
                self.apply(cameras, bank, dpg.get_value("tune_gain"), dpg.get_value(bank + "_us"),
                           dpg.get_value(bank + "_volts"), fps=dpg.get_value("tune_fps"),
                           underfill=dpg.get_value(bank + "_underfill"))
                status = f"Previewing {bank}. Target {self.values['FPS']} FPS."
            except ValueError as exc:
                status = f"Settings not applied. {exc}"
            dpg.set_value("tune_status", status)

        def save():
            changes = {k: v for k, v in self.values.items() if v != self.saved[k]}
            try:
                if changes:
                    self.original = save_settings(self.path, self.original, changes)
                    self.saved = self.values.copy()
                status = f"Applied settings saved in {self.path.name}." if changes else "No applied changes to save."
            except (ValueError, OSError) as exc:
                status = str(exc)
            dpg.set_value("tune_status", status)

        with dpg.window(label="Tune", pos=(0, 0), height=self.PANEL_HEIGHT, width=self.PANEL_WIDTH, no_collapse=True):
            with dpg.group(horizontal=True):
                dpg.add_input_text(label="Gain (dB, all cameras)", tag="tune_gain", width=90,
                                   hint=gain_hint, default_value=self.values["GAIN_DB"])
                with dpg.tooltip("tune_gain"):
                    dpg.add_text("Startup gains (leave blank to keep each camera's settings)")
                    for key, reading in gains.items():
                        dpg.add_text(f"{key}   {reading}")
                dpg.add_input_double(label="Target FPS per camera", tag="tune_fps", width=120,
                                     step=1.0, format="%.4f", default_value=float(self.values["FPS"]))
                dpg.add_button(label="Save applied settings", callback=save)
            for bank in BANKS:
                lit_percent = float(self.values[bank.upper() + "_ILLUMINATION_PERCENT"])
                with dpg.group(horizontal=True):
                    dpg.add_button(label=f"Apply {bank}", width=145, callback=apply_bank, user_data=bank)
                    dpg.add_input_int(label=f"Exposure (µs)##{bank}", tag=bank + "_us", width=140,
                                      default_value=int(self.values[bank.upper() + "_EXPOSURE_US"]))
                    dpg.add_input_double(label=f"Voltage (V)##{bank}", tag=bank + "_volts", width=120,
                                         step=.1, format="%.3f", default_value=float(self.values[bank.upper() + "_VOLTAGE"]))
                    dpg.add_input_double(label=f"Underfill (% dark)##{bank}", tag=bank + "_underfill", width=105,
                                         step=1, format="%.2f", default_value=round(100 - lit_percent, 2))
            with dpg.group(horizontal=True):
                for axis in ("width", "height"):
                    dpg.add_input_int(label=f"ROI {axis} (px)", tag="roi_" + axis, width=115,
                                      default_value=int(self.values["ROI_" + axis.upper()]))
                dpg.add_button(label="Apply centered crop", callback=crop)
                dpg.add_text(current_size(), tag="roi_size")
            dpg.add_text(f"Previewing {self.mode}. Target {self.values['FPS']} FPS.", tag="tune_status")
            with dpg.group(horizontal=True):
                dpg.add_input_double(label="Margin (ms)", tag="margin_ms", width=100, default_value=1.5,
                                     min_value=1, max_value=2, min_clamped=True, max_clamped=True, step=.1, format="%.1f")
                dpg.add_button(label="Suggest alternating timing", callback=suggest)
            dpg.add_text("", tag="timing_advice", wrap=self.PANEL_WIDTH - 30)
            if metadata is not None:
                suggest()


@click.command(name="tune")
@click.option("--settings", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Trusted launcher settings file. Defaults to cammy-settings.sh on the desktop.")
@click.option("--mode", type=click.Choice(BANKS), default="surface", show_default=True)
@click.option("--trigger-port", default="/dev/ttyACM0", show_default=True)
@click.pass_context
def tune(ctx, settings, mode, trigger_port):
    """Tune single-bank previews, crop cameras, and estimate alternating timing."""
    path = settings or desktop_settings()
    if not path.is_file():
        raise click.ClickException(f"Cannot find {path}. Run launchers/install.py or supply --settings PATH.")
    try:
        session = TuneSession(path, mode, trigger_port)
        camera_options = Path(session.values["DATA_DIR"]) / session.values["CAMERA_OPTIONS"]
        if not camera_options.is_file():
            raise ValueError(f"Camera options file does not exist: {camera_options}")
    except (ValueError, OSError, OverflowError) as exc:
        raise click.ClickException(str(exc)) from exc
    preview = ctx.parent.command.get_command(ctx.parent, "run")
    ctx.invoke(preview, tuning=session, interface="aravis", hw_trigger=True, trigger_device="nucleo",
               light_mode=session.mode, trigger_port=trigger_port, buffer_size=512, display_downsample=6,
               roi_width=int(session.values["ROI_WIDTH"]), roi_height=int(session.values["ROI_HEIGHT"]),
               camera_options=str(camera_options.resolve()))
