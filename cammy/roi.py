"""Centered hardware crops and advisory timing for alternating exposures."""
import math

from cammy.trigger.nucleo import NucleoTriggerDevice


def centered_region(camera, original, width, height):
    """Zero preserves the corresponding axis of the configured camera region."""
    x, y, w, h = original

    def axis(name, size, start, span, bounds, increment, offset_increment):
        if size in (0, span):
            return start, span
        minimum, _ = bounds()
        step = max(1, increment())
        if not minimum <= size <= span or (size - minimum) % step:
            raise ValueError(f"{name} must be {minimum}–{span} px in steps of {step}, or 0.")
        if not camera.is_region_offset_available():
            raise ValueError("The camera does not support centered offsets.")
        step = max(1, offset_increment())
        return start + (span - size) // (2 * step) * step, size

    x, w = axis("Width", width, x, w, camera.get_width_bounds, camera.get_width_increment,
                camera.get_x_offset_increment)
    y, h = axis("Height", height, y, h, camera.get_height_bounds, camera.get_height_increment,
                camera.get_y_offset_increment)
    return x, y, w, h


def set_region(camera, region):
    if tuple(camera.get_region()) == region:
        return
    camera.set_region(*region)
    actual = tuple(camera.get_region())
    if actual != region:
        raise ValueError(f"Camera requested {region}, read back {actual}.")


def timing_advice(metadata, surface, fluorescence, dark_percent, fps, settle, margin_ms):
    """Estimate alternating trigger gaps with overlapping sensor readout at the applied ROI (µs)."""
    if not (surface > 0 and fluorescence > 0 and 0 <= dark_percent <= 100 and
            math.isfinite(fps) and fps > 0 and 1 <= margin_ms <= 2):
        raise ValueError("Use positive exposures and FPS, 0–100% underfill and a 1–2 ms margin.")
    period = 2e6 / fps
    limit = NucleoTriggerDevice.MAX_INTERVAL_US
    if not math.isfinite(period) or period > 2 * limit:
        raise ValueError("FPS is outside the trigger timing range.")
    period = round(period)
    if not metadata:
        raise ValueError("Timing unavailable. No camera readout data.")

    slowest_us = 0
    gap_floor = max(1, settle)
    min_width = 1
    for key, features in metadata.items():
        for node in ("SensorReadoutTime", "ReadoutTimeAbs"):
            readout = features.get(node)
            if type(readout) in (float, int) and math.isfinite(readout) and readout > 0:
                break
        else:
            raise ValueError(f"Timing unavailable. {key} did not report a valid sensor readout for this ROI.")
        slowest_us = max(slowest_us, readout)
        guard = math.ceil(readout + 1000 * margin_ms)
        if features.get("OverlapMode") == "Off" or features.get("TriggerOverlap") == "Off":
            gap_floor = max(gap_floor, guard)
        elif features.get("ExposureOverlapTimeMax") is not None and features.get("ExposureOverlapTimeMode") != "Automatic":
            # Basler uses this value as the shortest expected trigger width in manual mode.
            allowed = features["ExposureOverlapTimeMax"]
            if type(allowed) not in (float, int) or not math.isfinite(allowed) or not 0 <= allowed <= min(surface, fluorescence):
                raise ValueError(f"Check {key}'s ExposureOverlapTimeMax ({allowed} µs) against the shorter exposure.")
            min_width = max(min_width, math.ceil(allowed))
            gap_floor = max(gap_floor, math.ceil(guard - allowed))

    # Each exposure must end after the previous frame has finished reading out.
    guard = math.ceil(slowest_us + 1000 * margin_ms)
    before_fluo = max(gap_floor, guard - fluorescence)
    before_surface = max(gap_floor, guard - surface)
    fastest = surface + before_fluo + fluorescence + before_surface
    tail = before_surface + max(0, period - fastest)
    if max(surface + before_fluo, fluorescence + tail) > limit:
        raise ValueError("The proposed schedule exceeds the Nucleo interval range.")
    max_fluo = period - surface - gap_floor - before_surface
    envelope = (f"At {fps:g} FPS, fluorescence can be up to {max_fluo} µs with this surface exposure."
                if max_fluo >= min_width and max_fluo + gap_floor >= guard and max_fluo + before_surface <= limit
                else f"{fps:g} FPS is outside this budget with the selected surface exposure.")
    # Round down so copying the displayed ceiling to a launcher still fits.
    precision = 100 if fastest <= 2_000_000 else 1_000_000
    fps_limit = (2_000_000 * precision // fastest) / precision
    return (f"Slowest sensor readout {slowest_us / 1000:.2f} ms | margin {margin_ms:g} ms\n"
            f"Surface {surface} µs → gap {before_fluo} µs → fluorescence {fluorescence} µs "
            f"({fluorescence * (1 - dark_percent / 100):.0f} µs lit) → wait {tail} µs ({before_surface} µs min) → repeat\n"
            f"Estimated ceiling {fps_limit:g} FPS total ({fps_limit / 2:g} pairs/s). {envelope}")
