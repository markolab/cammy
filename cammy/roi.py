"""Hardware crops and advisory timing for alternating exposures."""
import math

from cammy.trigger.nucleo import NucleoTriggerDevice

HORIZONTAL_POSITIONS = ("left", "center", "right")
VERTICAL_POSITIONS = ("top", "center", "bottom")


def crop_region(camera, original, width, height, horizontal="center", vertical="center"):
    """Zero preserves the corresponding axis of the configured camera region."""
    x, y, w, h = original

    def axis(name, size, start, span, position, choices, bounds, increment, offset_increment):
        if position not in choices:
            raise ValueError(f"{name} position must be one of {', '.join(choices)}.")
        if size in (0, span):
            return start, span
        minimum, _ = bounds()
        step = max(1, increment())
        if not minimum <= size <= span or (size - minimum) % step:
            raise ValueError(f"{name} must be {minimum}–{span} px in steps of {step}, or 0.")
        shift = span - size
        if position == choices[0]:
            return start, size
        if position == "center":
            shift //= 2
        if not camera.is_region_offset_available():
            raise ValueError(f"The camera does not support positioning the crop at {position}.")
        step = max(1, offset_increment())
        return start + shift // step * step, size

    x, w = axis("Width", width, x, w, horizontal, HORIZONTAL_POSITIONS, camera.get_width_bounds, camera.get_width_increment,
                camera.get_x_offset_increment)
    y, h = axis("Height", height, y, h, vertical, VERTICAL_POSITIONS, camera.get_height_bounds, camera.get_height_increment,
                camera.get_y_offset_increment)
    return x, y, w, h


def centered_region(camera, original, width, height):
    return crop_region(camera, original, width, height)


def set_region(camera, region):
    if tuple(camera.get_region()) == region:
        return
    camera.set_region(*region)
    actual = tuple(camera.get_region())
    if actual != region:
        raise ValueError(f"Camera requested {region}, read back {actual}.")


def timing_advice(metadata, surface, fluorescence, dark_percent, fps, settle, margin_ms, requested_gap=0):
    """Estimate alternating trigger gaps with overlapping sensor readout at the applied ROI (µs)."""
    if not (surface > 0 and fluorescence > 0 and 0 <= dark_percent <= 100 and
            math.isfinite(fps) and fps > 0 and 1 <= margin_ms <= 2):
        raise ValueError("Use positive exposures and FPS, 0–100% underfill and a 1–2 ms margin.")
    period = 2e6 / fps
    limit = NucleoTriggerDevice.MAX_INTERVAL_US
    if not 0 <= requested_gap <= limit or int(requested_gap) != requested_gap:
        raise ValueError("Use a nonnegative whole number of µs for the surface-to-fluorescence gap.")
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
    shortest_gap = max(gap_floor, guard - fluorescence)
    before_fluo = max(requested_gap, shortest_gap)
    before_surface = max(gap_floor, guard - surface)
    fastest = surface + before_fluo + fluorescence + before_surface
    tail = before_surface + max(0, period - fastest)
    if max(surface + before_fluo, fluorescence + tail) > limit:
        raise ValueError("The proposed schedule exceeds the Nucleo interval range.")
    # A longer candidate fluorescence exposure may cover more of the surface readout.
    first_gap_floor = max(gap_floor, requested_gap)
    max_fluo = period - surface - first_gap_floor - before_surface
    envelope = (f"At {fps:g} FPS, fluorescence can be up to {max_fluo} µs with this surface exposure."
                if max_fluo >= min_width and max_fluo + first_gap_floor >= guard and max_fluo + before_surface <= limit
                else f"{fps:g} FPS is outside this budget with the selected surface exposure.")
    if period < fastest:
        envelope = f"Selected pair exceeds the target period by {fastest - period} µs. {envelope}"
    # Round down so copying the displayed ceiling to a launcher still fits.
    precision = 100 if fastest <= 2_000_000 else 1_000_000
    fps_limit = (2_000_000 * precision // fastest) / precision
    return (f"Slowest sensor readout {slowest_us / 1000:.2f} ms | margin {margin_ms:g} ms | "
            f"Shortest surface-to-fluorescence gap {shortest_gap} µs for these exposures\n"
            f"Surface {surface} µs → gap {before_fluo} µs → fluorescence {fluorescence} µs "
            f"({fluorescence * (1 - dark_percent / 100):.0f} µs lit) → wait {tail} µs ({before_surface} µs min) → repeat\n"
            f"Estimated ceiling {fps_limit:g} FPS total ({fps_limit / 2:g} pairs/s). {envelope}")
