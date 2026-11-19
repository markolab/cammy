# cammy
Simple python package to record from machine vision cameras (Basler and Lucid in particular)

## Desktop launchers for the Nucleo rig

From the updated checkout on the acquisition PC, use the terminal where Cammy
works. Verify that its CLI can load Aravis, then install the shortcuts.

```sh
cammy tune --help && python3 launchers/install.py
```

This puts **Cammy Tune** on your Linux desktop alongside the **Preview** and
**Record** shortcuts for **Surface**, **Fluorescence**, and **Alternate**. If GNOME prompts,
right-click the shortcut and choose **Allow Launching**. The shortcuts use this
checkout and the Cammy executable found at installation. When a Conda
environment is active during installation, each shortcut uses `conda run`
with that environment's full path. The installer also carries over exported
`GI_TYPELIB_PATH` and `LD_LIBRARY_PATH` values for custom Aravis installations.
Rerun the installer after moving the checkout or changing the environment or
library paths.

Edit `cammy-settings.sh` beside the desktop shortcuts and save it before the
next run. All seven shortcuts read that file at startup. The installer copies it
from [launchers/settings.sh](launchers/settings.sh) on first install. Rerun the
installer to add new settings to an older desktop copy as commented examples.
Uncomment an example to override its default. The settings include gain in dB
(empty uses the camera TOML), exposure widths in µs, and total FPS per camera.

With the default alternating settings, one pair takes 21.5 ms (46.51 pairs/s,
93.02 frames/s per camera). Times below are relative to the surface trigger rising edge.

| Bank | Camera exposure | Light enabled | Brightness after amplifier |
| --- | --- | --- | --- |
| Surface | 0–1 ms | −0.1 to 1 ms | 4 V |
| Fluorescence | 1.5–11.5 ms | 1.4–6.5 ms | 10 V |

`SURFACE_TO_FLUORESCENCE_GAP_US` sets the preferred gap from surface exposure end
to fluorescence exposure start (500 µs). The remaining low time follows fluorescence
(10 ms by default). Both gaps include `LIGHT_SETTLE_US` (100 µs by default). If the
preferred gap leaves too little settling time before the next surface frame, the
launcher reports the shorter gap it uses to maintain FPS.

`SURFACE_VOLTAGE` and `FLUORESCENCE_VOLTAGE` set independent brightness levels
at the output of the 3.3× amplifier (0.7–10 V). Each bank's
`*_ILLUMINATION_PERCENT` controls how far into its exposure the light stays on.
At 50%, fluorescence switches off 5 ms into the 10 ms exposure. At 0% the bank
stays off. In single-light modes the selected bank gets the full frame rate
and 100% fill keeps it on across frames.

The defaults use the rig's `/data/qdpi/basler_new_8bit.toml` and save recordings
under `/data/qdpi/session_*`. Preview shows live images. Record opens the usual
experiment dialog and uses seven writer cores, batches of 20, a queue setting
of 512, and disabled image updates. Direct preview and recording runs from the checkout use
`launchers/settings.sh` by default.

```sh
bash launchers/capture.sh fluorescence preview
bash launchers/capture.sh alternate record
```

## Live tuning

Open **Cammy Tune**, run `bash launchers/capture.sh tune`, or run `cammy tune`
in the acquisition environment to start with surface illumination. Run
`cammy tune --mode fluorescence` to start with fluorescence.
It reads the same desktop `cammy-settings.sh` as the launchers. Use
`--settings "/path/to/cammy-settings.sh"` for a custom location and
`--trigger-port` for a different serial port.

Target FPS and gain are shared across cameras. With no gain override, the gain
field shows the startup camera reading, or **Mixed** if readings differ. Hover
over it for each camera's reading. Leave the field blank to keep each camera's
original gain settings, or enter a number to set all cameras to that gain.

Each bank has its own exposure (µs), voltage (V), and **Underfill (% dark)**.
For example, 25% underfill illuminates the first 75% of the exposure and saves
`*_ILLUMINATION_PERCENT=75`. The camera integrates for the full exposure.
Click **Apply surface** or **Apply fluorescence** to preview that bank.
Apply checks the exposure against the
configured FPS and settle time, pauses the Nucleo, and restarts triggering over
the same connection. Changing gain also briefly disarms and rearms the cameras.
Each camera uses a fixed full-range display and shows the percentage of
saturated pixels in a full-resolution frame, refreshed four times a second.

**Save applied settings** writes the values you tried to the shared settings
file. The saved FPS is the total frame rate per camera for all launchers.
In alternation, each bank receives half that rate. Closing the preview stops
the lights and restores camera gain. For the final timing and illumination
duty cycle, check **Alternate Preview**.

**Apply centered crop** sets the width and height in camera pixels on every
camera. Enter `0` on either axis to keep its configured extent. Tune briefly
reopens the preview so each stream and image buffer uses the new dimensions.
**Save applied settings** writes `ROI_WIDTH` and `ROI_HEIGHT` for all preview and
recording launchers. Both tune and capture restore the camera's original region
on exit. Direct runs accept `--roi-width` and `--roi-height`.

**Suggest alternating timing** uses the applied crop and the exposure, underfill,
and target FPS currently in the fields. You can try a draft fluorescence exposure
without applying it. The result shows the surface exposure, gap to fluorescence,
fluorescence exposure and lit time, and wait before the next surface trigger.
Tune also prints the estimate in the terminal. To try that schedule in
**Alternate Preview**, use the displayed exposure widths, the displayed gap for
`SURFACE_TO_FLUORESCENCE_GAP_US` and an FPS at or below the displayed ceiling.

Tune assumes overlapping acquisition. It spaces successive **exposure ends** by
at least the slowest sensor readout at the applied ROI plus a 1.5 ms margin
(adjustable from 1 to 2 ms). The full fluorescence exposure counts toward
surface-frame readout, including its dark portion. When that exposure covers
the readout and margin, the gap after surface can be as short as the light
settling time. Any spare time at the target FPS goes after fluorescence.
Tune honors explicit camera overlap limits and identifies any camera missing
a sensor readout value. [Basler sensor readout time](https://docs.baslerweb.com/sensor-readout-time)
and [overlapping acquisition](https://docs.baslerweb.com/overlapping-image-acquisition)

## Acquisition and recording

On Linux, `--record --writer-cores 2` reserves two physical cores for writers,
one for the GUI, and gives the remaining cores to acquisition. SMT siblings stay
in the same pool. On a ten-core machine this gives seven acquisition cores.
The default still splits worker cores roughly equally between acquisition and
recording. Compare acquisition misses and recording drops using `--stream-stats`
at the same trigger settings; omit `--writer-cores` to restore the default split.

Add `--stream-stats` to `cammy run` to log one line per Aravis camera per second,
even with preview images disabled. Logs include free input buffers, queued
completed buffers, reception failures, buffer underruns, acquired/missed frames,
written frames, recording queue drops, and GigE resent/missing packets. Buffer
queue sizes are instantaneous; the other counters are cumulative for the run.
Rising queued buffers and falling free buffers indicate acquisition consumers
falling behind. Increasing underruns mean no free receive buffer was available;
increasing missing packets with healthy buffer queues points to reception loss.
Sampling runs in the GUI loop, without Python stream callbacks.

On Linux, use `--receiver-realtime-priority 10` to try `SCHED_RR` realtime
scheduling for native GigE receiver threads. The default is 0 (disabled).
Priority is applied once at startup without Python stream callbacks. Each
receiver TID and result is logged; permission failures keep existing scheduling.
This requires an appropriate realtime priority limit (`ulimit -r`) or
`CAP_SYS_NICE`. Omit the option or set it to 0 on the next run to disable it.

Use `--record-queue-frames N` (default 64) to tune recording stall buffering.
The limit applies per camera on both the sending and receiving sides. Queue
memory is roughly `2 * N * frame_bytes` per camera, in addition to writer
batches and filesystem cache. For five 2048x2048 Mono8 cameras, `N=256`
allows roughly 10 GiB of queued pixels. Zero is rejected to keep queues bounded.

On normal shutdown, capture stops first, then recording queues drain and the
final partial batch is written. Writers sync and close their own files before
the program exits; this can take time while disk writes finish. Recording
errors are reported rather than treating a failed writer as a successful save.

For the NUCLEO-G474RE light/camera controller, flash
[`trigger-nucleo.ino`](cammy/trigger-nucleo/trigger-nucleo.ino) using the
STM32duino **Nucleo G474RE** board. Copy it into its own sketch folder for
Arduino builds, as described in the [setup instructions](cammy/trigger/README.md).
For standalone testing, open a serial monitor at 115200 baud with Newline
line endings and send `HELP` to display configuration examples.

```sh
cammy run --hw-trigger --trigger-device nucleo --trigger-port /dev/ttyACM0 \
  --camera-options camera_options.toml --light-mode alternate \
  --hw-trigger-pulse-width 0.010 --hw-trigger-pulse-low 0.002 \
  --surface-illumination-percent 80 --fluorescence-illumination-percent 50 \
  --surface-voltage 3 --fluorescence-voltage 5
```

This alternates surface and fluorescence with a 10 ms exposure and 2 ms low
interval per frame. Surface illumination ends 8 ms into its exposure;
fluorescence ends 5 ms into its exposure. D4 remains high for the full 10 ms.
Use `--light-mode surface` or `--light-mode fluorescence` for one bank.
Each bank defaults to 100% illumination; in single-bank mode this keeps it
on continuously. Brightness voltages are **after** the 3.3x amplifier,
range from 0.7 to 10 V, and default to 0.7 V.

The camera must use `TriggerWidth` exposure with a rising-edge `FrameStart`
trigger on the input wired to D4. See the
[wiring, camera settings, and timing protocol](cammy/trigger/README.md).
Existing setups continue to use the original Arduino firmware by default.
