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

### Compress recorded sessions

From a Conda-enabled shell on the acquisition PC, run this separate install once:

```sh
python3 launchers/install.py --compression
```

This creates or updates the `markolab-cli` environment from
[markolab-cli.yml](launchers/markolab-cli.yml) and installs **Cammy Compress
Sessions** on the desktop. The install needs Git access to the lab's
`markolab-cli` and `markovids` repositories. An optional desktop directory
argument works the same as for the capture installer.

After recording and writer shutdown finish, click the shortcut. It reads
`DATA_DIR` from the same desktop `cammy-settings.sh` and compresses up to
two `session_*` directories concurrently. Each session converts one camera
`.dat` at a time using `markolab-cli convert-dat-to-avi` (lossless FFV1).
Set `COMPRESS_SESSIONS=3` to process three directories at once, or `1` for
sequential processing. `COMPRESS_THREADS` sets encoder threads per file.
Leaving it empty divides the available logical CPUs across the configured jobs,
up to five threads each. `COMPRESS_CHUNK_SIZE` sets frames per chunk for
encoding and verification (default 64). The runner prints all three settings at startup.
Each AVI appears beside its DAT after the converter's pixel integrity check
succeeds. Raw data, timestamps, and metadata are retained. A second click
during compression reports the active job. On later runs, existing nonempty
AVIs at least as recent as their DATs are skipped. Errors are reported per
file and processing continues. Inspect any reported stale outputs before retrying.

### Capture settings

Edit `cammy-settings.sh` beside the desktop shortcuts and save it before the
next run. All shortcuts read that file at startup. The installer copies it
from [launchers/settings.sh](launchers/settings.sh) on first install. Rerun the
installer to add new settings to an older desktop copy as commented examples.
Uncomment an example to override its default. The settings include gain in dB
(empty uses the camera TOML), exposure widths in µs, and total FPS per camera.
Set `STREAM_STATS_INTERVAL=5` to poll and log camera statistics every five seconds
in Preview, Record, and Tune. The default is one second. Set `STREAM_STATS=off`
to disable these polls. The GUI frame counters stay
active in either case. For a single Tune session, use `--stream-stats` or
`--no-stream-stats` to override the file.

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
experiment dialog and uses seven writer cores, batches of 20, and disabled
image updates. `ARAVIS_BUFFER_SIZE=512` sets the number of receive buffers per
camera in Preview, Record, and Tune. `RECORD_QUEUE_FRAMES=512` sets the per-camera
ZeroMQ queue limit in frames on both the sending and receiving sockets during
recording. Both settings take positive integers. Direct preview and recording runs from the checkout use
`launchers/settings.sh` by default.

Set `PREVIEW_TIME_DOWNSAMPLE=4` in `cammy-settings.sh` to show every fourth new
frame the GUI sees in Preview and Tune. The default is `1`. During a preview,
open **Display → Every Nth frame** to change it for all cameras. Set it to `0`
to pause images while acquisition and frame counters continue. Tune's **Save
applied settings** makes the new value the default for future previews. Direct
runs accept `--display-time-downsample N`. After preview initialization,
`cammy run` uploads a camera texture only when a new image is selected for
display. With `--record --display-time-downsample 0`, the window shows FPS,
acquisition misses and writer drops without allocating camera textures. This
status-only window reads camera counters at most ten times a second. Acquisition
and writing run independently between refreshes.

```sh
bash launchers/capture.sh fluorescence preview
bash launchers/capture.sh alternate record
```

## Live tuning

Open **Cammy Tune**, run `bash launchers/capture.sh tune`, or run `cammy tune`
in the acquisition environment to start with surface illumination. Use
`--mode fluorescence` or `--mode alternate` to start in either other mode.
It reads the same desktop `cammy-settings.sh` as the launchers. Use
`--settings "/path/to/cammy-settings.sh"` for a custom location and
`--trigger-port` for a different serial port.

Use **Display camera images** in the Tune panel to compare acquisition drops
with the images on and off. Switching logs each camera's missed frames and
received FPS for the preceding interval and starts a fresh measurement while
triggering continues. With images off, Tune skips preview image processing,
saturation counts and camera texture uploads. Turning images back on uses the
selected stride, or 1 for a paused stride.
Use `cammy tune --display off` to start with images hidden. **Save applied settings**
stores the switch as `TUNE_DISPLAY=on` or `off` for future Tune sessions.

Target FPS and gain are shared across cameras. With no gain override, the gain
field shows the startup camera reading, or **Mixed** if readings differ. Hover
over it for each camera's reading. Leave the field blank to keep each camera's
original gain settings, or enter a number to set all cameras to that gain.

Each bank has its own exposure (µs), voltage (V), and **Underfill (% dark)**.
For example, 25% underfill illuminates the first 75% of the exposure and saves
`*_ILLUMINATION_PERCENT=75`. The camera integrates for the full exposure.
Click **Apply surface** or **Apply fluorescence** to preview that bank. Click
**Apply alternate** to test both sets of fields, starting with surface.
Apply checks the exposure against the
configured FPS and settle time, pauses the Nucleo, and restarts triggering over
the same connection. Changing gain also briefly disarms and rearms the cameras.
Tune handles input every GUI frame and refreshes at most two camera images per
frame, targeting 15 preview updates per second per camera. Each update uses
the latest acquired image. Each camera uses a fixed full-range display and
shows the percentage of saturated pixels in a full-resolution frame up to
four times a second.

Each Apply and each crop change starts a fresh acquisition trial. Compare the
missed-frame count (gaps in camera frame IDs) and the mean received FPS against
the requested total FPS. A camera can ignore triggers and still deliver
consecutive frame IDs. In alternation, the additional FPS/channel reading uses
timestamps from the same bank. For example, `cammy tune --mode alternate --stream-stats`
also logs Aravis stream counters once a second. To measure writer drops with the
same timing, save the applied settings and use that mode's Record launcher.

**Save applied settings** writes the values you tried to the shared settings
file. The saved FPS is the total frame rate per camera for all launchers.
In alternation, each bank receives half that rate. Closing the preview stops
the lights and restores camera gain.

**Apply crop** sets the width and height in camera pixels on every camera.
A width or height overrides the camera TOML's size and offset on that axis.
Enter `0` to use the full available width or height after binning. The desktop
launchers pass both dimensions, so two zeros select the full frame. Position
the crop using left, center, or right and top, center, or bottom. Both axes
default to center when a size is specified. Tune briefly
pauses triggering and resizes the receive buffers on the affected streams.
The preview stays open and shows the crop at its original position with black
padding. Saturation and recordings use the cropped pixels only.
**Save applied settings** writes `ROI_WIDTH`, `ROI_HEIGHT`, `ROI_HORIZONTAL`
and `ROI_VERTICAL` for all preview and recording launchers. Both tune and capture
restore the camera's original region on exit. Direct runs accept the same
choices, e.g. `--roi-width 1024 --roi-horizontal right` takes the rightmost
1024 columns and keeps the configured height and vertical offset. On the CLI,
omit all options for an axis to use its TOML crop, or pass only its position
to move the configured size. Each session's
`metadata.toml` stores the camera's actual `OffsetX`, `OffsetY`, `Width`, and
`Height` under `camera_metadata.<camera ID>`. The requested flags are in
`cli_parameters`.

**Suggest alternating timing** uses the applied crop and the draft exposures,
underfill, target FPS, and **Surface → fluorescence gap**. The gap starts at
the end of the surface exposure and includes the fluorescence light's settling
time. For example, to leave 200 µs dark after surface with 100 µs of settling,
enter 300 µs. The field starts at the launcher's `SURFACE_TO_FLUORESCENCE_GAP_US`.
Enter **0** to estimate the shortest feasible gap for the selected exposures.

Tune displays both the shortest gap and the proposed sequence. It reserves the
larger of the requested and shortest gaps when estimating the FPS ceiling, and
accounts for the requested gap in the maximum fluorescence exposure at your
target FPS. **Suggest alternating timing** leaves the live sequence alone.
**Apply alternate** uses the estimate when the gap is zero and reports if it
cannot fit. A positive gap lets you test a specific timing, even below the
estimated readout margin. As in the launcher, it is shortened if needed to
leave settling time before the next surface frame. **Save applied settings**
writes the gap actually used to `SURFACE_TO_FLUORESCENCE_GAP_US`.

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

With the Nucleo, `cammy run` and `cammy tune` print each acknowledged trigger
configuration once. The terminal shows one full repeating cycle, including
each exposure, following gap, light-on interval, brightness, total FPS and
alternating pairs/s. Times are programmed microseconds relative to the first
D4 rising edge. The final `CONFIG` line is the exact command sent to the controller.

On Linux, set `WRITER_CORES`, `ACQUISITION_CORES`, and `SHARED_GUI_CORES` in
the desktop settings for Preview and Record. For direct runs use `--writer-cores`
(recording only), `--acquisition-cores`, and `--shared-gui-cores`. Counts are
physical cores within the process's allowed CPUs. SMT siblings stay together.
After setting aside the first core for the GUI, an unspecified worker count gets
the remaining cores. If both worker counts are omitted, recording splits them
roughly equally. Specifying both can leave cores unused by these workers.

With `SHARED_GUI_CORES=0` the GUI has its own core. At `1`, Aravis receivers can
also run on that core (the existing `--share-gui-core` behavior). At `2`, the GUI
can also run on the first acquisition core. Higher counts include more acquisition
cores. Each camera's acquisition reader and raw writer stays pinned to one CPU in
its respective pool. The terminal prints the masks and each worker's TID and CPU.

On a ten-core rig the Record defaults give writers seven cores and readers two.
Receivers can use those two plus the GUI core. Setting `SHARED_GUI_CORES=2` keeps
that worker split and lets the GUI run on two of the three receiver cores. Preview
defaults to a dedicated GUI core and allocates all other cores to acquisition.

Add `--stream-stats` to `cammy run` to poll each Aravis camera and log one line
per camera, even with preview images disabled. The first poll runs on the first GUI
iteration. Use `--stream-stats-interval 5` in Run or Tune to allow at least five
seconds between polls. Logs include free input buffers, queued
completed buffers, reception failures, buffer underruns, acquired/missed frames,
written frames, recording queue drops, and GigE resent/missing packets. Buffer
queue sizes are instantaneous; the other counters are cumulative for the run.
Rising queued buffers and falling free buffers indicate acquisition consumers
falling behind. Increasing underruns mean no free receive buffer was available;
increasing missing packets with healthy buffer queues points to reception loss.
Sampling runs in the GUI loop, without Python stream callbacks.

For synchronized GigE cameras, try `cammy run --transmission-stagger-us 100`
with your usual capture options. Cammy sorts the selected GigE cameras by ID
and requests frame transmission delays of 0, 100, 200 µs and so on across the
whole set, including cameras on different uplinks. Exposures still use the
same trigger sequence. Startup logs show each camera's delay and tick readback,
and recordings save the applied `GevSCFTD` ticks in camera metadata. Omit the
option to use the camera/TOML settings, or pass 0 for an all-zero comparison.
Cammy restores the previous settings on normal exit.

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

For raw recordings, `--writer-buffer-mib 512 --frame-writer-batch-size 16`
gives each camera a receiver that copies frames into a preallocated pool and a
disk thread that writes 16 frames at a time. At 2000 × 1800 Mono8, each pool holds
149 frames and each full write is 54.9 MiB. Eight pools add about 4 GiB of RAM
on top of the ZeroMQ queues. A write keeps its slots until both pixels and
timestamps have been written. When the pool and ZeroMQ queues fill, acquisition
continues and counts rejected recording frames as usual.

Set `WRITER_BUFFER_MIB` and `FRAME_WRITER_BATCH_SIZE` in the launcher settings.
The default buffer size of 0 selects the original single-threaded recorder.
With buffering enabled, `--stream-stats` reports occupied/total slots, peak
occupancy, the number of times the receiver waited on a full pool, and the last
and longest completed batch write in milliseconds. Batch timings exclude the
final file sync.

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
