# cammy
Simple python package to record from machine vision cameras (Basler and Lucid in particular)

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
[`trigger-nucleo.ino`](cammy/trigger/trigger-nucleo.ino) using the
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
