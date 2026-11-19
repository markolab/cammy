# NUCLEO-G474RE trigger controller

For Arduino IDE, copy `trigger-nucleo.ino` into a separate sketch folder named
`trigger-nucleo` before opening it, and select STM32duino's
**Nucleo-64 / Nucleo G474RE** board. Arduino combines `.ino` files in one
sketch folder, so build this firmware separately from the sibling `trigger.ino`.
Serial uses the ST-Link virtual COM port
at 115200 baud. Use `--trigger-port` to select its device path when several
serial devices are present. The sketch deliberately requires this board.

| Pin | Function |
| --- | --- |
| D2 | Surface bank: LOW on, HIGH off |
| D3 | Fluorescence bank: HIGH on, LOW off |
| D4 | Camera: HIGH throughout each exposure, LOW between exposures |
| PA4 | Surface brightness: DAC1 channel 1, 12 bit |
| PA6 | Fluorescence brightness: DAC2 channel 1, 12 bit |

Brightness settings specify the voltage after the external 3.3x amplifier.
Firmware assumes a 3.3 V DAC reference and converts 0.7–10 V settings to
12-bit DAC values using `voltage / (3.3 V * 3.3) * 4095`.
PA4 and PA6 are real DAC outputs on the G474RE; see the
[ST datasheet](https://www.st.com/resource/en/datasheet/stm32g474re.pdf) and
[STM32duino analog output API](https://github.com/stm32duino/Arduino_Core_STM32/wiki/API#analog).
Both banks and D4 are off on startup and after `STOP`; DAC levels are set
before starting the light sequence. Normal Python shutdown sends `STOP`
and closes the port.

To test without Python, open a serial monitor at **115200 baud** with
**Newline** or **Both NL & CR** line endings. Instructions print at boot;
send `HELP` or an empty line to display them again if you connect afterward.
`HELP` stops any running sequence before printing. Instruction lines start
with `# `; omit this prefix when pasting commands. For example:

```text
CONFIG,1,2,200,700,700,8000,5000,1,10000,2000
START
STOP
```

Send each line separately. After `OK CONFIG`, `START` begins alternating
10 ms exposures with 80% surface and 50% fluorescence illumination at
0.7 V brightness. Send `STOP` when finished. The boot/`HELP` instructions
also include surface-only and fluorescence-only examples.

Configure the camera's GenICam section in your camera options TOML, using
the camera's actual trigger input name in place of `Line1` if necessary:

```toml
[genicam.Basler.a]
AcquisitionMode = "Continuous"
AcquisitionFrameRateEnable = false
TriggerSelector = "FrameStart"
TriggerSource = "Line1"
TriggerActivation = "RisingEdge"
ExposureMode = "TriggerWidth"
TriggerMode = "On"
```

These settings use [Trigger Width exposure](https://docs.baslerweb.com/exposure-mode),
where the rising and falling trigger edges bound the exposure.
Python checks these trigger settings before starting acquisition. Python
configures the stopped controller, starts camera acquisition and receiving
threads, then sends `START`. The Nucleo owns all subsequent timing.
Camera readout and trigger latency must fit the chosen intervals.

`--hw-trigger-pulse-width` is exposure time in seconds; repeat it for a
cycle of up to 20 widths. `--hw-trigger-pulse-low` specifies the low time
following each width: supply one to reuse it or one per width. When widths
are explicit, the default low interval is 2 ms. With no widths, Python
derives exposure from `--hw-trigger-rate` (or the camera frame rate), minus
the low interval (default 5 ms). `--alternate-mode 0/1/2` can also select
surface/fluorescence/alternate when `--light-mode` is omitted.

Alternation always starts with surface and switches banks every frame,
independently of the width-list length. At 100% the selected alternating
bank switches off at exposure end. The low interval includes the next
bank's settling time (`--light-settle-us`, default 200 µs), so each period
is exposure plus low interval. Lights turn on that settling time before
D4 rises. Each bank's illumination percentage specifies how far into its
exposure the light switches off; 80 means 80% illuminated and 20% dark.
Underfill never shortens D4's high time. Percentages apply in all modes;
0 produces dark frames and a single bank at 100% stays on across frames.
Illumination percentages have 0.01% resolution, with cutoffs rounded to
microseconds. Timing uses a nonblocking `micros()` loop, so edges have
loop/interrupt latency and require verification on the physical setup.

The serial protocol is ASCII, one command per newline:

```text
HELP (or an empty line)      -> stops outputs and prints instructions
HELLO                       -> CAMMY_NUCLEO,1 (also stops outputs)
CONFIG,1,mode,settle_us,surface_mV,fluorescence_mV,surface_fill,fluorescence_fill,n,exposure_us,low_us,...
                            -> OK CONFIG
START                       -> OK START
STOP                        -> OK STOP
```

Mode is 0 (surface), 1 (fluorescence), or 2 (alternate). Fill values are
hundredths of a percent, from 0 to 10000. There are `n` exposure/low pairs.
Low intervals must be positive and at least the settling time; exposures
must be positive. Each exposure plus low interval must fit 2³¹−1 µs.
Invalid commands stop the outputs and return `ERR ...`; a rejected
configuration cannot be started. Python uses bounded acknowledgement
timeouts and attempts `STOP` if starting fails.

Verification from the repository root (Python requires `pyserial`):

```sh
python -m unittest discover -s tests -v
clang++ -std=c++17 -Wall -Wextra -Werror -DARDUINO_NUCLEO_G474RE \
  -Itests/firmware tests/firmware/test_nucleo.cpp -o /tmp/test-nucleo
/tmp/test-nucleo
mkdir -p /tmp/trigger-nucleo
cp cammy/trigger-nucleo/trigger-nucleo.ino /tmp/trigger-nucleo/trigger-nucleo.ino
arduino-cli compile --fqbn STMicroelectronics:stm32:Nucleo_64:pnum=NUCLEO_G474RE \
  --warnings all /tmp/trigger-nucleo
```
