# Shared by the capture and tuning launchers. Exposure widths are in microseconds.
GAIN_DB=""                      # Empty uses the gain from the camera TOML.
ROI_WIDTH=0                     # Overrides the TOML crop. 0 uses the full sensor axis.
ROI_HEIGHT=0
ROI_HORIZONTAL=center           # Part to keep: left, center, or right.
ROI_VERTICAL=center             # Part to keep: top, center, or bottom.
FLUORESCENCE_EXPOSURE_US=10000
SURFACE_EXPOSURE_US=1000
FPS=93.0233                     # Total frames/s per camera. Alternate pairs run at FPS/2.
STREAM_STATS=on                 # Poll and log per-camera stream counters in Preview, Record, and Tune (on/off).
STREAM_STATS_INTERVAL=1         # Seconds between stream-statistics polls (positive whole number).
PREVIEW_TIME_DOWNSAMPLE=1       # Display every Nth new preview frame. 0 pauses image updates.
TUNE_DISPLAY=""                 # Tune only: on or off. Empty follows the preview setting (0 is off).

ARAVIS_BUFFER_SIZE=512          # Receive buffers per camera in Preview, Record, and Tune (frames).
RECORD_QUEUE_FRAMES=512         # Record only. ZeroMQ queue limit per camera on each socket (frames).
WRITER_BUFFER_MIB=0             # Extra raw pixel buffer per camera. Try 512 for two-thread recording.
FRAME_WRITER_BATCH_SIZE=20      # Frames per disk write. 16 is 55 MiB at 2000 x 1800 Mono8.
WRITER_CORES=7                  # Record only. Empty lets Cammy choose the writer count.
ACQUISITION_CORES=""            # Preview and Record. Empty uses the remaining worker cores.
SHARED_GUI_CORES=""             # Physical cores shared with Aravis receivers. Empty uses 0 in Preview, 1 in Record.

SURFACE_TO_FLUORESCENCE_GAP_US=500  # Preferred time from surface exposure end to fluorescence exposure start.
LIGHT_SETTLE_US=100                 # Turn on the selected light this far before exposure, within the gap.
SURFACE_POLARITY=low                # D2 level that turns the surface bank on (low/high).
SURFACE_VOLTAGE=4.0                 # Brightness after the 3.3x amplifier (0.7–10 V).
FLUORESCENCE_VOLTAGE=10.0
SURFACE_ILLUMINATION_PERCENT=100     # Percent of exposure illuminated from the start (0–100).
FLUORESCENCE_ILLUMINATION_PERCENT=50

DATA_DIR=/data/qdpi             # Cammy creates session directories here.
COMPRESS_THREADS=""             # FFV1 encoder threads per file. Empty uses all available logical CPUs.
CAMERA_OPTIONS=basler_new_8bit.toml  # Absolute path or relative to DATA_DIR.
