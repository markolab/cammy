# Shared by all six Cammy launchers. Exposure widths are in microseconds.
GAIN_DB=""                      # Empty uses the gain from the camera TOML.
FLUORESCENCE_EXPOSURE_US=10000
SURFACE_EXPOSURE_US=1000
FPS=93.0233                     # Total frames/s per camera. Alternate pairs run at FPS/2.

SURFACE_TO_FLUORESCENCE_GAP_US=500  # Preferred time from surface exposure end to fluorescence exposure start.
LIGHT_SETTLE_US=100                 # Turn on the selected light this far before exposure, within the gap.
SURFACE_VOLTAGE=4.0                 # Brightness after the 3.3x amplifier (0.7–10 V).
FLUORESCENCE_VOLTAGE=10.0
SURFACE_ILLUMINATION_PERCENT=100     # Percent of exposure illuminated from the start (0–100).
FLUORESCENCE_ILLUMINATION_PERCENT=50

DATA_DIR=/data/qdpi             # Cammy creates session directories here.
CAMERA_OPTIONS=basler_new_8bit.toml  # Absolute path or relative to DATA_DIR.
