#!/usr/bin/env bash
set -eu

die() { echo "$*" >&2; exit 1; }
run_cammy() {
    {
        printf 'cd -- %q &&' "$PWD"
        printf ' %q' "$@"
        printf '\n'
    } >&2
    "$@"
}
check_us() {
    [[ $2 =~ ^(0|[1-9][0-9]{0,9})$ ]] && (( $2 >= $3 && $2 <= 2147483647 )) ||
        die "$1 must be a whole number of microseconds between $3 and 2147483647 (got '$2')."
}
if [[ -t 0 && ( $# -ge 3 || ${1:-} == compress ) ]]; then
    trap 'if (( $? != 0 )); then read -rp "Press Enter to close. " || true; fi' EXIT
fi
if [[ ${1:-} == tune ]]; then
    cammy=${2:-cammy}
    if [[ -n ${3:-} ]]; then
        run_cammy "$cammy" tune --settings "$3"
    else
        run_cammy "$cammy" tune
    fi
    exit
fi
if [[ ${1:-} == compress ]]; then
    source "$(dirname "$0")/settings.sh"
    if [[ -n ${2:-} ]]; then source "$2"; fi
    [[ -n $DATA_DIR && -d $DATA_DIR ]] || die "Cannot find DATA_DIR: $DATA_DIR"
    export COMPRESS_THREADS
    run_cammy python "$(dirname "$0")/compress.py" "$DATA_DIR"
    exit
fi

mode=${1:-}
action=${2:-}
cammy=${3:-cammy}
source "$(dirname "$0")/settings.sh"
if [[ -n ${4:-} ]]; then source "$4"; fi
case "$mode" in
    surface)      widths=("$SURFACE_EXPOSURE_US") ;;
    fluorescence) widths=("$FLUORESCENCE_EXPOSURE_US") ;;
    alternate)    widths=("$SURFACE_EXPOSURE_US" "$FLUORESCENCE_EXPOSURE_US") ;;
    *) die "Choose surface, fluorescence, or alternate." ;;
esac
case "$action" in
    preview) options=(--display-time-downsample "$PREVIEW_TIME_DOWNSAMPLE"
                      --shared-gui-cores "${SHARED_GUI_CORES:-0}") ;;
    record)  options=(--record --display-time-downsample 0 --shared-gui-cores "${SHARED_GUI_CORES:-1}"
                      --frame-writer-batch-size "$FRAME_WRITER_BATCH_SIZE" --writer-buffer-mib "$WRITER_BUFFER_MIB"
                      --record-queue-frames "$RECORD_QUEUE_FRAMES")
             if [[ -n $WRITER_CORES ]]; then options+=(--writer-cores "$WRITER_CORES"); fi ;;
    *) die "Choose preview or record." ;;
esac
if [[ -n $ACQUISITION_CORES ]]; then options+=(--acquisition-cores "$ACQUISITION_CORES"); fi
case "$STREAM_STATS" in
    on) options+=(--stream-stats --stream-stats-interval "$STREAM_STATS_INTERVAL") ;;
    off) ;;
    *) die "STREAM_STATS must be on or off." ;;
esac
if [[ -n $GAIN_DB ]]; then options+=(--gain-db "$GAIN_DB"); fi
options+=(--roi-width "$ROI_WIDTH" --roi-height "$ROI_HEIGHT"
          --roi-horizontal "$ROI_HORIZONTAL" --roi-vertical "$ROI_VERTICAL")

# Calculate one complete cycle at the Nucleo's microsecond resolution.
check_us LIGHT_SETTLE_US "$LIGHT_SETTLE_US" 0
min_low=$(( LIGHT_SETTLE_US > 0 ? LIGHT_SETTLE_US : 1 ))
cycle_us=$(LC_ALL=C awk -v fps="$FPS" -v n="${#widths[@]}" 'BEGIN {
    if (fps !~ /^[0-9]+([.][0-9]+)?$/ || fps <= 0 || n * 1e6 / fps > n * 2147483647) exit 1
    printf "%.0f", n * 1e6 / fps
}') || die "Set FPS to a positive frame rate within the Nucleo timing range."
idle_us=$cycle_us
for width in "${widths[@]}"; do
    check_us Exposure "$width" 1
    idle_us=$((idle_us - width))
done
(( idle_us >= min_low * ${#widths[@]} )) ||
    die "$mode at $FPS FPS needs at least $min_low µs between exposures. Lower FPS or shorten the exposure."
lows=("$idle_us")
if [[ $mode == alternate ]]; then
    check_us SURFACE_TO_FLUORESCENCE_GAP_US "$SURFACE_TO_FLUORESCENCE_GAP_US" "$min_low"
    surface_low=$SURFACE_TO_FLUORESCENCE_GAP_US
    if (( surface_low > idle_us - min_low )); then
        surface_low=$((idle_us - min_low))
        echo "Using a $surface_low µs surface-to-fluorescence gap to fit $FPS FPS." >&2
    fi
    lows=("$surface_low" "$((idle_us - surface_low))")
fi
for i in "${!widths[@]}"; do
    (( widths[i] + lows[i] <= 2147483647 )) || die "Trigger interval exceeds the Nucleo timing range."
    options+=(--hw-trigger-pulse-width "${widths[i]}e-6" --hw-trigger-pulse-low "${lows[i]}e-6")
done

cd -- "$DATA_DIR"
[[ -f $CAMERA_OPTIONS && -r $CAMERA_OPTIONS ]] || die "Cannot read camera settings: $CAMERA_OPTIONS (in $PWD)"
run_cammy "$cammy" run --hw-trigger --trigger-device nucleo --trigger-port /dev/ttyACM0 \
    --camera-options "$CAMERA_OPTIONS" --light-mode "$mode" --light-settle-us "$LIGHT_SETTLE_US" \
    --surface-illumination-percent "$SURFACE_ILLUMINATION_PERCENT" --fluorescence-illumination-percent "$FLUORESCENCE_ILLUMINATION_PERCENT" \
    --surface-polarity "$SURFACE_POLARITY" --surface-voltage "$SURFACE_VOLTAGE" --fluorescence-voltage "$FLUORESCENCE_VOLTAGE" --display-downsample 6 \
    --buffer-size "$ARAVIS_BUFFER_SIZE" "${options[@]}"
