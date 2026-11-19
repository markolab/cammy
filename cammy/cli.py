import click
import numpy as np
import toml
import logging
import sys
import os
import time
import cv2
import threading
import gc
import platform
# gc.disable()

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="[%(asctime)s]:%(levelname)s:%(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


from typing import Optional, Iterable
from cammy.util import (
    get_all_camera_ids,
    intensity_to_rgba,
    intensity_to_rgba_lut, 
    initialize_cameras,
    get_output_format,
    get_pixel_format_bit_depth,
    mpl_to_cv2_colormap,
    get_cpu_affinity,
    check_counters_equal,
)
from cammy.record.base import RECORD_QUEUE_FRAMES
from cammy.record.video import FfmpegVideoRecorder, RawVideoRecorder
from cammy.camera.spoof import SpoofCamera


@click.group()
def cli():
    pass


slider_defaults_min = {
    "default_value": 1800,
    "min_value": 0,
    "max_value": 5000,
}

slider_defaults_max = {
    "default_value": 2200,
    "min_value": 0,
    "max_value": 5000,
}
colormap_default = "gray"
gui_ncols = 3  # number of cols before we start new row
# for labeling videos
font = cv2.FONT_HERSHEY_SIMPLEX
white = (255, 255, 255)
txt_pos = (25, 25)


# TODO:
# 1) ADD OPTION TO READ COUNTERS WITH COLUMN NAME?
# 2) PTPENABLE FOR TIME CLOCK SYNC?
# fmt: off
@cli.command(name="run", context_settings={'show_default': True})
@click.option("--interface", type=click.Choice(["aravis", "fake_custom", "all"]), default="all")
@click.option("--buffer-size", "-b", type=int, default=5, help="Buffer size")
@click.option("--n-fake-cameras", type=int, default=1)
@click.option("--record", is_flag=True, help="Save frames to disk")
@click.option(
    "--writer-cores",
    type=click.IntRange(min=1),
    default=None,
    help="Linux physical cores reserved for writers; remaining worker cores go to acquisition. "
         "Requires --record. Default: half the worker cores."
)
@click.option("--jumbo-frames", default=True, type=bool, help="Turn on jumbo frames (GigE only)")
@click.option(
    "--share-gui-core",
    is_flag=True,
    help="Let Linux Aravis receivers also use the GUI core. Requires --display-time-downsample 0.",
)
@click.option(
    "--receiver-realtime-priority",
    type=click.IntRange(min=0, max=99),
    default=0,
    help="Linux GigE receiver SCHED_RR priority (1-99; 0 disables). Try 10. "
         "Requires realtime scheduling permission."
)
@click.option(
    "--stream-stats",
    is_flag=True,
    help="Log Aravis buffer, frame, packet, and recording counters once per second."
)
@click.option(
    "--save-engine",
    type=click.Choice(["ffmpeg", "raw"]),
    default="raw",
    help="Save raw frames or compressed frames using ffmpeg",
)
@click.option(
    "--display-downsample",
    type=int,
    default=1,
    help="Downsample frames for display (full data is saved)",
)
@click.option("--display-colormap", type=str, default="turbo", help="Look-up-table")
@click.option("--hw-trigger", is_flag=True, help="Trigger frames using an Arduino microcontroller")
@click.option("--trigger-device", type=click.Choice(["arduino", "nucleo"]), default="arduino", help="Trigger firmware to use")
@click.option("--trigger-port", type=str, default=None, help="Serial port of the trigger controller")
@click.option("--light-mode", type=click.Choice(["surface", "fluorescence", "alternate"]), default=None, help="Nucleo light mode (defaults to --alternate-mode: 0, 1, or 2)")
@click.option("--surface-illumination-percent", type=click.FloatRange(0, 100), default=100, help="Nucleo: percent of each surface exposure illuminated")
@click.option("--fluorescence-illumination-percent", type=click.FloatRange(0, 100), default=100, help="Nucleo: percent of each fluorescence exposure illuminated")
@click.option("--surface-voltage", type=click.FloatRange(0.7, 10), default=0.7, help="Nucleo: surface brightness voltage after 3.3x gain")
@click.option("--fluorescence-voltage", type=click.FloatRange(0.7, 10), default=0.7, help="Nucleo: fluorescence brightness voltage after 3.3x gain")
@click.option("--light-settle-us", type=click.IntRange(min=0), default=200, help="Nucleo: light settling time before D4 rises, included in pulse-low time")
@click.option("--hw-trigger-rate", type=float, default=0.0, help="Trigger rate")
@click.option(
    "--hw-trigger-pin-last", type=int, default=13, help="Final dig out pin to use on Arduino"
)
@click.option(
    "--hw-trigger-pulse-width",
    type=float,
    multiple=True,
    # default=[.02],
    help="Hardware trigger pulse hi width (secs)"
)
@click.option(
    "--hw-trigger-pulse-low",
    type=float,
    multiple=True,
    # default=.002,
    help="Hardware trigger pulse low width (secs)"
)
@click.option("--record-counters", type=int, default=0, help="Record counter data")
@click.option("--duration", type=float, default=0, help="Run for N minutes")
@click.option(
    "--camera-options",
    type=click.Path(resolve_path=True),
    default="camera_options.toml",
    help="TOML file with camera options",
)
@click.option(
    "--server",
    is_flag=True,
    help="Activate ZMQ server to send data to and control other python scripts",
)
@click.option(
    "--alternate-mode",
    type=int,
    default=0,
    help="Alternate mode (0 for first set of lights constant, 1 for second lights constant, 2 for alternation)"
)
@click.option(
    "--prefix",
    type=str,
    default=None,
    help="Only uses cameras whose IDs start with this string (None to use all IDs)"
)
@click.option(
    "--display-time-downsample",
    type=int,
    default=1,
    help="Display every nth frame (set to 0 for no display)"
)
@click.option(
    "--record-queue-frames",
    type=click.IntRange(min=1),
    default=RECORD_QUEUE_FRAMES,
    help="Recording queue frames per camera on each side of the connection. "
         "Queue RAM is roughly 2 x this value x frame bytes, plus writer batches."
)
@click.option(
    "--frame-writer-batch-size",
    type=int,
    default=20,
    help="Number of frames to accumulate before writing out to file"
)
@click.option(
    "-v", "--verbose",
    type=bool,
    is_flag=True,
    help="Verbose logging"
)
# fmt: on
def simple_preview(
    interface: str,
    buffer_size: int,
    n_fake_cameras: int,
    camera_options: Optional[str],
    record: bool,
    writer_cores: Optional[int],
    jumbo_frames: bool,
    share_gui_core: bool,
    receiver_realtime_priority: int,
    stream_stats: bool,
    save_engine: str,
    display_downsample: int,
    display_colormap: Optional[str],
    hw_trigger: bool,
    trigger_device: str,
    trigger_port: Optional[str],
    light_mode: Optional[str],
    surface_illumination_percent: float,
    fluorescence_illumination_percent: float,
    surface_voltage: float,
    fluorescence_voltage: float,
    light_settle_us: int,
    hw_trigger_rate: float,
    hw_trigger_pin_last: int,
    hw_trigger_pulse_width: Iterable[float],
    hw_trigger_pulse_low: Iterable[float],
    record_counters: int,
    duration: float,
    server: bool,
    alternate_mode: int,
    prefix: Optional[str],
    display_time_downsample: int,
    record_queue_frames: int,
    frame_writer_batch_size: int,
    verbose: bool,
):
    
    cli_params = locals()

    if writer_cores is not None and not record:
        raise click.UsageError("--writer-cores requires --record")
    if share_gui_core and display_time_downsample != 0:
        raise click.UsageError("--share-gui-core requires --display-time-downsample 0")
    if trigger_device == "nucleo" and not hw_trigger:
        raise click.UsageError("--trigger-device nucleo requires --hw-trigger")
    if trigger_device == "nucleo" and light_mode is None:
        light_mode = {0: "surface", 1: "fluorescence", 2: "alternate"}.get(alternate_mode)
        if light_mode is None:
            raise click.UsageError("Nucleo --alternate-mode must be 0, 1, or 2; or use --light-mode")
    if trigger_device != "nucleo" and light_mode is not None:
        raise click.UsageError("--light-mode requires --trigger-device nucleo")

    alternating = light_mode == "alternate" if trigger_device == "nucleo" else alternate_mode in (2, 3)
    fps_stride = 2 if hw_trigger and alternating else 1
    fps_unit = "FPS/channel" if fps_stride == 2 else "FPS"

    if verbose:
        logging.info("Setting logging to verbose")
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    import dearpygui.dearpygui as dpg
    import cv2
    import socket
    import datetime
    import zmq
    from cammy.camera.aravis import AravisCamera, acquisition_loop, set_receiver_realtime_priority

    basedir = os.path.dirname(os.path.abspath(__file__))
    hostname = socket.gethostname()

    if server:
        context = zmq.Context()
        zsocket = context.socket(zmq.PAIR)
        zsocket.bind("tcp://*:50165")

        # communicate state of program
        logger.info("Sending CLI parameters to client...")
        zsocket.send_pyobj(cli_params)
        logger.info("Done")
    else:
        zsocket = None

    if display_colormap is None:
        display_colormap = mpl_to_cv2_colormap(colormap_default)
    else:
        display_colormap = mpl_to_cv2_colormap(display_colormap)

    if (camera_options is not None) and os.path.exists(camera_options):
        logging.info(f"Loading camera options from {camera_options}")
        camera_dct = toml.load(camera_options)
    else:
        camera_dct = {}

    cameras = {}
    ids = get_all_camera_ids(interface, n_cams=n_fake_cameras, prefix=prefix)
    if not ids:
        raise click.ClickException(
            f"No cameras found for interface={interface!r}, prefix={prefix!r}"
        )

    # TODO: TURN INTO AN AUTOMATIC CHECK, IF NO FRAMES ARE GETTING
    # ACQUIRED, PAUSE FOR 1 SEC AND RE-INITIALIZE
    # don't initialize streams, just alter settings
    cameras = initialize_cameras(
        ids,
        camera_dct,
        jumbo_frames=jumbo_frames,
        record_counters=record_counters,
        buffer_size=buffer_size,
        fps_stride=fps_stride,
    )
    if not cameras:
        raise click.ClickException(
            "No cameras could be initialized; see the camera initialization errors above."
        )

    # de-reference, then initialize again, now with updated settings...
    del cameras
    gc.collect()
    time.sleep(2)

    cameras_metadata = {}
    bit_depth = {}
    spoof_cameras = {}
    trigger_pins = []
    cameras = initialize_cameras(
        ids,
        camera_dct,
        jumbo_frames=jumbo_frames,
        record_counters=record_counters,
        buffer_size=buffer_size,
        fps_stride=fps_stride,
    )
    if not cameras:
        raise click.ClickException(
            "No cameras could be initialized; see the camera initialization errors above."
        )
    try:
        gui_cpus, acquire_cpus, write_cpus = get_cpu_affinity(record, writer_cores=writer_cores)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    logger.info(
        "CPU affinity: GUI=%s, acquisition=%s, recording=%s",
        gui_cpus, acquire_cpus, write_cpus,
    )
    try:
        # Native Aravis receive threads inherit this mask on Ubuntu/Linux.
        if gui_cpus is not None:
            receiver_cpus = set(acquire_cpus)
            if share_gui_core:
                receiver_cpus.update(gui_cpus)
            os.sched_setaffinity(0, receiver_cpus)
            logger.info("Aravis receiver CPUs: %s", sorted(receiver_cpus))
        elif share_gui_core:
            logger.info("--share-gui-core only applies on Linux")
        for camera in cameras.values():
            camera.initialize_acquisition_stream()
    finally:
        if gui_cpus is not None:
            os.sched_setaffinity(0, gui_cpus)
    set_receiver_realtime_priority(receiver_realtime_priority)
    for i, (k, v) in enumerate(cameras.items()):
        feature_dct = v.get_all_features()
        feature_dct = dict(sorted(feature_dct.items()))

        # make sure we set so we know how to decode frame buffers
        # v._pixel_format = feature_dct["PixelFormat"]
        _bit_depth, _spoof_ims = get_pixel_format_bit_depth(feature_dct["PixelFormat"])
        bit_depth[k] = _bit_depth
        cameras_metadata[k] = feature_dct
        trigger_pins.append(hw_trigger_pin_last - i)  # work backwards from last
        for _spoof_name, _spoof_bit_depth in _spoof_ims.items():
            new_id = f"{k}-{_spoof_name}"
            spoof_cameras[new_id] = SpoofCamera(id=new_id)
            spoof_cameras[new_id]._width = v._width
            spoof_cameras[new_id]._height = v._height
            bit_depth[new_id] = _spoof_bit_depth
            v._spoof_cameras.append(spoof_cameras[new_id])

    # merge in spoof cameras, should be no key collisions
    cameras = cameras | spoof_cameras
    ids = ids | {_id: "spoof" for _id in spoof_cameras.keys()}

    cameras = dict(sorted(cameras.items()))
    ids = dict(sorted(ids.items()))

    recorders = []
    write_dtype = {}

    if hw_trigger:
        if len(hw_trigger_pulse_width) == 0:
            if trigger_device == "nucleo" and hw_trigger_rate > 0:
                use_rate = hw_trigger_rate
            else:
                use_rate = np.round(
                    list(cameras.values())[0].get_feature("AcquisitionFrameRate")
                )
            use_period = 1 / use_rate
            print(f"Setting hw trigger pulse width to {use_period}")
            if trigger_device == "nucleo" and hw_trigger_pulse_low:
                if len(hw_trigger_pulse_low) != 1:
                    raise click.UsageError("Supply exposure widths when using multiple low intervals")
            else:
                hw_trigger_pulse_low = [.005]
            hw_trigger_pulse_width = [use_period - hw_trigger_pulse_low[0]]
            hw_trigger_rate = use_rate
        if trigger_device == "nucleo":
            from cammy.trigger.nucleo import NucleoTriggerDevice

            for camera_id, features in cameras_metadata.items():
                if (
                    features.get("ExposureMode") != "TriggerWidth"
                    or features.get("TriggerSelector") != "FrameStart"
                    or features.get("TriggerMode") != "On"
                    or features.get("TriggerActivation") != "RisingEdge"
                    or features.get("TriggerSource") in (None, "Software")
                ):
                    raise click.UsageError(
                        f"{camera_id}: Nucleo requires ExposureMode=TriggerWidth, "
                        "TriggerSelector=FrameStart, TriggerMode=On, "
                        "TriggerActivation=RisingEdge and a hardware "
                        "TriggerSource wired to D4 in the camera options TOML"
                    )
            if not hw_trigger_pulse_low:
                hw_trigger_pulse_low = [.002]
            try:
                trigger_dev = NucleoTriggerDevice(
                    com=trigger_port,
                    mode=light_mode,
                    pulse_widths=hw_trigger_pulse_width,
                    pulse_widths_low=hw_trigger_pulse_low,
                    surface_illumination_percent=surface_illumination_percent,
                    fluorescence_illumination_percent=fluorescence_illumination_percent,
                    surface_voltage=surface_voltage,
                    fluorescence_voltage=fluorescence_voltage,
                    light_settle_us=light_settle_us,
                )
            except ValueError as exc:
                raise click.UsageError(str(exc)) from exc
            logging.info("Nucleo: D2 surface, D3 fluorescence, D4 camera; mode=%s", light_mode)
        else:
            from cammy.trigger.trigger import TriggerDevice

            logging.info(f"Trigger pins: {trigger_pins}")
            trigger_dev = TriggerDevice(
                com=trigger_port,
                pins=trigger_pins,
                duration=duration,
                alternate_mode=alternate_mode,
                pulse_widths=hw_trigger_pulse_width,
                pulse_widths_low=hw_trigger_pulse_low,
            )
    else:
        trigger_dev = None

    zmq_addresses = {}
    if record:
        metadata_path = os.path.join(basedir, "metadata.toml")
        show_fields = toml.load(metadata_path)["show_fields"]
        init_timestamp = datetime.datetime.now()

        recording_metadata = {
            "codec": "raw",
            "start_time": init_timestamp.isoformat(),
            "cameras": ids,
            "bit_depth": bit_depth,
            "camera_metadata": cameras_metadata,
            "cli_parameters": cli_params,
        }

        write_dtype, codec = get_output_format(save_engine, bit_depth)
        recording_metadata["codec"] = codec
        recording_metadata["pixel_format"] = write_dtype

        init_timestamp_str = init_timestamp.strftime("%Y%m%d%H%M%S-%f")

        save_path = os.path.abspath(f"session_{init_timestamp_str} ({hostname})")

        dpg.create_context()

        # https://github.com/hoffstadt/DearPyGui/issues/1380
        with dpg.font_registry():
            # Download font here: https://fonts.google.com/specimen/Open+Sans
            font_path = os.path.join(
                basedir, "../assets", "OpenSans-VariableFont_wdth,wght.ttf"
            )
            default_font_large = dpg.add_font(font_path, 24 * 2, tag="ttf-font-large")
            default_font_small = dpg.add_font(font_path, 20 * 2, tag="ttf-font-small")

        settings_tags = {}
        settings_vals = {}

        with dpg.window(width=500, height=300, no_resize=True, tag="settings"):
            for k, v in show_fields.items():
                settings_tags[k] = dpg.add_input_text(default_value=v, label=k)
            dpg.add_spacer(height=5)
            dpg.add_spacing(count=5)

            def button_callback(sender, app_data):
                for k, v in settings_tags.items():
                    settings_vals[k] = dpg.get_value(v)
                dpg.stop_dearpygui()

            dpg.add_button(label="START EXPERIMENT", callback=button_callback)
            dpg.bind_font(default_font_large)
            dpg.set_global_font_scale(0.5)

        dpg.create_viewport(width=300, height=300, title="Settings")
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("settings", True)
        dpg.start_dearpygui()
        dpg.destroy_context()

        recording_metadata["user_input"] = settings_vals

        if os.path.exists(save_path):
            raise RuntimeError(f"Directory {save_path} already exists")
        else:
            # dump in metadata
            os.makedirs(save_path)
            with open(os.path.join(save_path, "metadata.toml"), "w") as f:
                toml.dump(recording_metadata, f)

        # dump settings to toml file (along with start time of recording and hostname)
        for i, (_id, _cam) in enumerate(cameras.items()):
            zmq_addresses[_id] = f"inproc://recording-{i}"
            logger.debug(f"Setting zmq endpoint for {_id} to {zmq_addresses[_id]}")
            try:
                # Keep recording transport's native I/O threads off GUI/receive cores.
                if gui_cpus is not None:
                    os.sched_setaffinity(0, set(write_cpus))
                # inproc publishers and recorders must share the same context.
                cameras[_id].zmq_context = zmq.Context.instance()
                cameras[_id].zmq_publisher = cameras[_id].zmq_context.socket(zmq.PUSH)
                cameras[_id].zmq_publisher.setsockopt(zmq.SNDHWM, record_queue_frames)
                cameras[_id].zmq_publisher.setsockopt(zmq.LINGER, 0)     # Don't block on close
                cameras[_id].zmq_publisher.bind(zmq_addresses[_id])
            finally:
                if gui_cpus is not None:
                    os.sched_setaffinity(0, gui_cpus)
        
            timestamp_fields = ["frame_id", "device_timestamp", "system_timestamp"]
            if save_engine == "ffmpeg":
                _recorder = FfmpegVideoRecorder(
                    width=cameras[_id]._width,
                    height=cameras[_id]._height,
                    filename=os.path.join(save_path, f"{_id}.mkv"),
                    pixel_format=write_dtype[_id],
                    timestamp_fields=timestamp_fields,
                    batch_size=frame_writer_batch_size,
                )
            elif save_engine == "raw":
                _recorder = RawVideoRecorder(
                    filename=os.path.join(save_path, f"{_id}.dat"),
                    write_dtype=write_dtype[_id],
                    timestamp_fields=timestamp_fields,
                    batch_size=frame_writer_batch_size,
                    cpu_id=write_cpus[i % len(write_cpus)],
                    zmq_address=zmq_addresses[_id],
                    queue_frames=record_queue_frames,
                )
            else:
                raise RuntimeError(
                    f"Did not understanding VideoRecorder option {save_engine}"
                )

            _recorder.daemon = False
            _cam.recorder = _recorder
            recorders.append(_recorder)
    else:
        show_fields = {}
        save_path = None
        recording_metadata = None

    if server and (zsocket is not None):
        #  wait for ready signal
        # msg = zsocket.recv_pyobj()
        # communicate save path (by now it's been created)
        logger.info("Sending save path to client...")
        zsocket.send_pyobj(save_path)
        logger.info("Done")

    # start a new context for acquisition
    dpg.create_context()

    # https://github.com/hoffstadt/DearPyGui/issues/1380
    with dpg.font_registry():
        # Download font here: https://fonts.google.com/specimen/Open+Sans
        font_path = os.path.join(
            basedir, "../assets", "OpenSans-VariableFont_wdth,wght.ttf"
        )
        default_font_large = dpg.add_font(font_path, 20 * 2, tag="ttf-font-large")
        default_font_small = dpg.add_font(font_path, 16 * 2, tag="ttf-font-small")

    with dpg.texture_registry(show=False):
        for _id, _cam in cameras.items():
            blank_data = np.zeros(
                (
                    _cam._height // display_downsample,
                    _cam._width // display_downsample,
                    4,
                ),
                dtype="float32",
            )
            dpg.add_raw_texture(
                _cam._width / display_downsample,
                _cam._height / display_downsample,
                blank_data,
                tag=f"texture_{_id}",
                format=dpg.mvFormat_Float_rgba,
            )

    miss_status = {}
    fps_status = {}
    for _id, _cam in cameras.items():
        use_config = {}
        for k, v in camera_dct["display"].items():
            if k in _id:
                use_config = v

        with dpg.window(
            label=f"Camera {_id}",
            tag=f"Camera {_id}",
            no_collapse=True,
            no_scrollbar=True,
        ):
            dpg.add_image(f"texture_{_id}")
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    tag=f"texture_{_id}_min",
                    width=(_cam._width // display_downsample) / 3,
                    **{**slider_defaults_min, **use_config["slider_defaults_min"]},
                )
                dpg.add_slider_float(
                    tag=f"texture_{_id}_max",
                    width=(_cam._width // display_downsample) / 3,
                    **{**slider_defaults_max, **use_config["slider_defaults_max"]},
                )
            miss_status[_id] = dpg.add_text("0 missed frames / 0 total")
            fps_status[_id] = dpg.add_text(f"0 {fps_unit}")
            # add sliders/text boxes for exposure time and fps
            dpg.bind_font(default_font_small)
            dpg.set_global_font_scale(0.5)

    gui_x_offset = 0
    gui_y_offset = 0
    gui_x_max = 0
    gui_y_max = 0
    row_pos = 0
    for _id, _cam in cameras.items():
        cur_key = f"Camera {_id}"
        dpg.set_item_pos(cur_key, (gui_x_offset, gui_y_offset))

        width = _cam._width // display_downsample + 25
        height = _cam._height // display_downsample + 100

        gui_x_max = int(np.maximum(gui_x_offset + width, gui_x_max))
        gui_y_max = int(np.maximum(gui_y_offset + height, gui_y_max))

        row_pos += 1
        if row_pos == gui_ncols:
            row_pos = 0
            gui_x_offset = 0
            gui_y_offset += height
        else:
            gui_x_offset += width

    if server and (zsocket is not None):
        start_signal = zsocket.recv_pyobj()
        if start_signal == "START":
            logging.info("Sleeping for 5 seconds before starting acquisition...")
            time.sleep(5)  # allow fudge factor for other program to start
        else:
            raise RuntimeError(f"Did not understand signal {start_signal}")

    threads = {}
    acquisition_thread_shutdown_event = threading.Event()
    recording_errors = []
    try:
        # Configure the Nucleo while stopped, before arming any camera.
        if hw_trigger and trigger_device == "nucleo":
            trigger_dev.open()
        for _recorder in recorders:
            _recorder.start()
        for _recorder in recorders:
            _recorder.ready.wait()
            if _recorder.error is not None:
                raise click.ClickException(f"Could not open recording {_recorder.filename}: {_recorder.error}")
        [_cam.start_acquisition() for _cam in cameras.values()]

        for i, (_id, _cam) in enumerate(cameras.items()):
            t = threading.Thread(
                target=acquisition_loop,
                args=(_cam, acquisition_thread_shutdown_event, acquire_cpus[i % len(acquire_cpus)]),
            )
            t.daemon = True
            t.display_frame = (None, None)
            threads[_id] = t

        for _cam in cameras.values():
            _cam.count = 0

        dpg.create_viewport(title="Camera preview", width=gui_x_max, height=gui_y_max)

        # dpg.set_viewport_vsync(False)
        # dpg.show_metrics()
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # 3/7/23 REMOVED EXTRA START_ACQUISITION, PUT GPIO IN WEIRD STATE
        # [print(_cam.camera.get_trigger_source()) for _cam in cameras.values()]

        # push frame grabbing to separate thread...
        # https://github.com/AravisProject/aravis/issues/754

        start_time = -np.inf
        prior_fps = np.nan
        cur_duration = 0
        display_counter = {}
        last_seen_frame = {}
        next_stream_stats = 0.0

        for _camera in cameras.keys():
            display_counter[_camera] = 0
        [t.start() for t in threads.values()]
        # Cameras and acquisition threads are ready before external timing starts.
        if hw_trigger and (trigger_dev is not None):
            trigger_dev.start()
        
        while dpg.is_dearpygui_running():
            if stream_stats and time.perf_counter() >= next_stream_stats:
                for _cam in cameras.values():
                    if isinstance(_cam, AravisCamera):
                        _cam.log_stream_statistics()
                next_stream_stats = time.perf_counter() + 1.0
            dat = {}
            for _id, _cam in cameras.items():
                # with _cam.display_lock:
                dat[_id] = _cam.display_frame
                    # print(_thread.display_frame)
            
            for _id, _dat in dat.items():
                if ~np.isfinite(start_time) and (_dat[0] is not None):
                    start_time = time.perf_counter()

            cur_duration = (time.perf_counter() - start_time) / 60.0
            for _id, _dat in dat.items():
                if _dat[0] is not None and _dat is not last_seen_frame.get(_id):
                    last_seen_frame[_id] = _dat
                    disp_min = dpg.get_value(f"texture_{_id}_min")
                    disp_max = dpg.get_value(f"texture_{_id}_max")
                    if (display_time_downsample > 0)  and ((display_counter[_id] % display_time_downsample) == 0):
                        height, width = _dat[0].shape
                        disp_img = cv2.resize(
                            _dat[0],
                            (width // display_downsample, height // display_downsample),
                        )
                        plt_val = intensity_to_rgba_lut(
                            disp_img,
                            minval=disp_min,
                            maxval=disp_max,
                            colormap=display_colormap,
                        )
                        cv2.putText(
                            plt_val,
                            str(cameras[_id].frame_count),
                            txt_pos,
                            font,
                            1,
                            (1, 1, 1, 1),
                        )
                        dpg.set_value(f"texture_{cameras[_id].id}", plt_val)
                    else:
                        pass

                    display_counter[_id] += 1
                    cameras[_id].count += 1
                    miss_frames = cameras[_id].missed_frames
                    total_frames = cameras[_id].total_frames
                    cur_fps = cameras[_id].fps
                    # if np.isnan(prior_fps):
                    #     smooth_fps = cur_fps
                    # else:
                    #     smooth_fps = .01 * cur_fps + .99 * prior_fps
                    # prior_fps = smooth_fps
                    try:
                        percent_missed = (miss_frames / total_frames) * 100
                    except ZeroDivisionError:
                        percent_missed = 0
                    miss_text = f"{miss_frames} missed / {total_frames} total ({percent_missed:.1f}% missed)"
                    if record:
                        writer = cameras[_id].recorder
                        recording_drops = cameras[_id].recording_drops
                        recording_total = getattr(writer, "frames_written", 0) + recording_drops
                        recording_percent_missed = 100 * recording_drops / recording_total if recording_total else 0
                        miss_text += (
                            f"\nWriter: {recording_drops} missed / {recording_total} total "
                            f"({recording_percent_missed:.1f}% missed)"
                        )
                        if not writer.is_alive():
                            miss_text += " / WRITER STOPPED (see terminal)"
                    dpg.set_value(miss_status[_id], miss_text)
                    if cur_fps is not None:
                        dpg.set_value(
                            fps_status[_id],
                            f"{cur_fps:.0f} {fps_unit}",
                        )
                        
            if (
                np.isfinite(cur_duration)
                and (duration > 0)
                and (cur_duration > duration)
            ):
                logging.info(f"Exceeded {duration} minutes, exiting...")
                break
            if server and (zsocket is not None):
                try:
                    dat = zsocket.recv_pyobj(flags=zmq.NOBLOCK)
                    if dat == "EXIT":
                        logger.info("Received stop signal, exiting...")
                        break
                except zmq.Again:
                    pass
            # time.sleep(0.005)
            dpg.render_dearpygui_frame()
    finally:
        try:
            # Stop producers before asking writers to drain their inproc queues.
            if hw_trigger and (trigger_dev is not None):
                try:
                    if trigger_device == "nucleo":
                        trigger_dev.close()
                    else:
                        trigger_dev.stop()
                except Exception:
                    logger.exception("Could not stop the trigger controller")
            for _cam in cameras.values():
                try:
                    _cam.stop_acquisition()
                except Exception:
                    logger.exception("Could not stop camera %s", _cam.id)
            acquisition_thread_shutdown_event.set()
            [_t.join() for _t in threads.values() if _t.ident is not None]
            if server and (zsocket is not None):
                try:
                    zsocket.send_pyobj("EXIT", flags=zmq.NOBLOCK)
                except zmq.ZMQError:
                    logger.exception("Could not notify the client of shutdown")
            for _recorder in recorders:
                _recorder.stop()
            if recorders:
                logger.info("Draining recording queues and syncing files; waiting for writers...")
            for _recorder in recorders:
                if _recorder.ident is not None:
                    _recorder.join()
                if _recorder.error is not None:
                    recording_errors.append(_recorder.filename)
                    logger.error("Recording failed for %s: %s", _recorder.filename, _recorder.error)
            for _id, _cam in cameras.items():
                publisher = getattr(_cam, "zmq_publisher", None)
                if publisher is not None:
                    publisher.close(linger=0)
                if record:
                    logger.info("%s: %d frames written, %d recording queue drops", _id,
                                _cam.recorder.frames_written, getattr(_cam, "recording_drops", 0))
        finally:
            dpg.destroy_context()
    if recording_errors:
        raise click.ClickException("Recording failed: " + ", ".join(recording_errors))


@cli.command(name="save-intrinsics")
@click.argument("filename", type=click.Path(exists=False))
@click.option(
    "--interface", type=click.Choice(["aravis", "fake_custom", "all"]), default="all"
)
def save_intrinsics(
    filename: str,
    interface: str,
):
    ids = get_all_camera_ids(interface)
    cameras = initialize_cameras(ids, configs={})

    intrinsics = {}

    param_names = [
        "CalibFocalLengthX",
        "CalibFocalLengthY",
        "CalibOpticalCenterX",
        "CalibOpticalCenterY",
    ]

    distortion_vals = ["k1", "k2", "p1", "p2", "k3"]

    for k, v in cameras.items():
        intrinsics[k] = {}
        for _param in param_names:
            intrinsics[k][_param] = v.get_feature(_param)
        for i, _name in enumerate(distortion_vals):
            v.set_feature("CalibLensDistortionValueSelector", f"Value{i}")
            intrinsics[k][_name] = v.get_feature("CalibLensDistortionValue")

    with open(filename, "w") as f:
        toml.dump(intrinsics, f)


@cli.command(name="calibrate")
@click.argument("camera_options_file", type=click.Path(exists=True))
@click.option("--intrinsics-file", type=click.Path(exists=True), default=None)
@click.option(
    "--interface", type=click.Choice(["aravis", "fake_custom", "all"]), default="all"
)
@click.option("--display-colormap", type=str, default="gray")
@click.option("--record", is_flag=True, help="Save output to disk")
@click.option(
    "--detect-threshold-image", is_flag=True, help="Threshold image prior to detection"
)
@click.option(
    "--light-control",
    type=int,
    default=-1,
    help="Turn on specific light bank with arduino (<0 to skip)",
)
@click.option(
    "--display-downsample",
    type=int,
    default=1,
    help="Downsample images for display",
)
def calibrate(
    camera_options_file: str,
    intrinsics_file: str,
    interface: str,
    display_colormap: Optional[str],
    record: bool,
    detect_threshold_image: bool,
    light_control: int,
    display_downsample: int
):
    import cv2
    import socket
    import datetime
    import pickle
    import dearpygui.dearpygui as dpg
    from cammy.util import intrinsics_file_to_cv2
    from cammy.calibrate import (
        initialize_boards,
        estimate_pose,
        detect_charuco,
        threshold_image,
    )

    hostname = socket.gethostname()
    init_timestamp = datetime.datetime.now()
    init_timestamp_str = init_timestamp.strftime("%Y%m%d%H%M%S-%f")
    save_path = os.path.abspath(
        f"session_{init_timestamp_str} ({hostname}, calibration)"
    )

    if light_control >= 0:
        # logging.info(f"Trigger pins: {trigger_pins}")
        from cammy.trigger.trigger import TriggerDevice

        trigger_dev = TriggerDevice(
            pins=[], alternate_mode=light_control
        )
        trigger_dev.start()
    else:
        trigger_dev = None

    if intrinsics_file is not None:
        intrinsic_matrix, distortion_coeffs = intrinsics_file_to_cv2(intrinsics_file)
    else:
        intrinsic_matrix = None
        distortion_coeffs = None

    # INIT CHARUCO PARAMETERS
    if display_colormap is None:
        display_colormap = mpl_to_cv2_colormap(colormap_default)
    else:
        display_colormap = mpl_to_cv2_colormap(display_colormap)

    logging.info(f"Loading camera options from {camera_options_file}")
    camera_dct = toml.load(camera_options_file)

    # TODO: add multiple boards here, detect in loop, keep board with
    # largest number of markers...
    boards = initialize_boards(
        squares=camera_dct["charuco"]["squares"],
        marker_length=camera_dct["charuco"]["marker_length_mm"],
        square_length=camera_dct["charuco"]["square_length_mm"],
        num_slices=camera_dct["charuco"]["num_slices"],
        markers_per_slice=camera_dct["charuco"]["markers_per_slice"],
        ar_dict=camera_dct["charuco"]["aruco_dictionary"],
    )
    ids = get_all_camera_ids(interface)
    cameras = initialize_cameras(ids, configs=camera_dct)
    [_camera.initialize_acquisition_stream() for _camera in cameras.values()]

    metadata = {"calibration": {}}
    metadata["calibration"]["session_time"] = init_timestamp_str
    metadata["calibration"]["cameras"] = list(ids.keys())
    metadata["calibration"]["camera_internal_data"] = {}
    metadata["calibration"]["camera_internal_data"][
        "intrinsic_matrix"
    ] = intrinsic_matrix
    metadata["calibration"]["camera_internal_data"][
        "distortion_coeffs"
    ] = distortion_coeffs
    metadata["calibration"]["board"] = camera_dct["charuco"]

    dpg.create_context()

    bit_depth = {}
    disp_mins = {}
    disp_maxs = {}
    for k, v in cameras.items():
        feature_dct = v.get_all_features()
        feature_dct = dict(sorted(feature_dct.items()))
        _bit_depth, _spoof_ims = get_pixel_format_bit_depth(feature_dct["PixelFormat"])
        # v._pixel_format = feature_dct["PixelFormat"]
        bit_depth[k] = _bit_depth
        disp_mins[k] = 0
        disp_maxs[k] = 2**_bit_depth

    # append everything lists and dump to pickle
    with dpg.texture_registry(show=False):
        for _id, _cam in cameras.items():
            blank_data = np.zeros(
                (_cam._height, _cam._width, 3),
                dtype="float32",
            )
            dpg.add_raw_texture(
                _cam._width // display_downsample,
                _cam._height // display_downsample,
                blank_data,
                tag=f"texture_{_id}",
                format=dpg.mvFormat_Float_rgb,
            )

    for _id, _cam in cameras.items():
        use_config = {}
        for k, v in camera_dct["display"].items():
            if k in _id:
                use_config = v

        with dpg.window(
            label=f"Camera {_id}",
            tag=f"Camera {_id}",
            no_collapse=True,
            no_scrollbar=True,
        ):
            dpg.add_image(f"texture_{_id}")

    gui_x_offset = 0
    gui_y_offset = 0
    gui_x_max = 0
    gui_y_max = 0
    row_pos = 0
    for _id, _cam in cameras.items():
        cur_key = f"Camera {_id}"
        dpg.set_item_pos(cur_key, (gui_x_offset, gui_y_offset))

        width = _cam._width // display_downsample + 25
        height = _cam._height // display_downsample + 100

        gui_x_max = int(np.maximum(gui_x_offset + width, gui_x_max))
        gui_y_max = int(np.maximum(gui_y_offset + height, gui_y_max))

        row_pos += 1
        if row_pos == gui_ncols:
            row_pos = 0
            gui_x_offset = 0
            gui_y_offset += height
        else:
            gui_x_offset += width

    [_cam.start_acquisition() for _cam in cameras.values()]
    time.sleep(1)
    dpg.create_viewport(title="Camera preview", width=gui_x_max, height=gui_y_max)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # save everything in a pickle file...
    aruco_save_data = {
        _cam: {"corners": [], "ids": [], "board_idx": []} for _cam in cameras.keys()
    }
    charuco_save_data = {
        _cam: {"corners": [], "ids": [], "board_idx": []} for _cam in cameras.keys()
    }
    pose_save_data = {
        _cam: {"pose": [], "rvec": [], "tvec": []} for _cam in cameras.keys()
    }
    img_save_data = {_cam: [] for _cam in cameras.keys()}

    # https://stackoverflow.com/questions/13180941/how-to-kill-a-while-loop-with-a-keystroke
    import _thread

    def input_thread(a_list):
        input("Press enter to grab a new frame")
        a_list.append(True)

    frame_count = {}
    for _cam in cameras.keys():
        frame_count[_cam] = 0
    try:
        while dpg.is_dearpygui_running():
            a_list = []
            _thread.start_new_thread(input_thread, (a_list,))
            while not a_list:
                dpg.render_dearpygui_frame()

            [_cam.camera.software_trigger() for _cam in cameras.values()]
            time.sleep(0.1)
            dat = {}
            for _id, _cam in cameras.items():
                new_frame = None
                new_ts = None
                while new_frame is None:
                    _dat = _cam.try_pop_frame()
                    if _dat[0] is not None:
                        new_frame = _dat[0]
                        new_ts = _dat[1]
                dat[_id] = (new_frame, new_ts)

            for _id, _dat in dat.items():
                # import matplotlib.pyplot as plt
                # plt.imshow(_dat[0])
                # plt.show()
                height, width = _dat[0].shape
                plt_val = cv2.merge([_dat[0]] * 3)
                plt_val = plt_val.astype("uint8")
                # use_img = threshold_image(_dat[0].copy())
                proc_img = _dat[0].copy()
                # proc_img = cv2.normalize(_dat[0], None, 0, 255, cv2.NORM_MINMAX)
                # proc_img = cv2.equalizeHist(proc_img.astype("uint8"))
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                proc_img = clahe.apply(proc_img)
                # print(threshold_image)
                if detect_threshold_image:
                    print("Thresholding image...")
                    proc_img = threshold_image(proc_img)
                # smooth = cv2.GaussianBlur(proc_img, (95, 95), 0)
                # proc_img = cv2.divide(proc_img, smooth, scale=50)

                # TODO: add support for multiple boards here...
                # detect board with largest number of markers in FOV...
                aruco_dat, charuco_dat = detect_charuco(proc_img, boards)

                for _board_id, (_aruco_dat, _charuco_dat) in enumerate(
                    zip(aruco_dat, charuco_dat)
                ):
                    # add if we get more than three detections
                    if len(_aruco_dat[0]) > 3:
                        aruco_save_data[_id]["corners"].append(_aruco_dat[0])
                        aruco_save_data[_id]["ids"].append(_aruco_dat[1])
                        aruco_save_data[_id]["board_idx"].append(_board_id)
                        charuco_save_data[_id]["corners"].append(_charuco_dat[0])
                        charuco_save_data[_id]["ids"].append(_charuco_dat[1])
                        charuco_save_data[_id]["board_idx"].append(_board_id)
                        img_save_data[_id] += _dat

                        # draw results
                        plt_val = cv2.aruco.drawDetectedMarkers(
                            plt_val, *_aruco_dat, [0, 255, 255]
                        )
                        plt_val = cv2.aruco.drawDetectedCornersCharuco(
                            plt_val, *_charuco_dat, [255, 0, 0, 0]
                        )

                plt_val = cv2.putText(
                    plt_val, str(frame_count[_id]), txt_pos, font, 1, (255, 255, 255)
                )

                frame_count[_id] += 1
                # SKIP pose if we do not have intrinsic and distortion estimates.
                # TODO: add corner subpixel refinement...
                # if intrinsic_matrix is not None:
                #     pose, rvec, tvec = estimate_pose(
                #         *charuco_dat,
                #         intrinsic_matrix[_id],
                #         distortion_coeffs[_id],
                #         board,
                #     )

                #     pose_save_data[_id]["pose"].append(pose)
                #     pose_save_data[_id]["rvec"].append(rvec)
                #     pose_save_data[_id]["tvec"].append(tvec)
                #     plt_val = cv2.drawFrameAxes(
                #         plt_val, intrinsic_matrix[_id], distortion_coeffs[_id], rvec, tvec, 0.05
                #     )

                # convert to [0,1] float for dpg
                plt_val = plt_val.astype("float32") / 255.0
                if display_downsample > 1:
                    plt_val = cv2.resize(
                                plt_val,
                                (width // display_downsample, height // display_downsample),
                            )
                dpg.set_value(f"texture_{cameras[_id].id}", plt_val)

            dpg.render_dearpygui_frame()

    except (KeyboardInterrupt, EOFError):
        # save the data
        if record:
            logging.info("Saving data...")
            os.makedirs(save_path)
            save_dictionary = {}
            save_dictionary["pose"] = pose_save_data
            save_dictionary["img"] = img_save_data
            save_dictionary["charuco"] = charuco_save_data
            save_dictionary["aruco"] = aruco_save_data
            save_dictionary["camera_options"] = camera_dct
            with open(os.path.join(save_path, "calibration.pkl"), "wb") as f:
                pickle.dump(save_dictionary, f)
            with open(os.path.join(save_path, "metadata.toml"), "w") as f:
                toml.dump(metadata, f)
        [_cam.stop_acquisition() for _cam in cameras.values()]
        if trigger_dev is not None:
            trigger_dev.stop()
        logging.info("Done...")
        dpg.destroy_context()


if __name__ == "__main__":
    cli()
