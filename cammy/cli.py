import click
import numpy as np
import toml
import logging
import sys
import os
import time
import cv2

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s]:%(levelname)s:%(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


from typing import Optional, Iterable
from cammy.util import (
    get_all_camera_ids,
    intensity_to_rgba,
    get_queues,
    initialize_cameras,
    get_output_format,
    get_pixel_format_bit_depth,
    mpl_to_cv2_colormap,
    check_counters_equal,
)
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
@click.option("--jumbo-frames", default=True, type=bool, help="Turn on jumbo frames (GigE only)")
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
    default=.002,
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
    help="Alternate mode (0 for first set of lets constant, 1 for other, 2 for alternation)"
)
@click.option(
    "--prefix",
    type=str,
    default=None,
    help="Only uses cameras whose IDs start with this string (None to use all IDs)"
)
# fmt: on
def simple_preview(
    interface: str,
    buffer_size: int,
    n_fake_cameras: int,
    camera_options: Optional[str],
    record: bool,
    jumbo_frames: bool,
    save_engine: str,
    display_downsample: int,
    display_colormap: Optional[str],
    hw_trigger: bool,
    hw_trigger_rate: float,
    hw_trigger_pin_last: int,
    hw_trigger_pulse_width: Iterable[float],
    hw_trigger_pulse_low: float,
    record_counters: int,
    duration: float,
    server: bool,
    alternate_mode: int,
    prefix: Optional[str],
):
    
    cli_params = locals()

    import dearpygui.dearpygui as dpg
    import cv2
    import socket
    import datetime
    import zmq

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

    # TODO: TURN INTO AN AUTOMATIC CHECK, IF NO FRAMES ARE GETTING
    # ACQUIRED, PAUSE FOR 1 SEC AND RE-INITIALIZE
    cameras = initialize_cameras(
        ids,
        camera_dct,
        jumbo_frames=jumbo_frames,
        record_counters=record_counters,
        buffer_size=buffer_size,
    )
    del cameras
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
    )
    for i, (k, v) in enumerate(cameras.items()):
        feature_dct = v.get_all_features()
        feature_dct = dict(sorted(feature_dct.items()))

        # make sure we set so we know how to decode frame buffers
        v._pixel_format = feature_dct["PixelFormat"]
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
        logging.info(f"Trigger pins: {trigger_pins}")
        from cammy.trigger.trigger import TriggerDevice

        # if rate < 0, set to AcquisitionFrameRate of first cam
        # if hw_trigger_rate <= 0:
        if len(hw_trigger_pulse_width) == 0:
            use_rate = np.round(
                list(cameras.values())[0].get_feature("AcquisitionFrameRate")
            )
            use_period = 1 / use_rate
            print(f"Setting hw trigger pulse width to {use_period}")
            hw_trigger_pulse_width = [use_period]
            hw_trigger_rate = use_rate
        trigger_dev = TriggerDevice(
            # frame_rate=hw_trigger_rate,
            pins=trigger_pins,
            duration=duration,
            alternate_mode=alternate_mode,
            pulse_widths=hw_trigger_pulse_width,
            pulse_width_low=hw_trigger_pulse_low,
        )
    else:
        trigger_dev = None

    if record:
        # from parameters construct single names...
        use_queues = get_queues(list(ids.keys()))
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
        for _id, _cam in cameras.items():
            cameras[_id].save_queue = use_queues["storage"][_id]
            timestamp_fields = ["frame_id", "device_timestamp", "system_timestamp"]
            if save_engine == "ffmpeg":
                _recorder = FfmpegVideoRecorder(
                    width=cameras[_id]._width,
                    height=cameras[_id]._height,
                    save_queue=cameras[_id].save_queue,
                    filename=os.path.join(save_path, f"{_id}.mkv"),
                    pixel_format=write_dtype[_id],
                    timestamp_fields=timestamp_fields,
                )
            elif save_engine == "raw":
                _recorder = RawVideoRecorder(
                    save_queue=cameras[_id].save_queue,
                    filename=os.path.join(save_path, f"{_id}.dat"),
                    write_dtype=write_dtype[_id],
                    timestamp_fields=timestamp_fields,
                )
            else:
                raise RuntimeError(
                    f"Did not understanding VideoRecorder option {save_engine}"
                )

            _recorder.daemon = True
            _recorder.start()
            recorders.append(_recorder)
    else:
        show_fields = {}
        use_queues = {}
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
            fps_status[_id] = dpg.add_text("0 FPS")
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

    [_cam.start_acquisition() for _cam in cameras.values()]

    # if using a hardware trigger, send out signals now...
    if hw_trigger and (trigger_dev is not None):
        # trigger_armed = np.array([_cam.get_feature("TriggerArmed") for _cam in cameras.values()])
        # while ~np.all(trigger_armed):
        #     time.sleep(.5)
        #     trigger_armed = np.array([_cam.get_feature("TriggerArmed") for _cam in cameras.values()])
        #     logging.info("Waiting for trigger armed signal on all cameras...")
        # logging.info("Starting Arduino...")
        trigger_dev.start()

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
    try:
        while dpg.is_dearpygui_running():
            dat = {}
            for _id, _cam in cameras.items():
                new_frame = None
                new_ts = None

                # do we need a separate thread for this, then grab whatever frame is latest???
                while True:
                    _dat = _cam.try_pop_frame()
                    if _dat[0] is None:
                        break
                    else:
                        if ~np.isfinite(start_time):
                            start_time = time.perf_counter()
                        new_frame = _dat[0]
                        new_ts = _dat[1]
                dat[_id] = (new_frame, new_ts)

            cur_duration = (time.perf_counter() - start_time) / 60.0
            for _id, _dat in dat.items():
                if _dat[0] is not None:
                    disp_min = dpg.get_value(f"texture_{_id}_min")
                    disp_max = dpg.get_value(f"texture_{_id}_max")
                    height, width = _dat[0].shape
                    disp_img = cv2.resize(
                        _dat[0],
                        (width // display_downsample, height // display_downsample),
                    )
                    plt_val = intensity_to_rgba(
                        disp_img,
                        minval=disp_min,
                        maxval=disp_max,
                        colormap=display_colormap,
                    ).astype("float32")
                    cv2.putText(
                        plt_val,
                        str(cameras[_id].frame_count),
                        txt_pos,
                        font,
                        1,
                        (1, 1, 1, 1),
                    )
                    dpg.set_value(f"texture_{cameras[_id].id}", plt_val)
                    cameras[_id].count += 1
                    miss_frames = float(cameras[_id].missed_frames)
                    total_frames = float(cameras[_id].total_frames)
                    cur_fps = cameras[_id].fps
                    # if np.isnan(prior_fps):
                    #     smooth_fps = cur_fps
                    # else:
                    #     smooth_fps = .01 * cur_fps + .99 * prior_fps
                    # prior_fps = smooth_fps
                    percent_missed = (miss_frames / total_frames) * 100
                    dpg.set_value(
                        miss_status[_id],
                        f"{miss_frames} missed / {total_frames} total ({percent_missed:.1f}% missed)",
                    )
                    if cur_fps is not None:
                        dpg.set_value(
                            fps_status[_id],
                            f"{cur_fps:.0f} FPS",
                        )
                    if "storage" in use_queues.keys():
                        for k, v in use_queues["storage"].items():
                            logging.debug(v.qsize())

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
        [_cam.stop_acquisition() for _cam in cameras.values()]
        if hw_trigger and (trigger_dev is not None):
            trigger_dev.stop()
        if server and (zsocket is not None):
            # don't wait for the client, just bail (we want this to exit first)
            logger.info("Sending STOP to client...")
            zsocket.send_pyobj("EXIT", flags=zmq.NOBLOCK)
        if record:
            # for every camera ID wait until the queue has been written out
            print("Issuing stop signal...")
            for k, v in use_queues["storage"].items():
                v.put(None)  # stop signal
                time.sleep(0.1)
                if v.qsize() is not None:
                    while v.qsize() > 0:
                        time.sleep(0.1)
            for _recorder in recorders:
                _recorder.is_running = 0
                time.sleep(1)
        # ensure all files are flushed and closed...
        [_recorder.close_writer() for _recorder in recorders]
        dpg.destroy_context()


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
def calibrate(
    camera_options_file: str,
    intrinsics_file: str,
    interface: str,
    display_colormap: Optional[str],
    record: bool,
    detect_threshold_image: bool,
    light_control: int,
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
            frame_rate=-1, pins=[], alternate_mode=light_control
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
        v._pixel_format = feature_dct["PixelFormat"]
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
                _cam._width,
                _cam._height,
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

        width = _cam._width + 25
        height = _cam._height + 100

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
