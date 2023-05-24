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
gui_ncols = 2  # number of cols before we start new row
# for labeling videos
font = cv2.FONT_HERSHEY_SIMPLEX
white = (255, 255, 255)
txt_pos = (25, 25)


# TODO:
# 1) ADD OPTION TO READ COUNTERS WITH COLUMN NAME?
# 2) PTPENABLE FOR TIME CLOCK SYNC?
@cli.command(name="run")
@click.option("--interface", type=click.Choice(["aravis", "fake_custom", "all"]), default="all")
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
@click.option("--hw-trigger-rate", type=float, default=100.0, help="Trigger rate")
@click.option(
    "--hw-trigger-pin-last", type=int, default=13, help="Final dig out pin to use on Arduino"
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
def simple_preview(
    interface: str,
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
    record_counters: int,
    duration: float,
    server: bool,
):
    cli_params = locals()

    import dearpygui.dearpygui as dpg
    import cv2
    import socket
    import datetime
    import zmq

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
    ids = get_all_camera_ids(interface, n_cams=n_fake_cameras)

    # TODO: TURN INTO AN AUTOMATIC CHECK, IF NO FRAMES ARE GETTING
    # ACQUIRED, PAUSE FOR 1 SEC AND RE-INITIALIZE
    cameras = initialize_cameras(
        ids, camera_dct, jumbo_frames=jumbo_frames, record_counters=record_counters
    )
    del cameras
    time.sleep(2)

    cameras_metadata = {}
    bit_depth = {}
    spoof_cameras = {}
    trigger_pins = []
    import copy

    cameras = initialize_cameras(
        ids, camera_dct, jumbo_frames=jumbo_frames, record_counters=record_counters
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

    dpg.create_context()
    recorders = []
    write_dtype = {}

    if hw_trigger:
        logging.info(f"Trigger pins: {trigger_pins}")
        from cammy.trigger.trigger import TriggerDevice

        trigger_dev = TriggerDevice(
            frame_rate=hw_trigger_rate, pins=trigger_pins, duration=duration
        )
    else:
        trigger_dev = None

    if record:
        # from parameters construct single names...
        use_queues = get_queues(list(ids.keys()))
        basedir = os.path.dirname(os.path.abspath(__file__))
        metadata_path = os.path.join(basedir, "metadata.toml")
        show_fields = toml.load(metadata_path)["show_fields"]
        init_timestamp = datetime.datetime.now()

        recording_metadata = {
            "codec": "raw",
            "start_time": init_timestamp.isoformat(),
            "cameras": ids,
            "bit_depth": bit_depth,
            "camera_metadata": cameras_metadata,
        }

        write_dtype, codec = get_output_format(save_engine, bit_depth)
        recording_metadata["codec"] = codec
        recording_metadata["pixel_format"] = write_dtype

        init_timestamp_str = init_timestamp.strftime("%Y%m%d%H%M%S-%f")

        save_path = os.path.abspath(f"session_{init_timestamp_str} ({hostname})")

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
        # start a new context for acquisition
        dpg.create_context()

        # dump settings to toml file (along with start time of recording and hostname)
        for _id, _cam in cameras.items():
            cameras[_id].save_queue = use_queues["storage"][_id]
            timestamp_fields = ["capture_number", "device_timestamp", "system_timestamp"]
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
                raise RuntimeError(f"Did not understanding VideoRecorder option {save_engine}")

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

    with dpg.texture_registry(show=False):
        for _id, _cam in cameras.items():
            blank_data = np.zeros(
                (_cam._height // display_downsample, _cam._width // display_downsample, 4),
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
            label=f"Camera {_id}", tag=f"Camera {_id}", no_collapse=True, no_scrollbar=True
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
    start_time = -np.inf
    prior_fps = np.nan
    cur_duration = 0
    try:
        while dpg.is_dearpygui_running():
            dat = {}
            for _id, _cam in cameras.items():
                new_frame = None
                new_ts = None
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
                        _dat[0], (width // display_downsample, height // display_downsample)
                    )
                    plt_val = intensity_to_rgba(
                        disp_img, minval=disp_min, maxval=disp_max, colormap=display_colormap
                    ).astype("float32")
                    cv2.putText(
                        plt_val, str(cameras[_id].frame_count), txt_pos, font, 1, (1, 1, 1, 1)
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

            if np.isfinite(cur_duration) and (duration > 0) and (cur_duration > duration):
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
            time.sleep(0.005)
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
@click.option("--interface", type=click.Choice(["aravis", "fake_custom", "all"]), default="all")
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
@click.argument("intrinsics_file", type=click.Path(exists=True))
@click.argument("camera_options_file", type=click.Path(exists=True))
@click.option("--interface", type=click.Choice(["aravis", "fake_custom", "all"]), default="all")
@click.option("--display-colormap", type=str, default="gray")
@click.option("--record", is_flag=True, help="Save output to disk")
def calibrate(
    intrinsics_file: str,
    camera_options_file: str,
    interface: str,
    display_colormap: Optional[str],
    record: bool,
):
    import cv2
    import socket
    import datetime
    import pickle
    import dearpygui.dearpygui as dpg
    from cammy.util import intrinsics_file_to_cv2
    from cammy.calibrate import initialize_board, estimate_pose, detect_charuco, threshold_image

    hostname = socket.gethostname()
    init_timestamp = datetime.datetime.now()
    init_timestamp_str = init_timestamp.strftime("%Y%m%d%H%M%S-%f")
    save_path = os.path.abspath(f"session_{init_timestamp_str} ({hostname}, calibration)")

    intrinsic_matrix, distortion_coeffs = intrinsics_file_to_cv2(intrinsics_file)
    
    # INIT CHARUCO PARAMETERS
    if display_colormap is None:
        display_colormap = mpl_to_cv2_colormap(colormap_default)
    else:
        display_colormap = mpl_to_cv2_colormap(display_colormap)

    logging.info(f"Loading camera options from {camera_options_file}")
    camera_dct = toml.load(camera_options_file)
    board = initialize_board(**camera_dct["charuco"])

    

    ids = get_all_camera_ids(interface)
    cameras = initialize_cameras(ids, configs=camera_dct)

    metadata = {"calibration": {}}
    metadata["calibration"]["session_time"] = init_timestamp_str
    metadata["calibration"]["cameras"] = list(ids.keys())
    metadata["calibration"]["camera_internal_data"] = {}
    metadata["calibration"]["camera_internal_data"]["intrinsic_matrix"] = intrinsic_matrix
    metadata["calibration"]["camera_internal_data"]["distortion_coeffs"] = distortion_coeffs
    metadata["calibration"]["board"] = camera_dct["charuco"]

    dpg.create_context()

    bit_depth = {}
    disp_mins = {}
    disp_maxs = {}
    for k, v  in cameras.items():
        feature_dct = v.get_all_features()
        feature_dct = dict(sorted(feature_dct.items()))
        _bit_depth, _spoof_ims = get_pixel_format_bit_depth(feature_dct["PixelFormat"])
        v._pixel_format = feature_dct["PixelFormat"]
        bit_depth[k] = _bit_depth
        disp_mins[k] = 0
        disp_maxs[k] = 2 ** _bit_depth
    
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
            label=f"Camera {_id}", tag=f"Camera {_id}", no_collapse=True, no_scrollbar=True
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
    aruco_save_data = {_cam: {"corners": [], "ids": []} for _cam in cameras.keys()}
    charuco_save_data = {_cam: {"corners": [], "ids": []} for _cam in cameras.keys()}
    pose_save_data = {_cam: {"pose": [], "rvec": [], "tvec": []} for _cam in cameras.keys()}
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
            time.sleep(.1)
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
                plt_val = cv2.merge([_dat[0]] *3)
                plt_val = plt_val.astype("uint8")
                use_img = threshold_image(_dat[0].copy())
                aruco_dat, charuco_dat = detect_charuco(use_img, board)
                
                # add if we get more than three detections
                if len(aruco_dat[0]) > 3:
                    aruco_save_data[_id]["corners"].append(aruco_dat[0])
                    aruco_save_data[_id]["ids"].append(aruco_dat[1])
                    charuco_save_data[_id]["corners"].append(charuco_dat[0])
                    charuco_save_data[_id]["ids"].append(charuco_dat[1])
                    img_save_data[_id] += _dat

                    pose, rvec, tvec = estimate_pose(
                        *charuco_dat,
                        intrinsic_matrix[_id],
                        distortion_coeffs[_id],
                        board,
                    )

                    pose_save_data[_id]["pose"].append(pose)
                    pose_save_data[_id]["rvec"].append(rvec)
                    pose_save_data[_id]["tvec"].append(tvec)
                     
                    # draw results
                    plt_val = cv2.aruco.drawDetectedMarkers(plt_val, *aruco_dat, [0, 255, 255])
                    plt_val = cv2.aruco.drawDetectedCornersCharuco(
                        plt_val, *charuco_dat, [255, 0, 0, 0]
                    )
                    plt_val = cv2.drawFrameAxes(
                        plt_val, intrinsic_matrix[_id], distortion_coeffs[_id], rvec, tvec, 0.05
                    )
                    plt_val = cv2.putText(
                        plt_val, str(frame_count[_id]), txt_pos, font, 1, (255, 255, 255)
                    )
                    frame_count[_id] += 1

                plt_val = plt_val.astype("float32")  / 255.
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
            with open(os.path.join(save_path, "calibration.pkl"), "wb") as f:
                pickle.dump(save_dictionary, f)
            with open(os.path.join(save_path, "metadata.toml"), "w") as f:
                toml.dump(metadata, f)
        [_cam.stop_acquisition() for _cam in cameras.values()]
        logging.info("Done...")
        dpg.destroy_context()


if __name__ == "__main__":
    cli()
