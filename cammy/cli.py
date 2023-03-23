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


from typing import Optional
from cammy.util import (
    get_all_camera_ids,
    intensity_to_rgba,
    get_queues,
    initialize_cameras,
    get_output_format,
    get_pixel_format_bit_depth,
    mpl_to_cv2_colormap,
)
from cammy.camera.aravis import AravisCamera
from cammy.camera.fake import FakeCamera
from cammy.record.video import FfmpegVideoRecorder, RawVideoRecorder


@click.group()
def cli():
    pass


@cli.command(name="aravis-load-settings")
def aravis_load_settings():
    # loads settings into camera memory
    raise NotImplementedError


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
# 1) ADD OPTION TO READ COUNTERS WITH COLUMN NAME THEN WE'RE DONE?
@cli.command(name="run")
@click.option("--all-cameras", is_flag=True)
@click.option("--interface", type=click.Choice(["aravis", "fake_custom", "all"]), default="all")
@click.option("--n-fake-cameras", type=int, default=1)
@click.option("--acquire", is_flag=True)
@click.option("--jumbo-frames", default=True, type=bool)
@click.option("--save-engine", type=click.Choice(["ffmpeg", "raw"]), default="raw")
@click.option("--display-downsample", type=int, default=1)
@click.option("--display-colormap", type=str, default=None)
@click.option("--hw-trigger", is_flag=True)
@click.option("--hw-trigger-rate", type=float, default=100.)
@click.option("--hw-trigger-pin-last", type=int, default=13)
@click.option(
    "--camera-options",
    type=click.Path(resolve_path=True, exists=True),
    help="TOML file with camera options",
)
def simple_preview(
    all_cameras: bool,
    interface: str,
    n_fake_cameras: int,
    camera_options: Optional[str],
    acquire: bool,
    jumbo_frames: bool,
    save_engine: str,
    display_downsample: int,
    display_colormap: Optional[str],
    hw_trigger: bool,
    hw_trigger_rate: float,
    hw_trigger_pin_last: int,
):
    import dearpygui.dearpygui as dpg
    import cv2
    import socket
    import datetime

    hostname = socket.gethostname()

    if display_colormap is None:
        display_colormap = mpl_to_cv2_colormap(colormap_default)
    else:
        display_colormap = mpl_to_cv2_colormap(display_colormap)

    if camera_options is not None:
        camera_dct = toml.load(camera_options)
    else:
        camera_dct = {}

    cameras = {}
    if all_cameras:
        ids = get_all_camera_ids(interface, n_cams=n_fake_cameras)
    else:
        raise NotImplementedError()

    # TODO: TURN INTO AN AUTOMATIC CHECK, IF NO FRAMES ARE GETTING
    # ACQUIRED, PAUSE FOR 1 SEC AND RE-INITIALIZE
    cameras = initialize_cameras(ids, camera_dct, jumbo_frames=jumbo_frames)
    del cameras
    time.sleep(2)

    cameras_metadata = {}
    bit_depth = {}
    trigger_pins = []
    cameras = initialize_cameras(ids, camera_dct, jumbo_frames=jumbo_frames)
    for i, (k, v) in enumerate(cameras.items()):
        feature_dct = v.get_all_features()
        feature_dct = dict(sorted(feature_dct.items()))
        bit_depth[k] = get_pixel_format_bit_depth(feature_dct["PixelFormat"])
        cameras_metadata[k] = feature_dct
        trigger_pins.append(hw_trigger_pin_last - i) # work backwards from last

    dpg.create_context()
    recorders = []
    write_dtype = {}
    if acquire:
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

        save_path = f"session_{init_timestamp_str} ({hostname})"
        if os.path.exists(save_path):
            raise RuntimeError(f"Directory {save_path} already exists")
        else:
            # dump in metadata
            os.makedirs(save_path)
            with open(os.path.join(save_path, "metadata.toml"), "w") as f:
                toml.dump(recording_metadata, f)

        settings_tags = {}
        settings_vals = {}
        with dpg.window(width=500, height=300, no_resize=True, tag="settings"):
            for k, v in show_fields.items():
                settings_tags[k] = dpg.add_input_text(default_value=v, label=k)
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

        # start a new context for acquisition
        dpg.create_context()

        # dump settings to toml file (along with start time of recording and hostname)
        for _id, _cam in cameras.items():
            cameras[_id].queue = use_queues["storage"][_id]
            if save_engine == "ffmpeg":
                _recorder = FfmpegVideoRecorder(
                    width=cameras[_id]._width,
                    height=cameras[_id]._height,
                    queue=cameras[_id].queue,
                    filename=os.path.join(save_path, f"{_id}.mkv"),
                    pixel_format=write_dtype[_id],
                )
            elif save_engine == "raw":
                _recorder = RawVideoRecorder(
                    queue=cameras[_id].queue,
                    filename=os.path.join(save_path, f"{_id}.dat"),
                    write_dtype=write_dtype[_id],
                )
            else:
                raise RuntimeError(f"Did not understanding VideoRecorder option {save_engine}")

            _recorder.daemon = True
            _recorder.start()
            recorders.append(_recorder)
    else:
        show_fields = {}
        use_queues = {}

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
    for _id, _cam in cameras.items():
        use_config = {}
        for k, v in camera_dct["display"].items():
            if k in _id:
                use_config = v

        with dpg.window(
            label=f"Camera {_id}", tag=f"Camera {_id}"
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
            miss_status[_id] = dpg.add_text(f"0 missed frames / 0 total")
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

    [_cam.start_acquisition() for _cam in cameras.values()]
    for _cam in cameras.values():
        _cam.count = 0

    dpg.create_viewport(title="Live preview", width=gui_x_max, height=gui_y_max)

    # dpg.set_viewport_vsync(False)
    # dpg.show_metrics()
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # 3/7/23 REMOVED EXTRA START_ACQUISITION, PUT GPIO IN WEIRD STATE
    # [print(_cam.camera.get_trigger_source()) for _cam in cameras.values()]
    # if using a hardware trigger, send out signals now...
    if hw_trigger:
        logging.info(f"Trigger pins: {trigger_pins}")
        from cammy.trigger.trigger import TriggerDevice
        trigger_dev = TriggerDevice(frame_rate=hw_trigger_rate, pins=trigger_pins)
        trigger_dev.start()
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
                        new_frame = _dat[0]
                        new_ts = _dat[1]
                dat[_id] = (new_frame, new_ts)

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
                    cam_fps = cameras[_id].fps
                    percent_missed = (miss_frames / total_frames) * 100
                    dpg.set_value(
                        miss_status[_id],
                        f"{miss_frames} missed / {total_frames} total ({percent_missed:.1f}% missed)\n{cam_fps:.1f} FPS",
                    )
                    # for k, v in use_queues["storage"].items():
                    # 	print(v.qsize())
            time.sleep(0.03)
            dpg.render_dearpygui_frame()
    finally:
        [_cam.stop_acquisition() for _cam in cameras.values()]
        if acquire:
            # for every camera ID wait until the queue has been written out
            for k, v in use_queues["storage"].items():
                while v.qsize() > 0:
                    time.sleep(0.1)
            for _recorder in recorders:
                _recorder.is_running = 0
                time.sleep(1)
        dpg.destroy_context()


if __name__ == "__main__":
    cli()
