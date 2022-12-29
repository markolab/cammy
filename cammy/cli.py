import click
import numpy as np
import toml
import logging
import sys
import os
import time

logging.basicConfig(
    stream=sys.stdout,
    level=logging.WARNING,
    format="[%(asctime)s]:%(levelname)s:%(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


from typing import Optional
from cammy.util import get_all_camera_ids, intensity_to_rgba, get_queues, initialize_camera
from cammy.camera.aravis import AravisCamera
from cammy.camera.fake import FakeCamera
from cammy.record.video import VideoRecorder


@click.group()
def cli():
    pass


@cli.command(name="aravis-load-settings")
def aravis_load_settings():
    # loads settings into camera memory
    raise NotImplementedError


# TODO:
# 1) TEST TO ENSURE EVERYTHING GETS CLOSED AND FLUSHED PROPERLY
# 2) ANYTHING TO ADD TO FILE FORMAT?
# 3) ADD STATUS BAR TO SHOW NUMBER OF FRAMES DROPPED RELATIVE TO TOTAL
@cli.command(name="simple-preview")
@click.option("--all-cameras", is_flag=True)
@click.option("--interface", type=click.Choice(["aravis", "fake_custom", "all"]), default="all")
@click.option("--n-fake-cameras", type=int, default=1)
@click.option("--acquire", is_flag=True)
@click.option("--jumbo-frames", default=True, type=bool)
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
):
    import dearpygui.dearpygui as dpg
    import cv2
    import socket
    import datetime

    hostname = socket.gethostname()

    if camera_options is not None:
        camera_dct = toml.load(camera_options)
    else:
        camera_dct = {}

    # for labeling videos
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    txt_pos = (25, 25)

    cameras = {}
    if all_cameras:
        ids = get_all_camera_ids(interface, n_cams=n_fake_cameras)
    else:
        raise NotImplementedError()

    cameras_metadata = {}
    for _id, _interface in ids.items():
        cameras[_id] = initialize_camera(
            _id, _interface, camera_dct.get(_id), jumbo_frames=jumbo_frames
        )
        feature_dct = cameras[_id].get_all_features()
        feature_dct = dict(sorted(feature_dct.items()))
        cameras_metadata[_id] = feature_dct

    dpg.create_context()
    recorders = []

    if acquire:

        use_queues = get_queues(list(ids.keys()))
        basedir = os.path.dirname(os.path.abspath(__file__))
        metadata_path = os.path.join(basedir, "metadata.toml")
        show_fields = toml.load(metadata_path)["show_fields"]
        init_timestamp = datetime.datetime.now()

        recording_metadata = {
            "data_type": "UInt16[]",
            "codec": "ffv1",
            "pixel_format": "gray16le",
            "start_time": init_timestamp.isoformat(),
            "cameras": ids,
            "camera_metadata": cameras_metadata,
        }

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
            _recorder = VideoRecorder(
                width=cameras[_id]._width,
                height=cameras[_id]._height,
                queue=cameras[_id].queue,
                filename=os.path.join(save_path, f"{_id}.avi"),
            )
            _recorder.daemon = True
            _recorder.start()
            recorders.append(_recorder)
    else:
        show_fields = {}
        use_queues = {}

    with dpg.texture_registry(show=False):
        for _id, _cam in cameras.items():
            blank_data = np.zeros((_cam._height, _cam._width, 4), dtype="float32")
            dpg.add_raw_texture(
                _cam._width,
                _cam._height,
                blank_data,
                tag=f"texture_{_id}",
                format=dpg.mvFormat_Float_rgba,
            )
    for _id, _cam in cameras.items():
        miss_status = {}
        with dpg.window(label=f"Camera {_id}"):
            dpg.add_image(f"texture_{_id}")
            miss_status[_id] = dpg.add_text(f"0 missed frames / 0 total")
            # add sliders/text boxes for exposure time and fps

    [_cam.start_acquisition() for _cam in cameras.values()]
    for _cam in cameras.values():
        _cam.count = 0

    dpg.create_viewport(title="Live preview", width=1000, height=1000)
    dpg.set_viewport_vsync(False)
    dpg.show_metrics()
    dpg.setup_dearpygui()
    dpg.show_viewport()

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
                    plt_val = intensity_to_rgba(_dat[0]).astype("float32")
                    cv2.putText(plt_val, str(cameras[_id].count), txt_pos, font, 1, (1, 1, 1, 1))
                    dpg.set_value(f"texture_{cameras[_id].id}", plt_val)
                    cameras[_id].count += 1
                    percent_missed = (
                        float(cameras[_id].missed_frames) / camera[_id].total_frames
                    ) * 100
                    dpg.set_value(
                        miss_status[_id],
                        f"{cameras[_id].missed_frames} missed / {cameras[_id].total_frames} total ({percent_missed}% missed)",
                    )
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


@cli.command(name="get-genicam-xml")
@click.argument("device")
def generate_config(device: str):
    # uses aravis to aget a genicam xml with all features on camera
    raise NotImplementedError


if __name__ == "__main__":
    cli()
