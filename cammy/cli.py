import click
import numpy as np
import toml
import logging
import sys
import os
import time

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
    get_queues,
    initialize_cameras,
    get_output_format,
    get_pixel_format_bit_depth,
)
from cammy.camera.aravis import AravisCamera
from cammy.camera.fake import FakeCamera
from cammy.record.video import FfmpegVideoRecorder, RawVideoRecorder
from cammy.gui import FrameDisplay


@click.group()
def cli():
    pass


@cli.command(name="aravis-load-settings")
def aravis_load_settings():
    # loads settings into camera memory
    raise NotImplementedError


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
@click.option("--hw-trigger-rate", type=float, default=100.0)
@click.option("--hw-trigger-pin-last", type=int, default=13)
@click.option("--counters-name", type=str, default=["Trigger", "Exposure"], multiple=True)
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
    counters_name,
):
    import dearpygui.dearpygui as dpg
    import socket
    import datetime

    print(counters_name)
    counters_name = list(counters_name)
    counters_name = []
    hostname = socket.gethostname()

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
    cameras = initialize_cameras(
        ids, camera_dct, jumbo_frames=jumbo_frames, counters_name=counters_name
    )
    del cameras
    time.sleep(2)

    cameras_metadata = {}
    bit_depth = {}
    trigger_pins = []
    cameras = initialize_cameras(
        ids, camera_dct, jumbo_frames=jumbo_frames, counters_name=counters_name
    )
    for i, (k, v) in enumerate(cameras.items()):
        feature_dct = v.get_all_features()
        feature_dct = dict(sorted(feature_dct.items()))
        bit_depth[k] = get_pixel_format_bit_depth(feature_dct["PixelFormat"])
        cameras_metadata[k] = feature_dct
        trigger_pins.append(hw_trigger_pin_last - i)  # work backwards from last

    recorders = []
    write_dtype = {}

    if hw_trigger:
        logging.info(f"Trigger pins: {trigger_pins}")
        from cammy.trigger.trigger import TriggerDevice

        trigger_dev = TriggerDevice(frame_rate=hw_trigger_rate, pins=trigger_pins)
    else:
        trigger_dev = None

    use_queues = get_queues(list(ids.keys()))
    
    if acquire:
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

        # dump settings to toml file (along with start time of recording and hostname)
        for _id, _cam in cameras.items():
            cameras[_id].queue = use_queues["storage"][_id]
            if len(counters_name) > 0:
                timestamp_fields = counters_name + ["device_timestamp", "system_timestamp"]
            else:
                timestamp_fields = ["device_timestamp", "system_timestamp"]
            if save_engine == "ffmpeg":
                _recorder = FfmpegVideoRecorder(
                    width=cameras[_id]._width,
                    height=cameras[_id]._height,
                    queue=cameras[_id].queue,
                    filename=os.path.join(save_path, f"{_id}.mkv"),
                    pixel_format=write_dtype[_id],
                    timestamp_fields=timestamp_fields,
                )
            elif save_engine == "raw":
                _recorder = RawVideoRecorder(
                    queue=cameras[_id].queue,
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

    [_cam.start_acquisition() for _cam in cameras.values()]
    # if using a hardware trigger, send out signals now...
    if hw_trigger and (trigger_dev is not None):
        trigger_dev.start()

    for _cam in cameras.values():
        _cam.count = 0

    # 3/7/23 REMOVED EXTRA START_ACQUISITION, PUT GPIO IN WEIRD STATE
    # [print(_cam.camera.get_trigger_source()) for _cam in cameras.values()]
    frame_display = FrameDisplay(
        queue=use_queues["display"],
        cameras=cameras,
        downsample=display_downsample,
        display_params=camera_dct["display"],
        display_colormap=display_colormap,
    )
    frame_display.start()
    try:
        for _id, _cam in cameras.items():
            new_frame = None
            new_ts = None
            while True:
                _dat = _cam.try_pop_frame()
                if _dat[0] is None:
                    break
                else:
                    # load up the queues
                    for k, v in use_queues.items():
                        v[_id].put(_dat)
                        # if "storage" in use_queues.keys():
                #     for k, v in use_queues["storage"].items():
                #         logging.debug(v.qsize())

    finally:
        [_cam.stop_acquisition() for _cam in cameras.values()]
        if acquire:
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


if __name__ == "__main__":
    cli()
