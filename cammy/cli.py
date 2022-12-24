import click
import numpy as np
import toml
import logging
import sys
import time
import queue

logging.basicConfig(
	stream=sys.stdout,
	level=logging.DEBUG,
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


@cli.command(name="simple-preview")
@click.option("--all-cameras", is_flag=True)
@click.option("--interface", type=click.Choice(["aravis", "fake_custom", "all"]), default="all")
@click.option("--n-fake-cameras", type=int, default=1)
@click.option("--acquire", is_flag=True)
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
):
	import dearpygui.dearpygui as dpg
	import cv2

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
		NotImplementedError()

	if acquire:
		queue_keys = ["display", "storage"]
	else:
		queue_keys = ["display"]
	use_queues = get_queues(list(ids.keys()), keys=queue_keys)
	for _id, _interface in ids.items():
		cameras[_id] = initialize_camera(
			_id, _interface, camera_dct.get(_id), queues=use_queues[_id]
		)

	if acquire:
		recorders = []
		for _id, _cam in cameras.items():
			_recorder = VideoRecorder(
				width=cameras[_id]._width,
				height=cameras[_id]._height,
				queue=cameras[_id].queues["storage"],
			)
			_recorder.daemon = True
			_recorder.start()
			recorders.append(_recorder)

	dpg.create_context()
	dpg.create_viewport(title="Custom Title", width=1000, height=1000)
	dpg.setup_dearpygui()

	with dpg.texture_registry(show=True):
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
		with dpg.window(label=f"Camera {_id}"):
			dpg.add_image(f"texture_{_id}")
			# add sliders/text boxes for exposure time and fps

	[_cam.start_acquisition() for _cam in cameras.values()]
	for _cam in cameras.values():
		_cam.count = 0

	dpg.show_metrics()
	dpg.show_viewport()

	try:
		while dpg.is_dearpygui_running():
			dat = {}
			for _id, _q in use_queues.items():
				# always clear out the queue and get whatever was last
				new_frame = None
				new_ts = None
				while True:
					try:
						new_frame, new_ts = _q["display"].get_nowait()
					except queue.Empty:
						break
				dat[_id] = (new_frame, new_ts)

			for _id, _dat in dat.items():
				if _dat[0] is not None:
					plt_val = intensity_to_rgba(_dat[0]).astype("float32")
					cv2.putText(plt_val, str(cameras[_id].count), txt_pos, font, 1, (1, 1, 1, 1))
					dpg.set_value(f"texture_{cameras[_id].id}", plt_val)
					cameras[_id].count += 1
			dpg.render_dearpygui_frame()
	finally:
		[_cam.stop_acquisition() for _cam in cameras.values()]
		if acquire:
			for _recorder in recorders:
				_recorder.is_running = 0
				time.sleep(4)
		dpg.destroy_context()


@cli.command(name="get-genicam-xml")
@click.argument("device")
def generate_config(device: str):
	# uses aravis to aget a genicam xml with all features on camera
	raise NotImplementedError


if __name__ == "__main__":
	cli()
