import click
import numpy as np
import toml
import logging
import sys
from typing import Optional
from cammy.util import get_all_cameras_aravis, intensity_to_rgba, get_queues
from cammy.camera.aravis import AravisCamera
from cammy.camera.fake import FakeCamera
from cammy.framegrabber.framegrabber import FrameGrabber


@click.group()
def cli():
	pass


@cli.command(name="aravis-load-settings")
def aravis_load_settings():
	# loads settings into camera memory
	raise NotImplementedError


@cli.command(name="live-preview")
def live_preview():
	# fire up all aravis devices and gives the user widgets to test relevant settings
	raise NotImplementedError


@cli.command(name="simple-preview")
@click.option("--all-cameras-aravis", is_flag=True)
@click.option("--use-fake-camera", is_flag=True)
@click.option("--n-fake-cameras", type=int, default=1)
@click.option("--fake-camera-interface", type=str, default="custom")
@click.option(
	"--camera-options",
	type=click.Path(resolve_path=True, exists=True),
	help="TOML file with camera options",
)
def simple_preview(
	all_cameras_aravis: bool,
	use_fake_camera: bool,
	n_fake_cameras: int,
	fake_camera_interface: str,
	camera_options: Optional[str],
):
	import dearpygui.dearpygui as dpg
	import time
	import queue
	import cv2

	logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
						format="[%(asctime)s]: %(message)s",
						datefmt="%Y-%m-%d %H:%M:%S")

	if camera_options is not None:
		camera_dct = toml.load(camera_options)
	else:
		camera_dct = {}
		
	# for labeling videos
	font = cv2.FONT_HERSHEY_SIMPLEX
	white = (255, 255, 255)
	txt_pos = (25, 25)

	# simply spool up and show input from all detected cameras
	cameras = []
	if all_cameras_aravis and not use_fake_camera:
		ids = get_all_cameras_aravis()  # ids of all cameras
		for _id in ids:
			logging.info(f"Found Aravis camera {_id}")
			_cam = AravisCamera(id=_id)
			if _id in camera_dct.keys():
				for k, v in camera_dct[_id].items():
					logging.info(f"{k} is {_cam.get_feature(k)}")
					_cam.set_feature(k, v)
			cameras.append(_cam)
	elif use_fake_camera:
		# spool up n fake cameras
		for i in range(n_fake_cameras):
			if fake_camera_interface == "aravis":
				_cam = AravisCamera(fake_camera=True, id=f"Fake_{i+1}")
			elif fake_camera_interface == "custom":
				_cam = FakeCamera(id=f"Fake_{i+1}")
			else:
				raise RuntimeError(
					f"Did not understand fake camera interface {fake_camera_interface}"
				)
			cameras.append(_cam)
	else:
		raise RuntimeError("Incompatible flag settings")

	dpg.create_context()
	dpg.create_viewport(title="Custom Title", width=1000, height=1000)
	dpg.setup_dearpygui()

	with dpg.texture_registry(show=True):
		for _cam in cameras:
			_id = _cam.id
			blank_data = np.zeros((_cam._height, _cam._width, 4), dtype="float32")
			dpg.add_raw_texture(
				_cam._width,
				_cam._height,
				blank_data,
				tag=f"texture_{_id}",
				format=dpg.mvFormat_Float_rgba,
			)
	# queues = get_queues([_cam.id for _cam in cameras]) # returns a dictionary of queues
	# frame_grabbers = []
	for _cam in cameras:
		_id = _cam.id
		# new_grabber = FrameGrabber(queue=queues["display"][_id], camera_object=_cam, id=_cam.id)
		# new_grabber.daemon = True
		# frame_grabbers.append(new_grabber)
		with dpg.window(label=f"Camera {_id}"):
			dpg.add_image(f"texture_{_id}")
			# add sliders/text boxes for exposure time and fps

	# [_grabber.start() for _grabber in frame_grabbers]
	[_cam.start_acquisition() for _cam in cameras]
	counts = [0 for _cam in cameras]
	for _cam in cameras:
		_cam.count = 0

	# initiate a framegrabber per camera, then turn them on
	dpg.show_metrics()
	dpg.show_viewport()

	try:
		while dpg.is_dearpygui_running():
			# for k, v in queues["display"].items():
			#     # always clear out the queue and get whatever was last
			#     dat = None
			#     while True:
			#         try:
			#             dat = v.get_nowait()
			#         except queue.Empty:
			#             break
			# if dat is not None:
			#     plt_val = intensity_to_rgba(dat[0])
			#     dpg.set_value(f"texture_{k}", plt_val)
			# don't use multiprocessing so we can mod on the fly, for acquisition we can use mp
			# dat = [(_cam.try_pop_frame(), _cam) for _cam in cameras]
			dat = []
			for _cam in cameras:
				new_frame = None
				new_ts = None
				while True:
					_dat = _cam.try_pop_frame()
					if _dat[0] is None:
						break
					else:
						new_frame = _dat[0]
						new_ts = _dat[1]
				dat.append((new_frame, new_ts))

			for (_dat, _cam) in zip(dat, cameras):
				if _dat[0] is not None:
					plt_val = intensity_to_rgba(_dat[0]).astype("float32")
					cv2.putText(plt_val, str(_cam.count), txt_pos, font, 1, (1, 1, 1, 1))
					dpg.set_value(f"texture_{_cam.id}", plt_val)
					_cam.count += 1
			dpg.render_dearpygui_frame()
			# time.sleep(0.01)
	finally:
		[_cam.stop_acquisition() for _cam in cameras]
		dpg.destroy_context()
		# for _grabber in frame_grabbers:
		#     _grabber.is_running = 0
		# time.sleep(1)


@cli.command(name="get-genicam-xml")
@click.argument("device")
def generate_config(device: str):
	# uses aravis to aget a genicam xml with all features on camera
	raise NotImplementedError


if __name__ == "__main__":
	cli()
