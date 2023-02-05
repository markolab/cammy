import gi
import multiprocessing
import cv2
import numpy as np
import logging
import sys
gi.require_version("Aravis", "0.8")
from gi.repository import Aravis


logger = logging.getLogger(__name__)


def get_all_camera_ids(interface="aravis", n_cams=1):
	if (interface == "aravis") or (interface == "all"):
		Aravis.update_device_list()
		n_cams = Aravis.get_n_devices()
		use_ids = [Aravis.get_device_id(i) for i in range(n_cams)] 
	elif (interface == "fake") or (interface == "fake_custom"):
		use_ids = {f"Fake_{i + 1}" for i in range(n_cams)}
	else:
		raise RuntimeError(f"Did not understand interface {interface}")
	ids = {_id: interface for _id in use_ids}
	return ids


def get_pixel_format_aravis(pixel_format):
	if pixel_format == "MONO_16":
		arv_format = Aravis.PIXEL_FORMAT_MONO_16
	elif pixel_format == "MONO_12":
		arv_format = Aravis.PIXEL_FORMAT_MONO_12
	elif pixel_format == "MONO_8":
		arv_format = Aravis.PIXEL_FORMAT_MONO_8
	else:
		raise RuntimeError(f"Did not understand pixel format {pixel_format}")

	return arv_format


def get_pixel_format_bit_depth(pixel_format):
	if pixel_format == "MONO_16":
		bit_depth = 16
	elif pixel_format == "MONO_12":
		bit_depth = 12
	elif pixel_format == "MONO_8":
		bit_depth = 8
	else:
		raise RuntimeError(f"Did not understand pixel format {pixel_format}")

	return bit_depth


def get_queues(ids=None) -> dict:
	if ids:
		queues = {}
		queues["display"] = {id: multiprocessing.Manager().Queue(100) for id in ids}
		queues["storage"] = {id: multiprocessing.Manager().Queue(100) for id in ids} 
		return queues
	else:
		raise RuntimeError("Must specify IDs to construct queues")



def intensity_to_rgba(frame, minval=1800, maxval=2200, colormap=cv2.COLORMAP_TURBO):
	new_frame = np.ones((frame.shape[0], frame.shape[1], 4))
	disp_frame = frame.copy().astype("float")
	disp_frame -= minval
	disp_frame[disp_frame<0] = 0
	disp_frame /= np.abs(maxval - minval)
	disp_frame[disp_frame>=1] = 1
	disp_frame *= 255
	bgr_frame = cv2.applyColorMap(disp_frame.astype(np.uint8), colormap)
	rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
	new_frame[:,:,:3] = rgb_frame / 255.
	return new_frame


def initialize_camera(id, interface: str, config={}, **kwargs):
	if (interface == "aravis") or (interface == "all"):
		from cammy.camera.aravis import AravisCamera
		cam = AravisCamera(id=id, **kwargs)
		if config is not None:
			for k, v in config.items():
				cam.set_feature(k, v)
	elif interface == "fake_custom":
		from cammy.camera.fake import FakeCamera
		cam = FakeCamera(id=id)
	else:
		raise RuntimeError(f"Did not understand interface {interface}")
	return cam