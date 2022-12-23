import gi
import multiprocessing
import cv2
import numpy as np
gi.require_version("Aravis", "0.8")
from gi.repository import Aravis



def get_all_cameras_aravis():
	Aravis.update_device_list()
	n_cams = Aravis.get_n_devices()
	ids = [Aravis.get_device_id(i) for i in range(n_cams)]
	return ids


def get_pixel_format_aravis(pixel_format):
	if pixel_format == "MONO_16":
		arv_format = Aravis.PIXEL_FORMAT_MONO_16
	elif pixel_format == "MONO_8":
		arv_format = Aravis.PIXEL_FORMAT_MONO_8
	else:
		raise RuntimeError(f"Did not understand pixel format {pixel_format}")

	return arv_format


def get_queues(ids=None) -> dict:
	if ids:
		queues = {}
		queues["display"] = {id: multiprocessing.Manager().Queue(50) for id in ids}
		queues["storage"] = {id: multiprocessing.Manager().Queue(50) for id in ids} 
		return queues
	else:
		raise RuntimeError("Must specify IDs to construct queues")



def intensity_to_rgba(frame, minval=300, maxval=800, colormap=cv2.COLORMAP_TURBO):
	new_frame = np.ones((frame.shape[0], frame.shape[1], 4))
	disp_frame = frame.copy().astype("float")
	disp_frame -= minval
	disp_frame[disp_frame<0] = 0
	disp_frame /= maxval
	disp_frame[disp_frame>=1] = 1
	disp_frame *= 255
	rgb_frame = cv2.applyColorMap(disp_frame.astype(np.uint8), colormap)
	new_frame[:,:,:3] = rgb_frame / 255.
	return new_frame