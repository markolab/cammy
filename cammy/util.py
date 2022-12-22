import gi
import multiprocessing
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


def get_queues(ids=None):
	if ids:
		queues = {id: multiprocessing.Manager().Queue(20) for id in ids}
	else:
		return None