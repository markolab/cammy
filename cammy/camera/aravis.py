import gi
import os
import ctypes
import numpy as np
import logging
from cammy.util import get_pixel_format_aravis
from cammy.camera.camera import CammyCamera
from typing import Optional

gi.require_version("Aravis", "0.8")
from gi.repository import Aravis


class AravisCamera(CammyCamera):
	def __init__(
		self,
		id: Optional[str],
		exposure_time: float=1000,
		fps: float=30,
		pixel_format: str="MONO_16",
		buffer_size: int=3,
		fake_camera: bool=False,
		auto_exposure: bool=False
	):

		super().__init__()
		# prelims for using fake camera
		if fake_camera:
			Aravis.set_fake_camera_genicam_filename(os.getenv("FAKE_GENICAM_PATH"))
			Aravis.enable_interface("Fake")

		self.camera = Aravis.Camera.new(id)
		# self.camera = Aravis.Camera() # THIS IS JUST FOR PYLANCE
		
		if not auto_exposure:
			try:
				self.camera.set_exposure_time_auto(0)
				self.camera.set_gain_auto(0)
			except (AttributeError, gi.repository.GLib.GError) as e:
				print(e)

		self.set_exposure_time(exposure_time)
		self.set_frame_rate(fps)
		self.set_pixel_format(get_pixel_format_aravis(pixel_format))
		[x,y,width,height] = self.camera.get_region()

		self._payload = self.camera.get_payload() # size of payload
		self._width = width
		self._height = height
	
		# stage stream
		self.stream = self.camera.create_stream()
		# Aravis.FakeCamera.set_fill_pattern(self.camera, self.fill_pattern_callback, [0,0,0])
		# self.stream = self.camera.create_stream(self.fake_data_callback, None)
		
		for i in range(buffer_size):
			self.stream.push_buffer(Aravis.Buffer.new_allocate(self._payload))

	

	# def fill_pattern_callback(self, user_data, cb_type):
	# 	print(user_data)


	# def fake_data_callback(self, user_data, cb_type, buffer):
	# 	print(buffer)
	# 	print(type(buffer))
	# 	self.stream.push_buffer(Aravis.Buffer.new_allocate(self._payload))		


	# https://github.com/SintefManufacturing/python-aravis/blob/master/aravis.py#L162
	def try_pop_frame(self):
		buffer = self.stream.try_pop_buffer()
		if buffer:
			frame = self._array_from_buffer_address(buffer)
			timestamp = buffer.get_timestamp()
			return frame, timestamp
		else:
			return None, None

	# https://github.com/SintefManufacturing/python-aravis/blob/master/aravis.py#L180
	def _array_from_buffer_address(self, buffer):
		if not buffer:
			return None
		pixel_format = buffer.get_image_pixel_format()
		bits_per_pixel = pixel_format >> 16 & 0xff
		if bits_per_pixel == 8:
			INTP = ctypes.POINTER(ctypes.c_uint8)
		else:
			INTP = ctypes.POINTER(ctypes.c_uint16)
		addr = buffer.get_data()
		ptr = ctypes.cast(addr, INTP)
		im = np.ctypeslib.as_array(ptr, (buffer.get_image_height(), buffer.get_image_width()))
		im = im.copy()
		return im