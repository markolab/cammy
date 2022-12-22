import os
import numpy as np
import logging
from cammy.camera.camera import CammyCamera
from typing import Optional


# make random data and scale by exposure time/frame rate
class DummyDevice():
	def __init__(self, pixel_format: str="MONO_16"):
		self.exposure_time = None
		self.frame_rate = None
		self.pixel_format = pixel_format
		self.is_running = False

	def set_exposure_time(self, exposure_time):
		self.exposure_time = exposure_time


	def get_exposure_time(self):
		return self.exposure_time


	def set_frame_rate(self, frame_rate):
		self.frame_rate = frame_rate


	def get_frame_rate(self):
		return self.frame_rate


	def set_pixel_format(self, pixel_format):
		self.pixel_format = pixel_format


	def get_pixel_format(self):
		return self.pixel_format


	def start_acquisition(self):
		self.is_running = True


	def stop_acquisition(self):
		self.is_running = False



	

	
class FakeCamera(CammyCamera):
	def __init__(
		self,
		id: Optional[str],
		exposure_time: float=1000,
		fps: float=30,
		pixel_format: str="MONO_16",
		width: int=600,
		height: int=480,
	):

		super(CammyCamera, self).__init__()

		self.camera = DummyDevice()
		self.id = id
		self.set_exposure_time(exposure_time)
		self.set_frame_rate(fps)
		self.set_pixel_format(pixel_format)
		self._height = height
		self._width = width


	def try_pop_frame(self):
		if self.camera.is_running:
			return np.random.randint(0, 256, size=(self._height, self._width)), None
		else:
			return None