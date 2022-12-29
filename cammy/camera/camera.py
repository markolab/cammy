import numpy as np
import logging
from typing import Optional


class CammyCamera:
	def __init__(
		self,
		id: Optional[str],
	):

		self.id = id
		self.camera = None
		self.queue = None
		self.missed_frames = 0
		self.total_frames = 0

	def start_acquisition(self):
		if self.camera:
			self.camera.start_acquisition()


	def stop_acquisition(self):
		if self.camera:
			self.camera.stop_acquisition()


	def set_exposure_time(self, exposure_time):
		if self.camera:
			self.camera.set_exposure_time(exposure_time)


	def get_exposure_time(self):
		if self.camera:
			return self.camera.get_exposure_time()


	def set_frame_rate(self, fps):
		if self.camera:
			self.camera.set_frame_rate(fps)


	def get_frame_rate(self, fps):
		if self.camera:
			return self.camera.get_frame_rate(fps)


	def set_pixel_format(self, pixel_format):
		if self.camera:
			self.camera.set_pixel_format(pixel_format)


	def get_pixel_format(self):
		if self.camera:
			return self.camera.get_pixel_format()

	def get_all_features(self):
		return {}	