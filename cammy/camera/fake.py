import numpy as np
import time
from cammy.camera.base import CammyCamera
from typing import Optional


# make random data and scale by exposure time/frame rate
class DummyDevice:
	def __init__(self, pixel_format: str = "MONO_16"):
		self.exposure_time = 100.0
		self.frame_rate = 30.0
		self.pixel_format = pixel_format
		self.is_running = False

	def set_exposure_time(self, exposure_time):
		self.exposure_time = exposure_time

	def get_exposure_time(self) -> float:
		return self.exposure_time

	def set_frame_rate(self, frame_rate):
		self.frame_rate = frame_rate

	def get_frame_rate(self) -> float:
		return self.frame_rate

	def set_pixel_format(self, pixel_format):
		self.pixel_format = pixel_format

	def get_pixel_format(self) -> str:
		return self.pixel_format

	def start_acquisition(self):
		self.is_running = True

	def stop_acquisition(self):
		self.is_running = False


# add multiprocessing.Process, then run is where we generate frames and fill the queue...
class FakeCamera(CammyCamera):
	def __init__(
		self,
		id: Optional[str],
		exposure_time: float = 0.01,
		fps: float = 60,
		pixel_format: str = "MONO_8",
		width: int = 600,
		height: int = 480,
	):

		super(CammyCamera, self).__init__()

		self.camera = DummyDevice()
		self.id = id
		self.set_exposure_time(exposure_time)
		self.set_frame_rate(fps)
		self.set_pixel_format(pixel_format)
		self._height = height
		self._width = width
		self._last_frame_grab = time.time()
		self.fps = fps
		self.frame_count = 0
		self.missed_frames = 0
		self.total_frames = 0

		if pixel_format == "MONO_16":
			self._use_dtype = np.uint16
			self._max_val = 2**16
		elif pixel_format == "MONO_8":
			self._use_dtype = np.uint8
			self._max_val = 2**8

	def try_pop_frame(self):
		cur_time = time.time()
		if self.camera.is_running and ((cur_time - self._last_frame_grab) > (1. / self.fps)):
			dat = np.random.randint(0, self._max_val, size=(self._height, self._width)).astype(
				"float"
			)
			# dat = np.ones((self._height, self._width))
			dat *= self.get_exposure_time()
			self._last_frame_grab = cur_time
			self.frame_count += 1
			return dat.astype(self._use_dtype), None
		else:
			return None, None
