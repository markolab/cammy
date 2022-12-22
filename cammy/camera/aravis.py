import gi
import os
from cammy.util import get_pixel_format_aravis
from typing import Optional

gi.require_version("Aravis", "0.8")
from gi.repository import Aravis


class AravisCamera:
	def __init__(
		self,
		id: Optional[str],
		exposure_time: float=1000,
		fps: float=30,
		pixel_format: str="MONO_16",
		acquisition_mode: str="continuous",
		buffer_size: int=10,
		fake_camera: bool=False,
		auto_exposure: bool=False
	):

		# prelims for using fake camera
		if fake_camera:
			Aravis.set_fake_camera_genicam_filename(os.getenv("FAKE_GENICAM_PATH"))
			Aravis.enable_interface("Fake")

		self.camera = Aravis.Camera.new(id)
		self.camera = Aravis.Camera() # THIS IS JUST FOR PYLANCE
		
		if not auto_exposure:
			self.camera.set_exposure_time_auto(0)
			self.camera.set_gain_auto(0)

	
		self.camera.set_exposure_time(exposure_time)
		self.camera.set_frame_rate(fps)
		self.camera.set_pixel_format(get_pixel_format_aravis(pixel_format))
		[x,y,width,height] = self.camera.get_region()


		self._payload = self.camera.get_payload() # size of payload
		self._width = width
		self._height = height
	
		# stage stream
		self.stream = self.camera.create_stream()

		for i in range(buffer_size):
			self.stream.push_buffer(Aravis.Buffer.new_allocate(self._payload))

		def initiate_acquisition(self):
			self.camera.start_acquisition()


