import multiprocessing
import queue
from typing import Optional
import sys
from cammy.util import initialize_camera

# HERE take the camera object and nest into a frame grabber that runs as multiprocessing.Process
# this will use try_pop_frame and fill a queue once started
class FrameGrabber(multiprocessing.Process):
	def __init__(self, queue, interface: str, id: str, config: Optional[dict]):
		multiprocessing.Process.__init__(self)
		self.queue = queue
		self._camera_kwargs = {"id": id, "interface": interface, "config": config}
		self.is_running = multiprocessing.Value("i", 0)
		self.id = id
	
	def run(self):
		camera_object = initialize_camera(**self._camera_kwargs)
		self.is_running = 1
		camera_object.start_acquisition()
		while True:
			if bool(self.is_running):
				try:
					frame, timestamp = camera_object.try_pop_frame()
					if frame is not None:
						self.queue.put((frame, timestamp))
				except (KeyboardInterrupt, SystemExit):
					camera_object.stop_acquisition()
					print(f"Exiting framegrabber {self.id}")
					break
			else:
				camera_object.stop_acquisition()
				print(f"Exiting framegrabber {self.id}")
				break