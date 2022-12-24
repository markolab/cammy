import multiprocessing
import queue
from typing import Optional
from util import initialize_camera

# HERE take the camera object and nest into a frame grabber that runs as multiprocessing.Process
# this will use try_pop_frame and fill a queue once started
class FrameGrabber(multiprocessing.Process):
	def __init__(self, queue, interface: str, id: str, config: Optional[dict]):
		multiprocessing.Process.__init__(self)
		self.queue = queue
		self.camera = initialize_camera(id=id, interface=interface, config=config)
		self.is_running = multiprocessing.Value("i", 0)
		self.id = id
	
	def run(self):
		self.is_running = 1
		self.camera_object.start_acquisition()
		while True:
			if bool(self.is_running):
				try:
					frame, timestamp = self.camera_object.try_pop_frame()
					if frame is not None:
						print(frame)
						self.queue.put((frame, timestamp))
				except (KeyboardInterrupt, SystemExit):
					self.camera_object.stop_acquisition()
					print(f"Exiting framegrabber {self.id}")
					break
			else:
				self.camera_object.stop_acquisition()
				print(f"Exiting framegrabber {self.id}")
				break