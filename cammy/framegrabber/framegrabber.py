import multiprocessing
import queue


# HERE take the camera object and nest into a frame grabber that runs as multiprocessing.Process
# this will use try_pop_frame and fill a queue once started
class FrameGrabber(multiprocessing.Process):
	def __init__(self, queue, camera_object, id):
		multiprocessing.Process.__init__(self)
		self.queue = queue
		self.camera_object = camera_object
		self.is_running = multiprocessing.Value("i", 0)
		self.id = id
	
	def run(self):
		self.is_running = 1
		self.camera_object.start_acquisition()
		while True:
			if bool(self.is_running):
				try:
					frame, timestamp = self.camera_object.try_pop_frame()
					print(frame)
					if frame is not None:
						self.queue.put((frame, timestamp))
				except (KeyboardInterrupt, SystemExit):
					self.camera_object.stop_acquisition()
					print(f"Exiting framegrabber {self.id}")
					break
			else:
				self.camera_object.stop_acquisition()
				print(f"Exiting framegrabber {self.id}")
				break