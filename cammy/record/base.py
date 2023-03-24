import multiprocessing
import queue
from typing import Optional


# simple data writer, should be general enough to take 1d/2d/etc. data
class BaseRecord(multiprocessing.Process):
	def __init__(self, queue, filename):
		multiprocessing.Process.__init__(self)
		self.queue = queue
		self.is_running = multiprocessing.Value("i", 0)

		print("TEST")
		self.id = id
		self.filename=filename


	def write_data(self, data):
		pass


	def open_writer(self):
		pass


	def close_writer(self):
		pass


	def run(self):
		self.is_running = 1
		self.open_writer()
		print("TEST")
		while True:
			print("TEST")
			if bool(self.is_running):
				dat = None			
				try:
					dat = self.queue.get_nowait()
					print("TEST")
					print(dat)
				except (queue.Empty, KeyboardInterrupt, EOFError):
					continue
				except (SystemExit, BrokenPipeError):
					pass

				if dat is not None:
					try:
						self.write_data(dat)
					except KeyboardInterrupt:
						self.write_data(dat)
				else:
					print(f"Exiting recorder {self.id}")
					break
			else:
				break