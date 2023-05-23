import multiprocessing
import queue
from typing import Optional
from pickle import UnpicklingError

# simple data writer, should be general enough to take 1d/2d/etc. data
class BaseRecord(multiprocessing.Process):
	def __init__(self, save_queue, filename):
		multiprocessing.Process.__init__(self)
		self.save_queue = save_queue
		self.is_running = multiprocessing.Value("i", 0)
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
		while True:
			if bool(self.is_running):
				dat = None			
				try:
					dat = self.save_queue.get_nowait()
				except (queue.Empty, KeyboardInterrupt, EOFError, UnpicklingError):
					continue

				if dat is not None:
					try:
						self.write_data(dat)
					except KeyboardInterrupt:
						self.write_data(dat)
				else:
					print(f"Exiting recorder {self.name}")
					self.close_writer()
					break
			else:
				self.close_writer()
				break