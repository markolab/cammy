import multiprocessing
import queue
from typing import Optional
import sys


# simple data writer, should be general enough to take 1d/2d/etc. data
class BaseRecord(multiprocessing.Process):
	def __init__(self, queue, filename):
		multiprocessing.Process.__init__(self)
		self.queue = queue
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
				try:
					dat = None
					try:
						dat = self.queue.get()
					except queue.Empty:
						continue
					if dat is not None:
						self.write_data(dat)
				except (KeyboardInterrupt, SystemExit):
					while True:
						try:
							dat = self.queue.get()
							if dat is not None:
								self.write_data(dat)
						except queue.Empty:
							break
					self.close_writer()
					print(f"Exiting recorder {self.id}")
					break
			else:
				while True:
					try:
						dat = self.queue.get()
						if dat is not None:
							self.write_data(dat)
					except queue.Empty:
						break
				self.close_writer()
				print(f"Exiting recorder {self.id}")
				break