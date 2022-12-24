from cammy.record.base import BaseRecord
import subprocess
import numpy as np


class VideoRecorder(BaseRecord):
	def __init__(
		self,
		width=600,
		height=420,
		threads=8,
		fps=30,
		pixel_format="gray16le",
		codec="ffv1",
		slices=24,
		slicecrc=1,
		filename="test.avi",
		queue=None,
	):

		super(BaseRecord, self).__init__()
		command = ['ffmpeg',
               '-y',
               '-loglevel', 'fatal',
               '-framerate', str(fps),
               '-f', 'rawvideo',
               '-s', f'{str(width)}x{str(height)}',
               '-pix_fmt', pixel_format,
               '-i', '-',
               '-an',
               '-vcodec', codec,
               '-threads', str(threads),
               '-slices', str(slices),
               '-slicecrc', str(slicecrc),
               '-r', str(fps),
               filename]
		print(command)
		self._command = command
		self.queue = queue
		# self.is_running = multiprocessing.Value("i", 0)
		self.id = id
		self.filename=filename


	def write_data(self, data):
		vdata, tstamp = data
		if vdata.ndim == 3:
			for _frame in vdata:
				self._pipe.stdin.write(_frame.astype("uint16").tostring())
		elif vdata.ndim == 2:
			self._pipe.stdin.write(vdata.astype("uint16").tostring())
		else:
			raise RuntimeError("Frames must be 2d or 3d")


	def open_writer(self):
		pipe = subprocess.Popen(self._command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
		self._pipe = pipe


	def close_writer(self):
		self._pipe.stdin.close()