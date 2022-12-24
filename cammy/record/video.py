from cammy.record.base import BaseRecord
import subprocess
import numpy as np


class VideoRecorder(BaseRecord):
	def __init__(
		self,
		width=600,
		height=420,
		threads=1,
		fps=30,
		pixel_format="gray16le",
		codec="ffv1 -level 3",
		slices=24,
		slicecrc=1,
		filename="test.avi",
	):
		super(BaseRecord, self).__init__()

		command = [
			"ffmpeg",
			"-y",
			"-loglevel fatal",
			f"-framerate {str(fps)}",
			"-f rawvideo",
			f"-s {width:d}x{height:d}",
			f"-pix_fmt {pixel_format}",
			"-i -",
			"-an",
			f"-vcodec {codec}",
			f"-threads {str(threads)}",
			f"-slices {str(slices)}",
			f"-slicecrc {str(slicecrc)}",
			f"-r {str(fps)}",
			"-g 1",
			filename,
		]
		self._command = command


	def write_data(self, data):
		if data.ndim == 3:
			for _frame in data:
				self._pipe.stdin.write(_frame.astype("uint16").tostring())
		elif data.ndim == 2:
			self._pipe.stdin.write(data.astype("uint16").tostring())
		else:
			raise RuntimeError("Frames must be 2d or 3d")


	def open_writer(self, filename):
		pipe = subprocess.Popen(self._command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
		self._pipe = pipe


	def close_writer(self):
		self._pipe.stdin.close()