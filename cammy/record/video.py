from cammy.record.base import BaseRecord
import subprocess
import numpy as np
import logging
import os


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
		timestamp_fields=["device_timestamp", "system_timestamp"],
		queue=None,
	):

		super(BaseRecord, self).__init__()
		self.logger = logging.getLogger(self.__class__.__name__)

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
		self.logger.debug(f"ffmpeg command: {command}")
		self._command = command
		self.queue = queue
		# self.is_running = multiprocessing.Value("i", 0)
		self.id = id
		basefile, ext = os.path.splitext(filename)
		filename_timestamps = f"{basefile}.txt"
		self.filenames = {"video": filename, "timestamps": filename_timestamps}
		self.timestamp_fields = timestamp_fields


	def write_data(self, data):
		vdata, tstamps = data
		if vdata.ndim == 3:
			for _frame in vdata:
				self._pipe.stdin.write(_frame.astype("uint16").tostring())
		elif vdata.ndim == 2:
			self._pipe.stdin.write(vdata.astype("uint16").tostring())
		else:
			raise RuntimeError("Frames must be 2d or 3d")
		for _field in self.timestamp_fields:
			self._tstamp_file.write(f"{tstamps[_field]}\t")
		self._tstamp_file.write("\n")	


	def open_writer(self):
		pipe = subprocess.Popen(self._command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
		tstamp_file = open(self.filenames["timestamps"], "w")
		for _field in self.timestamp_fields:
			tstamp_file.write(f"{_field}\t")
		tstamp_file.write("\n")
		self._pipe = pipe
		self._tstamp_file = tstamp_file
		

	def close_writer(self):
		self._pipe.stdin.close()
		self._tstamp_file.close()