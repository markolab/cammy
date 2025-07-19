from cammy.record.base import BaseRecord
import subprocess
import numpy as np
import logging
import os

# TODO: update for variable bit depth
class FfmpegVideoRecorder(BaseRecord):
	def __init__(
		self,
		width=600,
		height=420,
		threads=8,
		fps=30,
		pixel_format="gray16le",
		codec="ffv1" ,
		slices=24,
		slicecrc=0,
		filename="test.mkv",
		timestamp_fields=["device_timestamp", "system_timestamp"],
		queue=None,
	):

		super(BaseRecord, self).__init__()
		self.logger = logging.getLogger(self.__class__.__name__)

		command = [
				'nice',
				'-n', '20',
				'ffmpeg',
               '-y',
               '-loglevel', 'fatal',
               '-framerate', str(fps),
               '-f', 'rawvideo',
               '-s', f'{str(width)}x{str(height)}',
               '-pix_fmt', pixel_format,
               '-i', '-',
               '-an',
			#    '-g', '1',
			#    '-context', '1',
               '-vcodec', codec,
               '-threads', str(threads),
               '-slices', str(slices),
               '-slicecrc', str(slicecrc),
               '-r', str(fps),
               filename]
		self.logger.debug(f"ffmpeg command: {command}")
		self._command = command
		self.save_queue = save_queue
		# self.is_running = multiprocessing.Value("i", 0)
		self.id = id
		basefile = os.path.splitext(filename)[0]
		filename_timestamps = f"{basefile}.txt"
		self.filenames = {"video": filename, "timestamps": filename_timestamps}
		self.timestamp_fields = timestamp_fields


	def write_data(self, data):
		vdata, tstamps = data
		if vdata.ndim == 3:
			for _frame in vdata:
				self._pipe.stdin.write(_frame.astype("uint16").tobytes())
		elif vdata.ndim == 2:
			self._pipe.stdin.write(vdata.astype("uint16").tobytes())
		else:
			raise RuntimeError("Frames must be 2d or 3d")
		# print(self._pipe.stdout.read())
		# stderr_output = self._pipe.stderr.read()
		# if len(stderr_output) > 0:
		# 	print(str(stderr_output, "utf-8"))
		if tstamps is not None:
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


class RawVideoRecorder(BaseRecord):
	def __init__(
		self,
		filename="test.dat",
		timestamp_fields=["device_timestamp", "system_timestamp"],
		write_dtype="uint16",
		save_queue=None,
	):

		super(BaseRecord, self).__init__()
		self.logger = logging.getLogger(self.__class__.__name__)

		self.save_queue = save_queue
		self.id = id
		basefile = os.path.splitext(filename)[0]
		filename_timestamps = f"{basefile}.txt"
		
		self.filenames = {"video": filename, "timestamps": filename_timestamps}
		self.timestamp_fields = timestamp_fields
		self.write_dtype = write_dtype


	def write_data(self, vdata, tstamps):
		# vdata, tstamps = data
		# if vdata.ndim == 3:
		# 	for _frame in vdata:
		# 		self._video_file.write(_frame.astype(self.write_dtype).tobytes())
		# elif vdata.ndim == 2:
		# 	self._video_file.write(vdata.astype(self.write_dtype).tobytes())
		# else:
		# 	raise RuntimeError("Frames must be 2d or 3d")
		
		# stack the list into a numpy array and write out...
		if vdata is not None:
			self._video_file.write(np.stack(vdata).astype(self.write_dtype).tobytes())
		
		# similarly, build a big string from the batch and write out...
		if tstamps is not None:
			write_string = ""
			for _tstamps in tstamps: # should be a list
				for _field in self.timestamp_fields:
					write_string += f"{_tstamps[_field]}\t"
				write_string += "\n"
			self._tstamp_file.write(write_string)
		
		# if tstamps is not None:
		# 	for _field in self.timestamp_fields:
		# 		self._tstamp_file.write(f"{tstamps[_field]}\t")
		# 	self._tstamp_file.write("\n")

		# le
		# leads to ill effects after lots of frames pile up
		


	def open_writer(self):
		video_file = open(self.filenames["video"], "wb")
		tstamp_file = open(self.filenames["timestamps"], "w")
		for _field in self.timestamp_fields:
			tstamp_file.write(f"{_field}\t")
		tstamp_file.write("\n")
		self._video_file = video_file
		self._tstamp_file = tstamp_file
		

	def close_writer(self):
		self.logger.info(f"Closing writer {self.name}")
		if len(self.frame_batch) > 0:
			self.write_data(self.frame_batch, self.timestamp_batch)
			# clear the batch lists
			self.frame_batch.clear()
			self.timestamp_batch.clear()
		try:
			self._video_file.flush()
			os.fsync(self._video_file)
			self._video_file.close()
		except AttributeError:
			pass
		try:
			self._tstamp_file.flush()
			os.fsync(self._tstamp_file)
			self._tstamp_file.close()
		except AttributeError:
			pass