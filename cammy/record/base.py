import logging
import threading
import os
import zmq

# Per socket: 64 frames at 2048x2048 Mono8 is 256 MiB.
RECORD_QUEUE_FRAMES = 64

# simple data writer, should be general enough to take 1d/2d/etc. data
class BaseRecord(threading.Thread):
	def __init__(self, save_queue, filename, cpu_id=None, batch_size=16, zmq_address=None,
				 queue_frames=RECORD_QUEUE_FRAMES):
		super().__init__()
		self.logger = logging.getLogger(self.__class__.__name__)
		self.save_queue = save_queue
		self.filename=filename
		self.frame_batch = []
		self.timestamp_batch = []
		self.batch_size = batch_size
		self.cpu_id = cpu_id
		self.zmq_address = zmq_address
		self.queue_frames = queue_frames
		self.frames_written = 0
		self.error = None
		self.ready = threading.Event()
		self._stop_requested = threading.Event()
		self.zmq_socket = None

	def stop(self):
		"""After producers stop, drain queued frames and finish the recording."""
		self._stop_requested.set()

	def write_data(self, data):
		pass


	def write_batch(self, frames, timestamps):
		self.write_data(b''.join(frames), timestamps)


	def open_writer(self):
		pass


	def close_writer(self):
		pass


	def run(self):
		try:
			if self.cpu_id is not None:
				os.sched_setaffinity(0, {int(self.cpu_id)})
				self.logger.info("%s writer TID %d pinned to CPU %d", self.filename, threading.get_native_id(), self.cpu_id)

			self.zmq_context = zmq.Context.instance()
			self.zmq_socket = self.zmq_context.socket(zmq.PULL)
			self.zmq_socket.setsockopt(zmq.RCVHWM, self.queue_frames)
			self.zmq_socket.connect(self.zmq_address)
			self.open_writer()
			self.ready.set()

			while True:
				has_frame = self.zmq_socket.poll(100, zmq.POLLIN)
				if has_frame:
					metadata = self.zmq_socket.recv_json()
					self.frame_batch.append(self.zmq_socket.recv(copy=False))
					self.timestamp_batch.append(metadata["timestamps"])
				elif not self._stop_requested.is_set():
					continue

				if self.frame_batch and (len(self.frame_batch) >= self.batch_size or not has_frame):
					# Never replay a failed batch: pixels may already be partly written.
					self.write_batch(self.frame_batch, self.timestamp_batch)
					self.frames_written += len(self.frame_batch)
					self.frame_batch.clear()
					self.timestamp_batch.clear()
				if not has_frame:
					break
		except BaseException as error:
			self.error = error
			raise
		finally:
			try:
				self.close_writer()
			except BaseException as error:
				if self.error is None:
					self.error = error
					raise
				self.logger.exception("Could not finish closing %s", self.filename)
			finally:
				if self.zmq_socket is not None:
					self.zmq_socket.close(linger=0)
				self.ready.set()
