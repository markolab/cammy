"""Bounded two-thread recording for fixed-size raw camera frames."""

from collections import deque
import threading
import time

import numpy as np
import zmq

from cammy.record.video import RawVideoRecorder


class BufferedRawVideoRecorder(RawVideoRecorder):
    worker_role = "recording receiver"

    def __init__(self, *, frame_bytes, buffer_bytes, **kwargs):
        super().__init__(**kwargs)
        if frame_bytes <= 0:
            raise ValueError("Raw frame size must be positive")
        self.frame_bytes = frame_bytes
        self.capacity = buffer_bytes // frame_bytes
        if not 1 <= self.batch_size <= self.capacity:
            raise ValueError(f"Writer batch of {self.batch_size} frames must fit in the "
                             f"{self.capacity}-frame recording buffer")
        self._condition = threading.Condition()
        self._received = 0
        self._timestamps = deque()
        self._receiver_done = False
        self._peak = 0
        self._full_waits = 0
        self._last_write_ms = self._max_write_ms = 0.0

    def buffer_statistics(self):
        with self._condition:
            return (f" writer_buffer_frames={self._received - self.frames_written}/{self.capacity}"
                    f" writer_buffer_peak={self._peak} writer_full_waits={self._full_waits}"
                    f" writer_last_ms={self._last_write_ms:.1f} writer_max_ms={self._max_write_ms:.1f}")

    def receive_frames(self):
        pixels = np.empty((self.capacity, self.frame_bytes), dtype=np.uint8)
        pixels.fill(0)  # Touch the pages before acquisition starts.
        writer = threading.Thread(target=self._write_frames, args=(pixels,))
        writer.start()
        try:
            self.ready.set()
            while True:
                has_frame = self.zmq_socket.poll(100, zmq.POLLIN)
                with self._condition:
                    if has_frame and self._received - self.frames_written == self.capacity:
                        self._full_waits += 1
                        self._condition.wait_for(lambda: self.error is not None or
                                                 self._received - self.frames_written < self.capacity)
                    if self.error is not None:
                        raise self.error
                if not has_frame:
                    if self._stop_requested.is_set() and not self.zmq_socket.poll(0, zmq.POLLIN):
                        break
                    continue

                timestamps = self.zmq_socket.recv_json()["timestamps"]
                frame = self.zmq_socket.recv(copy=False)
                if len(frame) != self.frame_bytes:
                    raise ValueError(f"Expected {self.frame_bytes} raw frame bytes, received {len(frame)}")
                # NumPy can release the GIL during the full-frame copy.
                np.copyto(pixels[self._received % self.capacity], np.frombuffer(frame.buffer, dtype=np.uint8))
                del frame
                # Only the receiver publishes slots. Only the writer releases them.
                with self._condition:
                    self._timestamps.append(timestamps)
                    self._received += 1
                    self._peak = max(self._peak, self._received - self.frames_written)
                    if len(self._timestamps) == self.batch_size:
                        self._condition.notify_all()
        finally:
            with self._condition:
                self._receiver_done = True
                self._condition.notify_all()
            writer.join()
        if self.error is not None:
            raise self.error

    def _write_frames(self, pixels):
        try:
            if self.cpu_id is not None:
                # The disk thread inherits the recording receiver's CPU affinity.
                self.logger.info("%s writer TID %d pinned to CPU %d",
                                 self.filename, threading.get_native_id(), self.cpu_id)
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: len(self._timestamps) >= self.batch_size or
                                             self._receiver_done)
                    count = min(len(self._timestamps), self.batch_size)
                    if not count:
                        return
                    timestamps = [self._timestamps.popleft() for _ in range(count)]
                frames = [pixels[(self.frames_written + i) % self.capacity] for i in range(count)]
                start = time.monotonic()
                self.write_batch(frames, timestamps)
                elapsed_ms = (time.monotonic() - start) * 1000
                with self._condition:
                    self.frames_written += count
                    self._last_write_ms = elapsed_ms
                    self._max_write_ms = max(self._max_write_ms, elapsed_ms)
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self.error = exc
                self._condition.notify_all()
