import gi
import os
import numpy as np
import logging
import threading
import sys
import zmq
from cammy.camera.base import CammyCamera
from cammy.roi import set_region
from pathlib import Path
from typing import Optional

gi.require_version("Aravis", "0.8")
from gi.repository import Aravis


def set_receiver_realtime_priority(priority):
    """Set priority once for this process's native GigE receiver threads."""
    if priority == 0:
        return
    logger = logging.getLogger(__name__)
    if sys.platform != "linux":
        logger.warning("Receiver realtime priority requires Linux; keeping existing scheduling")
        return

    tids = []
    try:
        for task in Path("/proc/self/task").iterdir():
            try:
                if (task / "comm").read_text().strip() == "arv_gv_stream":
                    tids.append(int(task.name))
            except FileNotFoundError:
                continue  # A thread may exit while we enumerate it.
    except OSError as exc:
        logger.warning("Could not find Aravis receiver threads: %s", exc)
        return
    if not tids:
        logger.warning("No arv_gv_stream receiver threads found; realtime priority was not applied")

    for tid in sorted(tids):
        try:
            os.sched_setscheduler(tid, os.SCHED_RR, os.sched_param(priority))
        except OSError as exc:
            logger.warning("Could not set Aravis receiver TID %d to SCHED_RR priority %d: %s; "
                           "keeping existing scheduling", tid, priority, exc)
        else:
            logger.info("Aravis receiver TID %d: SCHED_RR priority %d", tid, priority)


# TODO:
# 1) Get data from counters and append to timestamp file
class AravisCamera(CammyCamera):
    def __init__(
        self,
        id: Optional[str],
        buffer_size: int = 1000,
        fake_camera: bool = False,
        save_queue=None,
        jumbo_frames: bool = True,
        record_counters: int = 0,
        pixel_format: str = "MONO16",
        fps_tau: float = 5,
        fps_stride: int = 1,
        **kwargs,
    ):
        super(CammyCamera, self).__init__()
        # prelims for using fake camera
        if fake_camera:
            Aravis.set_fake_camera_genicam_filename(os.getenv("FAKE_GENICAM_PATH"))
            Aravis.enable_interface("Fake")

        self.camera = Aravis.Camera.new(id)
        self.display_lock = threading.Lock()
        self.display_frame = (None, None)
        # Aravis.make_thread_high_priority(1)

        self.device = self.camera.get_device()
        self._genicam = self.device.get_genicam()  # genicam interface

        print(id)
        print(self.camera.is_uv_device())
        # NOT USING EXT_IDS just yet
        if jumbo_frames and self.camera.is_gv_device():
            self.camera.gv_set_packet_size(8192)

        self._frame_id_modulus = None
        if self.camera.is_uv_device():
            self.camera.uv_set_usb_mode(Aravis.UvUsbMode.ASYNC)
        elif self.camera.is_gv_device():
            ext_ids = self.get_feature("GevGVSPExtendedIDMode")
            if ext_ids.lower() == "off":
                # Legacy GigE IDs run from 1 through 65535, skipping zero.
                self._frame_id_modulus = 65535

        self.logger = logging.getLogger(self.__class__.__name__)
        self._tick_frequency = 1e9  # TODO: replace with actual tick frequency from gv interface
        self.fps = np.nan
        self.frame_count = 0
        # self._pixel_format = pixel_format
        self._last_framegrab = [np.nan] * fps_stride
        self._last_frame_id = None
        self._spoof_cameras = [] # we use these to push extra images
        self.zmq_publisher = None
        self.id = id

        # this is going to be zmq now...
        self.missed_frames = 0
        self.total_frames = 0
        self.buffer_size = buffer_size
        

    def initialize_acquisition_stream(self):
        requested_packet_size = (
            self.camera.gv_get_packet_size() if self.camera.is_gv_device() else None
        )
        # Keep packet reception out of Python callbacks and the GIL.
        self.stream = self.camera.create_stream(None, None)
        if requested_packet_size is not None:
            packet_size = self.camera.gv_get_packet_size()
            self.logger.info("%s GigE packet size: %d bytes", self.id, packet_size)
            if packet_size != requested_packet_size:
                self.logger.warning(
                    "%s GigE packet size changed from %d to %d bytes during stream setup; "
                    "check NIC/switch jumbo-frame and MTU settings",
                    self.id, requested_packet_size, packet_size,
                )
        self._allocate_stream_buffers()

    def _allocate_stream_buffers(self):
        self._payload = self.camera.get_payload()
        for _ in range(self.buffer_size):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(self._payload))
        x, y, self._width, self._height = self.camera.get_region()
        self.logger.info("%s ROI readback: Width=%d Height=%d OffsetX=%d OffsetY=%d (pixels)",
                         self.id, self._width, self._height, x, y)

    def change_region(self, region):
        """Called with acquisition stopped and no reader holding a buffer."""
        self.stream.stop_thread(True)  # Discard queued frames and free the old pool first.
        try:
            set_region(self.camera, region)
            self._allocate_stream_buffers()
        finally:
            self.stream.start_thread()
        self.display_frame = (None, None)
        self._last_frame_id = None
        self._last_framegrab = [np.nan] * len(self._last_framegrab)
        self.fps = np.nan
        for child in self._spoof_cameras:
            child._width, child._height = self._width, self._height
            child.display_frame = (None, None)
            while not child.recv_queue.empty():
                child.recv_queue.get_nowait()


    def reset_tune_trial(self, fps_stride):
        """Start a new measurement with the trigger stopped and acquisition reader paused."""
        while (buffer := self.stream.try_pop_buffer()) is not None:
            self.stream.push_buffer(buffer)
        self.display_frame = (None, None)
        self.frame_count = self.missed_frames = self.total_frames = 0
        self._last_frame_id = None
        self._last_framegrab = [np.nan] * fps_stride
        self.fps = np.nan

    def log_stream_statistics(self):
        """Sample from the GUI thread, independently of frame acquisition."""
        free, queued = self.stream.get_n_buffers()
        # GvStream.get_statistics() shadows the base method with packet counters.
        completed, failures, underruns = Aravis.Stream.get_statistics(self.stream)
        packet_stats = ""
        if isinstance(self.stream, Aravis.GvStream):
            resent, missing = Aravis.GvStream.get_statistics(self.stream)
            packet_stats = f" resent_packets={resent} missing_packets={missing}"
        writer = getattr(self, "recorder", None)
        self.logger.info(
            "%s stream: free_buffers=%d queued_buffers=%d completed=%d failures=%d "
            "underruns=%d acquired=%d missed=%d written=%d recording_drops=%d%s",
            self.id, free, queued, completed, failures, underruns,
            self.frame_count, self.missed_frames, getattr(writer, "frames_written", 0),
            getattr(self, "recording_drops", 0), packet_stats,
        )


    # https://github.com/SintefManufacturing/python-aravis/blob/master/aravis.py#L162
    # added timeout in ms
    def try_pop_frame(self, timeout=50000):
        # buffer = self.stream.try_pop_buffer()
        buffer = self.stream.timeout_pop_buffer(timeout)
        if buffer:
            # self.total_frames += 1
            # can potentially use this, are bit depths handled automatically by aravis? 
            # if all come back as uint64s need to rethink...
            status = buffer.get_status()
            # print(status)
            if status == Aravis.BufferStatus.TIMEOUT:
                self.logger.debug("missed frame")
                # self.missed_frames += 1
                frame = None
                timestamps = None
                self.stream.push_buffer(buffer)
            elif status == Aravis.BufferStatus.SIZE_MISMATCH:
                self.logger.debug("buffer size mismatch")
                # self.missed_frames += 1
                frame = None
                timestamps = None
                self.stream.push_buffer(buffer)
            elif status == Aravis.BufferStatus.SUCCESS:
                frame = self._array_from_buffer_address(buffer)
                timestamp = buffer.get_timestamp()
                system_timestamp = buffer.get_system_timestamp()
                timestamps = {
                    "capture_number": self.total_frames,
                    "device_timestamp": timestamp,
                    "system_timestamp": system_timestamp,
                    "frame_id": buffer.get_frame_id(),
                } 
                self.stream.push_buffer(buffer)
                if isinstance(frame, tuple):
                    for _frame, _cam in zip(frame[1:], self._spoof_cameras):
                        # send _frame and timestamps...        
                        _cam.recv_queue.put((_frame, timestamps))
                    # now proceed as if we only collected the first...
                    frame = frame[0]
                
                frame_id = timestamps["frame_id"]
                advance = 1
                if self._last_frame_id is not None:
                    advance = frame_id - self._last_frame_id
                    if self._frame_id_modulus is not None:
                        advance %= self._frame_id_modulus
                        if advance > self._frame_id_modulus // 2:
                            advance = 0  # Backward/reset ID, rather than a forward gap.
                    if advance <= 0:
                        advance = 1
                        self._last_framegrab = [np.nan] * len(self._last_framegrab)
                self._last_frame_id = frame_id
                self.missed_frames += advance - 1
                self.total_frames += advance

                grab_time = timestamp
                self.frame_count += 1
                # In alternate mode, compare timestamps from the same light bank.
                fps_index = (self.total_frames - 1) % len(self._last_framegrab)
                last_grab = self._last_framegrab[fps_index]
                if grab_time > last_grab:
                    new_fps_val = self._tick_frequency / (grab_time - last_grab)
                    if np.isnan(self.fps):
                        self.fps = new_fps_val
                    else:
                        self.fps = .01 * new_fps_val + .99 * self.fps
                self._last_framegrab[fps_index] = grab_time
                
            else:
                raise RuntimeError(f"Did not understand status: {status}")
            #self.stream.push_buffer(buffer)
            return frame, timestamps
        else:
            return None, None

    # https://github.com/SintefManufacturing/python-aravis/blob/master/aravis.py#L180
    def _array_from_buffer_address(self, buffer):
        if not buffer:
            return None

        pixel_format = buffer.get_image_pixel_format()
        bits_per_pixel = pixel_format >> 16 & 0xFF
        if pixel_format == Aravis.PIXEL_FORMAT_MONO_8:
            dtype = np.uint8
        elif (pixel_format in (Aravis.PIXEL_FORMAT_MONO_12, Aravis.PIXEL_FORMAT_MONO_16, Aravis.PIXEL_FORMAT_COORD3D_C_16)):
            dtype = np.uint16
        else:
            raise RuntimeError(f"No unpacking strategy for {bits_per_pixel} bits with {pixel_format} format")

        height, width = buffer.get_image_height(), buffer.get_image_width()
        data = buffer.get_data()
        row_bytes = width * (bits_per_pixel // 8)
        if len(data) > height * row_bytes and hasattr(buffer, "get_image_padding"):
            x_padding, _ = buffer.get_image_padding()
            if x_padding:
                # Cropped rows may be padded on the wire. The raw file contains pixels only.
                return np.ndarray((height, width), dtype=dtype, buffer=data,
                                  strides=(row_bytes + x_padding, bits_per_pixel // 8)).copy()
        frame = np.frombuffer(data, dtype=dtype, count=height * width).reshape(height, width)
        # PyGObject returns an owned bytes snapshot; NumPy keeps it alive after
        # the Aravis buffer is recycled. Copy only if the binding returns mutable data.
        return frame if isinstance(data, bytes) else frame.copy(order="C")

    def get_counter_parameters(self, counter_num):
        param_names = ["CounterEventSource", "CounterEventActivation"]
        self.set_feature("CounterSelector", f"Counter{counter_num}")
        params = {key: self.get_feature(key) for key in param_names}
        return params

    def get_counter_value(self, counter_num):
        if isinstance(counter_num, int):
            self.set_feature("CounterSelector", f"Counter{counter_num}")
        elif isinstance(counter_num, str):
            self.set_feature("CounterSelector", counter_num)
        else:
            raise RuntimeError(f"Did not understand counter {counter_num}")

        return self.get_feature("CounterValue")

    # https://github.com/SintefManufacturing/python-aravis/blob/master/aravis.py#L79
    def get_feature_type(self, name):
        # genicam = self.device.get_genicam()
        node = self._genicam.get_node(name)
        if not node:
            raise RuntimeWarning("Feature {} does not seem to exist in camera".format(name))
        return node.get_node_name()

    def get_feature(self, name):
        """
        return value of a feature. independantly of its type
        """
        ntype = self.get_feature_type(name)

        if ntype in ("Enumeration", "String", "StringReg"):
            grab_func = self.device.get_string_feature_value
        elif ntype in ("Integer", "IntReg", "IntConverter", "IntSwissKnife"):
            grab_func = self.device.get_integer_feature_value
        elif ntype in ("Float", "FloatReg", "SwissKnife"):
            grab_func = self.device.get_float_feature_value
        elif ntype == "Boolean":
            grab_func = self.device.get_boolean_feature_value
        elif ntype == "Converter":
            grab_func = lambda x: self._genicam.get_node(x).get_value()
        else:
            self.logger.debug("Feature type not implemented: %s", ntype)
            return None

        try:
            return grab_func(name)
        except Exception as e:
            self.logger.debug(e)
            return None


    def set_feature(self, name, val):
        """
        set value of a feature
        """
        ntype = self.get_feature_type(name)
        if ntype in ("String", "Enumeration", "StringReg"):
            status = self.device.set_string_feature_value(name, val)
            newval = self.get_feature(name)
        elif ntype == "Integer":
            status = self.device.set_integer_feature_value(name, int(val))
            newval = self.get_feature(name)
        elif ntype == "Float":
            status = self.device.set_float_feature_value(name, float(val))
            newval = self.get_feature(name)
        elif ntype == "Boolean":
            status = self.device.set_boolean_feature_value(name, int(val))
            newval = self.get_feature(name)
        elif ntype == "Converter":
            node = self._genicam.get_node(name)
            status = node.set_value_from_string(val)
            newval = node.get_value_as_string()
        else:
            self.logger.debug("Feature type not implemented: %s", ntype)
            status = None
            newval = None

        self.logger.info(f"{self.id} {name} set to {newval}")

    def get_all_features(self, node_str="Root", return_dct=None):
        if return_dct is None:
            return_dct = {}
        node = self._genicam.get_node(node_str)
        if node.get_node_name() == "Category":
            features = node.get_features()
            for _feature in features:
                self.get_all_features(_feature, return_dct=return_dct)
        elif node is not None:
            return_dct[node_str] = self.get_feature(node_str)

        return return_dct


# alllllrighty time for zmq...
def acquisition_loop(camera, shutdown_event, cpu_id=None):
    # NOTE: this is a mission critical loop, do everything to ensure
    # that's it's own cpu with high priority
    import time
    camera.recording_drops = 0
    debug_enabled = camera.logger.isEnabledFor(logging.DEBUG)
    if cpu_id is not None:
        print(f"Setting affinity to {cpu_id}")
        os.sched_setaffinity(0, {int(cpu_id)})

    try:
        os.nice(-10)
    except:
        pass


    while not shutdown_event.is_set():
    
        start_time = time.perf_counter() if debug_enabled else None
        frame, ts = camera.try_pop_frame()

        if frame is not None and debug_enabled:
            camera.logger.debug("Frame processing time: %s", time.perf_counter() - start_time)
        # print(ts)
        if (frame is not None) and (camera.zmq_publisher is None):
            # with camera.display_lock:
            camera.display_frame = (frame, ts)
        elif (frame is not None) and (camera.zmq_publisher is not None):  
            camera.display_frame = (frame, ts)

            metadata = {
                'timestamps': ts,
                'shape': frame.shape,
                'dtype': str(frame.dtype)
            }
            
            # FOR THE FUTURE:
            # it's possible we just want a memoryview of the frame...
            try:
                # Buffer normally; skip this recording frame if the queue is full.
                camera.zmq_publisher.send_json(metadata, zmq.SNDMORE | zmq.NOBLOCK)
                camera.zmq_publisher.send(frame, flags=zmq.NOBLOCK, copy=False)
                if debug_enabled:
                    camera.logger.debug("Frame processing time after save queue: %s", time.perf_counter() - start_time)
            except zmq.Again:
                camera.recording_drops += 1
            except Exception as e:
                camera.logger.debug(e)
