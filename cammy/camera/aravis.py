import gi
import os
import ctypes
import numpy as np
import logging
import threading
import zmq
from cammy.camera.base import CammyCamera
from typing import Optional

gi.require_version("Aravis", "0.8")
from gi.repository import Aravis


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
            self.camera.gv_set_packet_size(8000)

        if self.camera.is_uv_device():
            self.camera.uv_set_usb_mode(Aravis.UvUsbMode.ASYNC)
        elif self.camera.is_gv_device():
            ext_ids = self.get_feature("GevGVSPExtendedIDMode")
            if ext_ids.lower() == "off":
                self._frame_id_dtype = 16
            else:
                self._frame_id_bit_depth = 64
        else:
            self._frame_id_bit_depth = 32

        self.logger = logging.getLogger(self.__class__.__name__)
        self._tick_frequency = 1e9  # TODO: replace with actual tick frequency from gv interface
        self.fps = np.nan
        self.frame_count = 0
        # self._pixel_format = pixel_format
        self._last_framegrab = np.nan
        # self._last_frame_id = np.nan
        self._spoof_cameras = [] # we use these to push extra images
        self.zmq_publisher = None
        self.id = id

        # this is going to be zmq now...
        self.missed_frames = 0
        self.total_frames = 0
        self.buffer_size = buffer_size
        

    def initialize_acquisition_stream(self):
        self._payload = self.camera.get_payload()  # size of payload
        self.stream = self.camera.create_stream(stream_cb, None)
        for i in range(self.buffer_size):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(self._payload))
        [x, y, width, height] = self.camera.get_region()
        self._width = width
        self._height = height  # stage stream


    # https://github.com/SintefManufacturing/python-aravis/blob/master/aravis.py#L162
    def try_pop_frame(self):
        buffer = self.stream.try_pop_buffer()
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
                
                grab_time = timestamp
                self.frame_count += 1
                new_fps_val = 1 / (((grab_time - self._last_framegrab) / self._tick_frequency) + 1e-12)
                if np.isnan(self.fps):
                    self.fps = new_fps_val
                else:
                    self.fps = .01 * new_fps_val + .99 * self.fps
                
                diff = (timestamps["frame_id"] - self.total_frames) - 1
                if ~np.isnan(diff):
                    self.missed_frames += diff
                self.total_frames = timestamps["frame_id"]
                self._last_framegrab = grab_time
                stream_stats = self.stream.get_statistics()
                self.logger.debug(f"Buffer pressure {stream_stats.n_completed_buffers - timestamps['frame_id']}")
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
            INTP = ctypes.POINTER(ctypes.c_uint8)
            addr = buffer.get_data()
            ptr = ctypes.cast(addr, INTP)
            im = np.ctypeslib.as_array(ptr, (buffer.get_image_height(), buffer.get_image_width()))
            target_array = im.copy(order="C")
        elif (pixel_format in (Aravis.PIXEL_FORMAT_MONO_12, Aravis.PIXEL_FORMAT_MONO_16, Aravis.PIXEL_FORMAT_COORD3D_C_16)):
            INTP = ctypes.POINTER(ctypes.c_uint16)
            addr = buffer.get_data()
            ptr = ctypes.cast(addr, INTP)
            im = np.ctypeslib.as_array(ptr, (buffer.get_image_height(), buffer.get_image_width()))
            target_array = im.copy(order="C")
        # elif (bits_per_pixel == 24) & (self._pixel_format.lower() in ("coord3D_c16y8")):
        #     INTP = ctypes.POINTER(ctypes.c_uint8 * 3) 
        #     addr = buffer.get_data()
        #     ptr = ctypes.cast(addr, INTP)
        #     # return 3 8 bit images, pack first two in 16 bit depth image, last is IR
        #     im = np.ctypeslib.as_array(ptr, (buffer.get_image_height(), buffer.get_image_width()))
        #     im = im.astype("uint16").copy()
        #     im1 = im[:,:,1]<<8 | im[:,:,0]
        #     im2 = im[:,:,2].astype("uint8")
        #     im = (im1, im2)
        else:
            raise RuntimeError(f"No unpacking strategy for {bits_per_pixel} bits with {pixel_format} format")
        
        return target_array

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
        elif ntype == "Integer":
            grab_func = self.device.get_integer_feature_value
        elif ntype == "Float":
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

    def get_all_features(self, node_str="Root", return_dct={}):
        node = self._genicam.get_node(node_str)
        if node.get_node_name() == "Category":
            features = node.get_features()
            for _feature in features:
                self.get_all_features(_feature, return_dct=return_dct)
        elif node is not None:
            return_dct[node_str] = self.get_feature(node_str)

        return return_dct


def stream_cb(user_data, type, buffer):
    if type == Aravis.StreamCallbackType.INIT:
        if not Aravis.make_thread_realtime(10) and \
            not Aravis.make_thread_high_priority(-10):
            print("Failed to make stream thread high priority")


# alllllrighty time for zmq...
def acquisition_loop(camera, shutdown_event, cpu_id=None):
    # NOTE: this is a mission critical loop, do everything to ensure
    # that's it's own cpu with high priority
    import time
    if cpu_id is not None:
        print(f"Setting affinity to {cpu_id}")
        os.sched_setaffinity(0, {int(cpu_id)})

    try:
        os.nice(-10)
    except:
        pass


    while not shutdown_event.is_set():
    
        start_time = time.perf_counter()
        frame, ts = camera.try_pop_frame()

        if frame is not None:
            camera.logger.debug(f"Frame processing time: {time.perf_counter() - start_time}")
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
                camera.zmq_publisher.send_json(metadata, zmq.SNDMORE)
                camera.zmq_publisher.send(frame, copy=False)  # Zero-copy!
                camera.logger.debug(f"Frame processing time after save queue: {time.perf_counter() - start_time}")
            except zmq.error.Again as e:
                camera.logger.debug("Skipping, peer not connected yet...")
            except Exception as e:
                camera.logger.debug(e)