import gi
import os
import ctypes
import numpy as np
import logging
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

        # NOT USING EXT_IDS just yet
        if jumbo_frames and self.camera.is_gv_device():
            self.camera.gv_set_packet_size(8000)
        #     ext_ids = self.get_feature('GevGVSPExtendedIDMode')
        #     if ext_ids.lower() == "off":
        #         self._frame_id_bit_depth = 16
        #     else:
        #         self._frame_id_bit_depth = 64
        # else:
        #     self._frame_id_bit_depth = 32
        self.device = self.camera.get_device()
        # self.camera = Aravis.Camera() # THIS IS JUST FOR PYLANCE

        self.logger = logging.getLogger(self.__class__.__name__)
        [x, y, width, height] = self.camera.get_region()

        self._payload = self.camera.get_payload()  # size of payload
        self._genicam = self.device.get_genicam()  # genicam interface

        self._width = width
        self._height = height  # stage stream
        self._tick_frequency = 1e9  # TODO: replace with actual tick frequency from gv interface
        self.fps = np.nan
        self.frame_count = 0
        self._pixel_format = pixel_format
        self._last_framegrab = np.nan
        self._spoof_cameras = [] # we use these to push extra images

        self.id = id

        counter_names = [
            "_".join(self.get_counter_parameters(i).values()) for i in range(record_counters)
        ]
        if len(counter_names) > 0:
            self._counters = {_counter: f"Counter{i}" for i, _counter in enumerate(counter_names)}
            user_data = UserData(counters=counter_names, arv_obj=self)
            self.stream = self.camera.create_stream(callback, user_data)
        else:
            self._counters = {}
            self.stream = self.camera.create_stream()

        self.save_queue = save_queue
        self.missed_frames = 0
        self.total_frames = 0
        for i in range(buffer_size):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(self._payload))

    # https://github.com/SintefManufacturing/python-aravis/blob/master/aravis.py#L162
    def try_pop_frame(self):
        buffer = self.stream.try_pop_buffer()
        if buffer:
            self.total_frames += 1
            
            # can potentially use this, are bit depths handled automatically by aravis? 
            # if all come back as uint64s need to rethink...
            # print(buffer.get_frame_id())
            status = buffer.get_status()
            if status == Aravis.BufferStatus.TIMEOUT:
                self.logger.debug("missed frame")
                self.missed_frames += 1
                frame = None
                timestamps = None
            elif status == Aravis.BufferStatus.SIZE_MISMATCH:
                self.logger.debug("buffer size mismatch")
                self.missed_frames += 1
                frame = None
                timestamps = None
            elif status == Aravis.BufferStatus.SUCCESS:
                frame = self._array_from_buffer_address(buffer)
                timestamp = buffer.get_timestamp()
                system_timestamp = buffer.get_system_timestamp()
                timestamps = {
                    "capture_number": self.total_frames,
                    "device_timestamp": timestamp,
                    "system_timestamp": system_timestamp,
                }
                if isinstance(frame, tuple):
                    for _frame, _cam in zip(frame[1:], self._spoof_cameras):
                        # send _frame and timestamps...        
                        _cam.recv_queue.put((_frame, timestamps))
                    # now proceed as if we only collected the first...
                    frame = frame[0]
                
                grab_time = system_timestamp
                self.frame_count += 1
                self.fps = 1 / (((grab_time - self._last_framegrab) / self._tick_frequency) + 1e-12)
                # self.smooth_fps = (1 - self.fps_alpha) * self.fps + self.fps_alpha * self.prior_fps
                # self.prior_fps = self.smooth_fps

                self._last_framegrab = grab_time
                # user_data = buffer.get_user_data()
                # for k, v in user_data.counter_data.items():
                #     timestamps[k] = v
                if self.save_queue is not None:
                    self.save_queue.put((frame, timestamps))
            else:
                raise RuntimeError(f"Did not understand status: {status}")
            self.stream.push_buffer(buffer)
            return frame, timestamps
        else:
            return None, None

    # https://github.com/SintefManufacturing/python-aravis/blob/master/aravis.py#L180
    def _array_from_buffer_address(self, buffer):
        if not buffer:
            return None
        pixel_format = buffer.get_image_pixel_format()
        bits_per_pixel = pixel_format >> 16 & 0xFF
        if (bits_per_pixel == 8) & (self._pixel_format == "Mono8"):
            INTP = ctypes.POINTER(ctypes.c_uint8)
            addr = buffer.get_data()
            ptr = ctypes.cast(addr, INTP)
            im = np.ctypeslib.as_array(ptr, (buffer.get_image_height(), buffer.get_image_width()))
            im = im.copy()
        elif (bits_per_pixel == 16) & (self._pixel_format in ("Mono16", "Coord3D_C16")):
            INTP = ctypes.POINTER(ctypes.c_uint16)
            addr = buffer.get_data()
            ptr = ctypes.cast(addr, INTP)
            im = np.ctypeslib.as_array(ptr, (buffer.get_image_height(), buffer.get_image_width()))
            im = im.copy()
        elif (bits_per_pixel == 24) & (self._pixel_format in ("Coord3D_C16Y8")):
            INTP = ctypes.POINTER(ctypes.c_uint8 * 3) 
            addr = buffer.get_data()
            ptr = ctypes.cast(addr, INTP)
            # return 3 8 bit images, pack first two in 16 bit depth image, last is IR
            im = np.ctypeslib.as_array(ptr, (buffer.get_image_height(), buffer.get_image_width()))
            im = im.astype("uint16").copy()
            im1 = im[:,:,1]<<8 | im[:,:,0]
            im2 = im[:,:,2].astype("uint8")
            im = (im1, im2)
        else:
            raise RuntimeError(f"No unpacking strategy for {bits_per_pixel} bits with {self._pixel_format} format")
        
        return im

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


class UserData:
    def __init__(self, counters: Optional[dict], arv_obj: AravisCamera) -> None:
        self.counters = counters
        self.counter_data = {}
        self.camera = arv_obj
        # need the aravis object to grab counter values...


def callback(user_data, cb_type, buffer):
    if buffer is not None:
        for k, v in user_data.counters.items():
            user_data.counter_data[v] = user_data.camera.get_counter_value[k]
