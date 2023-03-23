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
        auto_exposure: bool = False,
        queue=None,
        jumbo_frames: bool = True,
        **kwargs,
    ):

        super(CammyCamera, self).__init__()
        # prelims for using fake camera
        if fake_camera:
            Aravis.set_fake_camera_genicam_filename(os.getenv("FAKE_GENICAM_PATH"))
            Aravis.enable_interface("Fake")

        self.camera = Aravis.Camera.new(id)

        if jumbo_frames and self.camera.is_gv_device():
            self.camera.gv_set_packet_size(8000)
        self.device = self.camera.get_device()
        # self.camera = Aravis.Camera() # THIS IS JUST FOR PYLANCE

        # if not auto_exposure:
        #     try:
        #         self.camera.set_exposure_time_auto(0)
        #         self.camera.set_gain_auto(0)
        #     except (AttributeError, gi.repository.GLib.GError) as e:
        #         print(e)

        self.logger = logging.getLogger(self.__class__.__name__)
        [x, y, width, height] = self.camera.get_region()

        self._payload = self.camera.get_payload()  # size of payload
        self._genicam = genicam = self.device.get_genicam()  # genicam interface
        
        self._width = width
        self._height = height  # stage stream
        self._tick_frequency = 1e9 # TODO: replace with actual tick frequency from gv interface
        self.fps = np.nan
        self.frame_count = 0
        self._last_framegrab = np.nan
        self.id = id
        self.stream = self.camera.create_stream()
        self.queue = queue
        self.missed_frames = 0
        self.total_frames = 0
        for i in range(buffer_size):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(self._payload))

    # https://github.com/SintefManufacturing/python-aravis/blob/master/aravis.py#L162
    def try_pop_frame(self):
        buffer = self.stream.try_pop_buffer()
        if buffer:
            self.total_frames += 1
            status = buffer.get_status()
            if status == Aravis.BufferStatus.TIMEOUT:
                logging.debug("missed frame")
                self.missed_frames += 1
                frame = None
                timestamps = None
            elif status == Aravis.BufferStatus.SIZE_MISMATCH:
                logging.debug("buffer size mismatch")
                self.missed_frames += 1
                frame = None
                timestamps = None
            elif status == Aravis.BufferStatus.SUCCESS:
                frame = self._array_from_buffer_address(buffer)
                timestamp = buffer.get_timestamp()
                system_timestamp = buffer.get_system_timestamp()
                timestamps = {"device_timestamp": timestamp, "system_timestamp": system_timestamp}
                grab_time = system_timestamp
                self.frame_count += 1
                self.fps = 1 / (((grab_time - self._last_framegrab) / self._tick_frequency) + 1e-12)
                self._last_framegrab = grab_time
                if self.queue is not None:
                    self.queue.put((frame, timestamps))
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
        if bits_per_pixel == 8:
            INTP = ctypes.POINTER(ctypes.c_uint8)
        else:
            INTP = ctypes.POINTER(ctypes.c_uint16)
        addr = buffer.get_data()
        ptr = ctypes.cast(addr, INTP)
        im = np.ctypeslib.as_array(ptr, (buffer.get_image_height(), buffer.get_image_width()))
        im = im.copy()
        return im

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

        self.logger.info(f"{name} set to {newval}")

    def get_all_features(self, node_str="Root", return_dct={}):
        node = self._genicam.get_node(node_str)
        if node.get_node_name() == "Category":
            features = node.get_features()
            for _feature in features:
                self.get_all_features(_feature, return_dct=return_dct)
        elif node is not None:
            return_dct[node_str] = self.get_feature(node_str)

        return return_dct
