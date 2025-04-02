import gi
import multiprocessing
import cv2
import numpy as np
import logging
import sys

gi.require_version("Aravis", "0.8")
from gi.repository import Aravis


logger = logging.getLogger(__name__)


def intrinsics_file_to_cv2(intrinsics_file):
    import toml

    cam_intrinsics = toml.load(intrinsics_file)
    intrinsic_matrix = {}
    distortion_coeffs = {}
    for k, v in cam_intrinsics.items():
        intrinsic_matrix[k] = np.array(
            [
                [v["CalibFocalLengthX"], 0, v["CalibOpticalCenterX"]],
                [0, v["CalibFocalLengthY"], v["CalibOpticalCenterY"]],
                [0, 0, 1],
            ]
        )
        distortion_coeffs[k] = np.array([v["k1"], v["k2"], v["p1"], v["p2"], v["k3"]])
    return intrinsic_matrix, distortion_coeffs


def get_all_camera_ids(interface="aravis", prefix="Lucid", n_cams=1):
    if (interface == "aravis") or (interface == "all"):
        Aravis.update_device_list()
        n_cams = Aravis.get_n_devices()
        use_ids = [Aravis.get_device_id(i) for i in range(n_cams)]
    elif (interface == "fake") or (interface == "fake_custom"):
        use_ids = {f"Fake_{i + 1}" for i in range(n_cams)}
    else:
        raise RuntimeError(f"Did not understand interface {interface}")
    ids = {_id: interface for _id in use_ids}
    if prefix is not None:
        ids = {k: v for k, v in ids.items() if prefix in k}
    return ids


def get_pixel_format_bit_depth(pixel_format):
    spoof_ims = {}
    if pixel_format.lower() in ("mono16", "coord3d_c16"):
        bit_depth = 16
    elif pixel_format.lower() in ("coord3D_c16y8"):
        # 24 bit, first 16 bits are depth, last 8 bits are IR
        bit_depth = 16
        spoof_ims = {"ir": 8}  # spoof multiple cameras...
    elif pixel_format.lower() == "mono12":
        bit_depth = 12
    elif pixel_format.lower() == "mono8":
        bit_depth = 8
    else:
        raise RuntimeError(f"Did not understand pixel format {pixel_format}")

    return bit_depth, spoof_ims


def get_queues(ids=None) -> dict:
    if ids:
        queues = {}
        queues["display"] = {id: multiprocessing.Manager().Queue(100) for id in ids}
        queues["storage"] = {id: multiprocessing.Manager().Queue(100) for id in ids}
        return queues
    else:
        raise RuntimeError("Must specify IDs to construct queues")


# https://stackoverflow.com/questions/52498777/apply-matplotlib-or-custom-colormap-to-opencv-image
def mpl_to_cv2_colormap(cmap_name):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # step from min to max, strip alpha, rgb to bgr
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    return color_range.reshape(256, 1, 3)


def intensity_to_rgba(frame, minval=1800, maxval=2200, colormap=cv2.COLORMAP_TURBO):
    new_frame = np.ones((frame.shape[0], frame.shape[1], 4))
    disp_frame = frame.copy().astype("float")
    disp_frame -= minval
    disp_frame[disp_frame < 0] = 0
    disp_frame /= np.abs(maxval - minval)
    disp_frame[disp_frame >= 1] = 1
    disp_frame *= 255
    bgr_frame = cv2.applyColorMap(disp_frame.astype(np.uint8), colormap)
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    new_frame[:, :, :3] = rgb_frame / 255.0
    return new_frame


def initialize_camera(id, interface: str, config={}, **kwargs):
    if (interface == "aravis") or (interface == "all"):
        from cammy.camera.aravis import AravisCamera

        cam = AravisCamera(id=id, **kwargs)
        for k, v in config.items():
            cam.set_feature(k, v)
    elif interface == "fake_custom":
        from cammy.camera.fake import FakeCamera

        cam = FakeCamera(id=id)
    else:
        raise RuntimeError(f"Did not understand interface {interface}")
    return cam


def initialize_cameras(ids, configs, **kwargs):
    cameras = {}
    for _id, _interface in ids.items():
        use_config = {}
        if "genicam" in configs.keys():
            for k, v in configs["genicam"].items():
                for k2, v2 in v.items():
                    if k in _id:
                        use_config = {**use_config, **v2}
        cameras[_id] = initialize_camera(_id, _interface, use_config, **kwargs)

    return cameras


def get_output_format(save_engine, bit_depth):
    write_dtype = {}
    if save_engine == "ffmpeg":
        # TODO: update and test dtypes per cam and support 8/12 bit
        for k, v in bit_depth.items():
            if v == 16:
                write_dtype[k] = "gray16le"
            elif v == 12:
                write_dtype[k] = "gray12le"
            elif v == 8:
                write_dtype[k] = "gray10le"  # AFAIK ffv1 only supported down to 10 bit
            else:
                raise RuntimeError(f"{k}: Did not recognize bit depth {v}")
        codec = "ffv1"
    elif save_engine == "raw":
        for k, v in bit_depth.items():
            if v == 16:
                write_dtype[k] = "uint16"
            elif v == 12:
                write_dtype[k] = "uint16"
            elif v == 8:
                write_dtype[k] = "uint8"
            else:
                raise RuntimeError(f"{k}: Did not recognize bit depth {v}")
        codec = "n/a"
    else:
        raise RuntimeError(f"Did not understand save engine {save_engine}")

    return write_dtype, codec


def check_counters_equal(ncounters, cameras):
    for i in range(ncounters):
        counter_metadata = {}
        for _id, _cam in cameras.items():
            counter_metadata[_id] = _cam.get_counter_parameters(i)

        for k, v in counter_metadata.items():
            for k2, v2 in counter_metadata.items():
                # skip self-checks
                if k2 == k:
                    continue
                if v != v2:
                    raise RuntimeError(
                        f"Counter settings for {k} and {k2} not equivalent: {v} != {v2}"
                    )


def get_single_buffer(config):
    import toml

    configs = toml.load(config)
    use_config = {}
    id = list(get_all_camera_ids("aravis").keys())[0]
    for k, v in configs["genicam"].items():
        for k2, v2 in v.items():
            if k in id:
                use_config = {**use_config, **v2}
    camera = initialize_camera(id, "aravis", config=use_config)
    camera.start_acquisition()

    buffer = None
    while not buffer:
        buffer = camera.stream.try_pop_buffer()
        if buffer:
            status = buffer.get_status()
            if status == Aravis.BufferStatus.SUCCESS:
                break
            else:
                buffer = None

    camera.stop_acquisition()
    return buffer
