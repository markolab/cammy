import gi
import cv2
import numpy as np
import logging
import os
import sys
from functools import lru_cache
from importlib.metadata import distribution
from pathlib import Path

gi.require_version("Aravis", "0.8")
from gi.repository import Aravis


logger = logging.getLogger(__name__)


def shared_asset(relative):
    """Locate a launcher or GUI asset in a checkout or the installed wheel."""
    path = Path(__file__).resolve().parents[1] / relative
    if path.is_file():
        return path
    package = distribution("cammy")
    for item in package.files or ():
        if item.parts[-3:] == ("share", "cammy", path.name):
            return Path(package.locate_file(item))
    raise FileNotFoundError(f"Cammy asset missing: {relative}")


def get_physical_core_map():
    from collections import defaultdict

    physical_cores = defaultdict(list)
    # Sibling CPU lists identify physical cores across sockets and SMT layouts.
    # Only assign CPUs allowed by the current taskset/container affinity.
    for cpu in sorted(os.sched_getaffinity(0)):
        with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
            physical_cores[f.read().strip()].append(cpu)
    return physical_cores


def get_cpu_affinity(record=False, writer_cores=None):
    """Give GUI one physical core and split the rest into disjoint worker pools."""
    import platform

    if platform.system().lower() != "linux":
        if writer_cores is not None:
            logger.warning("--writer-cores only applies on Linux")
        return None, [None], [None]

    cores = list(get_physical_core_map().values())
    minimum = 3 if record else 2
    if len(cores) < minimum:
        raise ValueError(
            f"Need at least {minimum} available physical cores for separate CPU "
            f"affinity pools; found {len(cores)}"
        )

    gui_cpus = set(cores[0])
    workers = cores[1:]
    if record:
        if writer_cores is None:
            writer_cores = len(workers) // 2
        if not 1 <= writer_cores < len(workers):
            raise ValueError(
                f"--writer-cores must be between 1 and {len(workers) - 1} "
                f"with {len(cores)} available physical cores"
            )
        acquisition, recording = workers[:-writer_cores], workers[-writer_cores:]
    else:
        acquisition, recording = workers, []

    # Use one CPU per physical core before using its sibling hardware threads.
    # Cameras can reuse CPUs within a pool without sharing cores across roles.
    acquire_cpus = [core[0] for core in acquisition] + [
        cpu for core in acquisition for cpu in core[1:]
    ]
    write_cpus = [core[0] for core in recording] + [
        cpu for core in recording for cpu in core[1:]
    ]
    return gui_cpus, acquire_cpus, write_cpus


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


def get_all_camera_ids(interface="aravis", n_cams=1, prefix="Basler"):
    if (interface == "aravis") or (interface == "all"):
        Aravis.update_device_list()
        n_cams = Aravis.get_n_devices()
        use_ids = [Aravis.get_device_id(i) for i in range(n_cams)]
        if prefix is not None:
            use_ids = [_id for _id in use_ids if _id.startswith(prefix)]
    elif (interface == "fake") or (interface == "fake_custom"):
        use_ids = {f"Fake_{i + 1}" for i in range(n_cams)}
    else:
        raise RuntimeError(f"Did not understand interface {interface}")
    ids = {_id: interface for _id in use_ids}
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


def intensity_to_rgba_lut(frame, minval=1800, maxval=2200, colormap=cv2.COLORMAP_TURBO):
    """Map a 2D uint8/uint16 image to float32 RGBA using a cached lookup table."""
    if frame.ndim != 2 or frame.dtype not in (np.uint8, np.uint16):
        raise ValueError("frame must be a 2D uint8 or uint16 array")
    if not np.isfinite(minval) or not np.isfinite(maxval) or maxval <= minval:
        raise ValueError("Display limits must be finite, with maxval greater than minval")
    if isinstance(colormap, np.ndarray):
        if colormap.dtype != np.uint8 or colormap.shape != (256, 1, 3):
            raise ValueError("Custom colormap must be a uint8 array with shape (256, 1, 3)")
        colormap = colormap.tobytes()
    return _intensity_rgba_lut(minval, maxval, colormap)[frame]


@lru_cache(maxsize=16)
def _intensity_rgba_lut(minval, maxval, colormap):
    # Bytes make custom colormaps hashable and detect changes to their contents.
    if isinstance(colormap, bytes):
        colormap = np.frombuffer(colormap, dtype=np.uint8).reshape(256, 1, 3)
    values = np.arange(65536, dtype=np.uint16).reshape(-1, 1)
    rgba = intensity_to_rgba(values, minval, maxval, colormap)
    return rgba.astype(np.float32).reshape(65536, 4)


def initialize_camera(id, interface: str, config={}, **kwargs):
    if (interface == "aravis") or (interface == "all"):
        from cammy.camera.aravis import AravisCamera

        cam = AravisCamera(id=id, **kwargs)
        for k, v in config.items():
            try:
                cam.set_feature(k, v)
            except Exception as exc:
                raise RuntimeError(f"Camera {id}: failed to set {k}={v!r}") from exc
    elif interface == "fake_custom":
        from cammy.camera.fake import FakeCamera

        cam = FakeCamera(id=id)
    else:
        raise RuntimeError(f"Did not understand interface {interface}")
    return cam


def initialize_cameras(ids, configs, *, skip_features=(), **kwargs):
    cameras = {}
    for _id, _interface in ids.items():
        use_config = {}
        for pattern, groups in configs.get("genicam", {}).items():
            if pattern in _id:
                for settings in groups.values():
                    use_config.update(settings)
        for name in skip_features:
            use_config.pop(name, None)
        try:
            cameras[_id] = initialize_camera(_id, _interface, use_config, **kwargs)
        except Exception:
            logger.exception("Failed to initialize camera %s", _id)

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
