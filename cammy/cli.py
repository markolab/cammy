import click
from cammy.util import get_all_cameras_aravis
from cammy.camera.aravis import AravisCamera
from cammy.camera.fake import FakeCamera


@click.group()
def cli():
    pass


@cli.command(name="aravis-load-settings")
def aravis_load_settings():
    # loads settings into camera memory
    raise NotImplementedError


@cli.command(name="live-preview")
def live_preview():
    # fire up all aravis devices and gives the user widgets to test relevant settings
    raise NotImplementedError


@cli.command(name="simple-preview")
@click.option("--all-cameras-aravis", is_flag=True)
@click.option("--use-fake-camera", is_flag=True)
@click.option("--n-fake-cameras", type=int, default=1)
@click.option("--fake-camera-interface", type=str, default="custom")
def simple_preview(all_cameras_aravis: bool, use_fake_camera: bool, n_fake_cameras: int, fake_camera_interface: str):
    import time
    # simply spool up and show input from all detected cameras
    cameras = []
    if all_cameras_aravis and not use_fake_camera:
        # ids = get_all_cameras_aravis()  # ids of all cameras
        # then spool here
        raise NotImplementedError
    elif use_fake_camera:
        # spool up n fake cameras
        for i in range(n_fake_cameras):
            if fake_camera_interface == "aravis": 
                _cam = AravisCamera(fake_camera=True, id=f"Fake_{i+1}")
            elif fake_camera_interface == "custom":
                _cam = FakeCamera(id=f"Fake_{i+1}")
            else:
                raise RuntimeError(f"Did not understand fake camera interface {fake_camera_interface}")
            cameras.append(_cam)
    else:
        raise RuntimeError("Incompatible flag settings")

    [_cam.start_acquisition() for _cam in cameras]
    try:
        while True:
            dat = [_cam.try_pop_frame() for _cam in cameras]
            for _dat in dat:
                if _dat[0] is not None:
                    print(_dat)
            time.sleep(.001)
    finally:
        [_cam.stop_acquisition() for _cam in cameras]



@cli.command(name="get-genicam-xml")
@click.argument("device")
def generate_config(device: str):
    # uses aravis to aget a genicam xml with all features on camera
    raise NotImplementedError


if __name__ == "__main__":
    cli()
