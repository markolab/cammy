import click
from cammy.util import get_all_cameras_aravis
from cammy.camera.aravis import AravisCamera

@click.group()
def cli():
    pass


@cli.command(name="aravis-load-settings")
def aravis_load_settings():
    # loads settings into camera memory
    raise NotImplementedError


@cli.command(name="preview_live")
def preview_live():
    # fire up all aravis devices and gives the user widgets to test relevant settings
    raise NotImplementedError


@cli.command(name="preview_simple")
@click.option("--all-cameras-aravis", is_flag=True)
@click.option("--use-fake-camera", is_flag=True)
@click.option("--n-fake-cameras", type=int, default=1)
def preview_simple(all_cameras_aravis: bool, use_fake_camera: bool, n_fake_cameras: int):
    # simply spool up and show input from all detected cameras
    cameras = []
    if all_cameras_aravis and not use_fake_camera:
        ids = get_all_cameras_aravis()  # ids of all cameras
    elif use_fake_camera:
        # spool up n fake cameras
        for i in range(n_fake_cameras):
            _cam = AravisCamera(fake_camera=True, id=f"Fake_{i}")
            cameras.append(_cam)
    else:
        raise RuntimeError("Incompatible flag settings")



@cli.command(name="get-genicam-xml")
@click.argument("device")
def generate_config(device: str):
    # uses aravis to aget a genicam xml with all features on camera
    raise NotImplementedError


if __name__ == "__main__":
    cli()
