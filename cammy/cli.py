import click


@click.group()
def cli():
    pass


@cli.command(name="aravis-load-settings")
def aravis_load_settings():
    # loads settings into camera memory
    raise NotImplementedError


@cli.command(name="aravis-live-preview")
def aravis_live_preview():
    # fire up all aravis devices and gives the user widgets to test relevant settings
    raise NotImplementedError


@cli.command(name="get-genicam-xml")
@click.argument("device")
def generate_config(device: str):
    # uses aravis to aget a genicam xml with all features on camera
    raise NotImplementedError


if __name__ == "__main__":
    cli()
