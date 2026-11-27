"""Install Cammy shortcuts or the separate session compression environment and shortcut."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("desktop", nargs="?", type=Path, help="Desktop directory")
parser.add_argument("--compression", action="store_true", help="Install the compression environment and storage shortcuts")
args = parser.parse_args()
scripts = Path(__file__).resolve().parent
conda_prefix = os.environ.get("CONDA_PREFIX")
if args.compression or conda_prefix:
    conda = shutil.which(os.environ.get("CONDA_EXE") or "conda")
    if not conda:
        sys.exit("Could not find conda. Run the installer from a Conda-enabled shell.")
if args.compression:
    subprocess.run([conda, "env", "update", "--name", "markolab-cli", "--file", scripts / "markolab-cli.yml"], check=True)
    compression_env = [os.path.abspath(conda), "run", "--no-capture-output", "--name", "markolab-cli"]
    subprocess.run([*compression_env, "markolab-cli", "convert-dat-to-avi", "--help"], check=True)
else:
    cammy = shutil.which("cammy")
    if cammy is None:
        sys.exit("Activate the environment that runs cammy, then rerun this installer.")
    cammy = os.path.abspath(cammy)
if args.desktop is not None:
    desktop = args.desktop.expanduser()
elif shutil.which("xdg-user-dir"):
    desktop = Path(subprocess.check_output(["xdg-user-dir", "DESKTOP"], text=True).strip())
else:
    desktop = Path.home() / "Desktop"
desktop.mkdir(parents=True, exist_ok=True)

settings = desktop.resolve() / "cammy-settings.sh"
defaults = (scripts / "settings.sh").read_text(encoding="utf-8")
try:
    with settings.open("x", encoding="utf-8") as dest:
        dest.write(defaults)
except FileExistsError:
    if not settings.is_file():
        sys.exit(f"Expected a settings file at {settings}")
    existing = settings.read_text(encoding="utf-8")
    additions = []
    for line in defaults.splitlines():
        name, sep, _ = line.partition("=")
        if sep and name.isidentifier() and name not in existing:
            additions.append(f"# {line}")
    if additions:
        with settings.open("a", encoding="utf-8") as dest:
            dest.write("\n# Uncomment a setting below to override its default.\n")
            dest.write("\n".join(additions) + "\n")
print(f"Shared settings: {settings}")


def exec_arg(arg):
    # Desktop entries parse backslash escapes before parsing quotes in Exec.
    value = os.fspath(arg).replace("%", "%%")
    for char in ('\\', '"', '`', '$'):
        value = value.replace(char, '\\' + char)
    return '"' + value.replace("\\", "\\\\") + '"'


def install_shortcut(slug, label, argv):
    script_dir = str(scripts).replace("\\", "\\\\")
    command = " ".join(exec_arg(arg) for arg in argv)
    shortcut = desktop / f"cammy-{slug}.desktop"
    shortcut.write_text(
        f"[Desktop Entry]\nType=Application\nName=Cammy {label}\n"
        f"Path={script_dir}\nExec={command}\nTerminal=true\nIcon=camera-video\n",
        encoding="utf-8",
    )
    shortcut.chmod(0o755)
    print(shortcut)


install_shortcut("trim", "Trim Data Drive", ["bash", "./capture.sh", "trim", settings])
if args.compression:
    install_shortcut("compress", "Compress Sessions", [*compression_env, "bash", "./capture.sh", "compress", settings])
    sys.exit(0)

launcher = []
if conda_prefix:
    launcher.extend([os.path.abspath(conda), "run", "--no-capture-output", "--prefix", conda_prefix])
    print(f"Desktop Conda environment: {conda_prefix}")
# Aravis may also depend on search paths exported by the rig's interactive shell.
paths = [f"{key}={os.environ[key]}" for key in ("GI_TYPELIB_PATH", "LD_LIBRARY_PATH") if os.environ.get(key)]
if paths:
    launcher.extend(["env", *paths])
launcher.extend(["bash", "./capture.sh"])
shortcuts = [(mode, action) for mode in ("surface", "fluorescence", "alternate") for action in ("preview", "record")]
shortcuts.append(("tune",))
for action in shortcuts:
    install_shortcut("-".join(action), " ".join(action).title(), [*launcher, *action, cammy, settings])
