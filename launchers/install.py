"""Install the capture and tuning shortcuts using the active Cammy environment."""

import os
from pathlib import Path
import shutil
import subprocess
import sys


cammy = shutil.which("cammy")
if cammy is None:
    sys.exit("Activate the environment that runs cammy, then rerun this installer.")
cammy = os.path.abspath(cammy)
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    conda = shutil.which(os.environ.get("CONDA_EXE") or "conda")
    if not conda:
        sys.exit(f"Could not find conda for {conda_prefix}. Activate the working Cammy environment and retry.")
if len(sys.argv) > 2:
    sys.exit("Usage: python3 install.py [desktop-directory]")
if len(sys.argv) == 2:
    desktop = Path(sys.argv[1]).expanduser()
elif shutil.which("xdg-user-dir"):
    desktop = Path(subprocess.check_output(["xdg-user-dir", "DESKTOP"], text=True).strip())
else:
    desktop = Path.home() / "Desktop"
desktop.mkdir(parents=True, exist_ok=True)

scripts = Path(__file__).resolve().parent
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


script_dir = str(scripts).replace("\\", "\\\\")
launcher = []
if conda_prefix:
    launcher.extend([os.path.abspath(conda), "run", "--no-capture-output", "--prefix", conda_prefix])
    print(f"Desktop Conda environment: {conda_prefix}")
# Aravis may also depend on search paths exported by the rig's interactive shell.
paths = [f"{key}={os.environ[key]}" for key in ("GI_TYPELIB_PATH", "LD_LIBRARY_PATH") if os.environ.get(key)]
if paths:
    launcher.extend(["env", *paths])
launcher.extend(["bash", "./capture.sh"])
command = " ".join(exec_arg(arg) for arg in launcher)
shortcuts = [(mode, action) for mode in ("surface", "fluorescence", "alternate") for action in ("preview", "record")]
shortcuts.append(("tune",))
for args in shortcuts:
    action = " ".join(args)
    shortcut = desktop / f"cammy-{'-'.join(args)}.desktop"
    shortcut.write_text(
        f"[Desktop Entry]\nType=Application\nName=Cammy {action.title()}\n"
        f"Path={script_dir}\nExec={command} {action} {exec_arg(cammy)} {exec_arg(settings)}\n"
        "Terminal=true\nIcon=camera-video\n",
        encoding="utf-8",
    )
    shortcut.chmod(0o755)
    print(shortcut)
