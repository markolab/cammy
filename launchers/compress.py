"""Convert finished Cammy sessions to lossless FFV1 AVIs, one camera at a time."""

import fcntl
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def encoder_threads():
    override = os.environ.get("COMPRESS_THREADS")
    if override:
        if not override.isascii() or not override.isdecimal() or int(override) < 1:
            raise SystemExit("COMPRESS_THREADS must be empty or a positive integer")
        return int(override)
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def compress_sessions(root):
    threads = encoder_threads()
    failures = 0
    converted = 0
    skipped = 0
    with (root / ".cammy-compress.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"A compression job is already running in {root}")

        print(f"Encoding with {threads} threads per file.", flush=True)
        for session in sorted(root.glob("session_*")):
            if not session.is_dir():
                continue
            print(f"Session: {session}", flush=True)
            for raw in sorted(session.glob("*.dat")):
                if not raw.is_file():
                    continue
                avi = raw.with_suffix(".avi")
                try:
                    original = raw.stat()
                    if avi.exists():
                        if not avi.is_file() or not avi.stat().st_size or avi.stat().st_mtime_ns < original.st_mtime_ns:
                            raise ValueError(f"Existing output is empty, invalid, or older than the raw file: {avi}")
                        print(f"  Skipping existing {avi.name}", flush=True)
                        skipped += 1
                        continue
                    print(f"  Compressing {raw.name}", flush=True)
                    with TemporaryDirectory(prefix=".cammy-compress-", dir=session) as tmp:
                        staged = Path(tmp) / avi.name
                        subprocess.run(["markolab-cli", "convert-dat-to-avi", str(raw),
                                        "--output-filename", str(staged), "--chunk-size", "64",
                                        "--threads", str(threads)], check=True)
                        if not staged.is_file() or not staged.stat().st_size:
                            raise ValueError("Converter produced no AVI. Check the camera entry in metadata.toml.")
                        current = raw.stat()
                        if (current.st_size, current.st_mtime_ns) != (original.st_size, original.st_mtime_ns):
                            raise ValueError("Raw file changed during conversion. Retry after recording stops.")
                        os.link(staged, avi)  # Publish verified output without replacing an existing file.
                    converted += 1
                except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                    print(f"Failed: {raw}: {exc}", file=sys.stderr, flush=True)
                    failures += 1
    print(f"{root}: {converted} converted, {skipped} skipped, {failures} failed.", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(compress_sessions(Path(sys.argv[1]).resolve(strict=True)))
