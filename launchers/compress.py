"""Convert finished Cammy sessions to lossless FFV1 AVIs, one camera at a time per session."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event


def positive_integer(name, value):
    if not value.isascii() or not value.isdecimal() or int(value) < 1:
        raise SystemExit(f"{name} must be a positive integer")
    return int(value)


def encoder_threads(session_jobs):
    override = os.environ.get("COMPRESS_THREADS")
    if override:
        return positive_integer("COMPRESS_THREADS", override)
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return max(1, min(5, available // session_jobs))


def compress_session(session, threads, chunk_size, stopping):
    failures = 0
    converted = 0
    skipped = 0
    print(f"Session: {session}", flush=True)
    for raw in sorted(session.glob("*.dat")):
        if stopping.is_set():
            break
        if not raw.is_file():
            continue
        avi = raw.with_suffix(".avi")
        try:
            original = raw.stat()
            if avi.exists():
                if not avi.is_file() or not avi.stat().st_size or avi.stat().st_mtime_ns < original.st_mtime_ns:
                    raise ValueError(f"Existing output is empty, invalid, or older than the raw file: {avi}")
                print(f"  Skipping existing {avi}", flush=True)
                skipped += 1
                continue
            print(f"  Compressing {raw}", flush=True)
            with TemporaryDirectory(prefix=".cammy-compress-", dir=session) as tmp:
                staged = Path(tmp) / avi.name
                subprocess.run(["markolab-cli", "convert-dat-to-avi", str(raw),
                                "--output-filename", str(staged), "--chunk-size", str(chunk_size),
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
    return converted, skipped, failures


def compress_sessions(root):
    session_jobs = positive_integer("COMPRESS_SESSIONS", os.getenv("COMPRESS_SESSIONS", "2"))
    threads = encoder_threads(session_jobs)
    chunk_size = positive_integer("COMPRESS_CHUNK_SIZE", os.getenv("COMPRESS_CHUNK_SIZE", "64"))
    with (root / ".cammy-compress.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"A compression job is already running in {root}")

        print(f"Compressing up to {session_jobs} sessions at once with {threads} threads per file "
              f"and {chunk_size} frames per chunk.", flush=True)
        sessions = sorted(path for path in root.glob("session_*") if path.is_dir())
        stopping = Event()
        worker = partial(compress_session, threads=threads, chunk_size=chunk_size, stopping=stopping)
        converted = skipped = failures = 0
        with ThreadPoolExecutor(max_workers=session_jobs) as pool:
            try:
                for done, existing, failed in pool.map(worker, sessions):
                    converted += done
                    skipped += existing
                    failures += failed
            finally:
                stopping.set()
    print(f"{root}: {converted} converted, {skipped} skipped, {failures} failed.", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(compress_sessions(Path(sys.argv[1]).resolve(strict=True)))
