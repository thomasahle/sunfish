"""Portable advisory locks for resumable tuning studies."""

import os
import pathlib


def exclusive(path):
    lock_path = pathlib.Path(f"{path}.lock")
    handle = lock_path.open("r+" if lock_path.exists() else "w+")
    handle.seek(0)
    handle.write(str(os.getpid()))
    handle.truncate()
    handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError(f"another process is using {path}") from None
    return handle
