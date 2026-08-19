"""Run and cache deterministic feasibility checks before spending games."""

import contextlib
import json
import os
import shlex
import signal
import subprocess
import time


def policy(command, timeout, payload, cache, lock=None):
    """Return whether a policy is feasible; exit 1 and timeouts mean no."""
    if not command:
        return True
    key = json.dumps(payload["options"], sort_keys=True, separators=(",", ":"))
    lock = lock or contextlib.nullcontext()
    with lock:
        if key in cache:
            return cache[key]["accepted"]
    started = time.perf_counter()
    process = subprocess.Popen(
        shlex.split(command), text=True, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=os.name != "nt")
    try:
        output = process.communicate(json.dumps(payload), timeout=timeout)[0]
        status = process.returncode
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        output = process.communicate()[0] + f"\ntimeout after {timeout:g}s"
        status = 1
    if status not in (0, 1):
        raise RuntimeError(f"policy gate failed with status {status}:\n{output}")
    record = {
        "accepted": status == 0,
        "knobs": payload["options"],
        "output": output[-2000:],
        "seconds": time.perf_counter() - started,
    }
    with lock:
        cache[key] = record
    print(f"[gate] {'accept' if record['accepted'] else 'reject'} "
          f"{record['seconds']:.2f}s {key}", flush=True)
    return record["accepted"]
