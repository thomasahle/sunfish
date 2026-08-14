"""The REAL pack paths, wrapped -- never composed arithmetic.

Layout A: tools/build/pack.sh entry.py out          (joint lzma stream)
Layout B: tools/build/pack_entry.sh entry.py w.bin out  (split, raw tail)

Both scripts do their own sed/pyminify/xz; the only number this module
ever reports is os.path.getsize of the artifact the script wrote, plus a
boot smoke of that artifact ('uci' -> 'uciok') so the head mechanism is
exercised, not assumed -- layout B's SF_A self-read in particular.
"""
import os
import subprocess

_here = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
PACK_A = os.path.join(REPO, "tools", "build", "pack.sh")
PACK_B = os.path.join(REPO, "tools", "build", "pack_entry.sh")
NICE = ["nice", "-n", "15"]


def _run(cmd, **kw):
    r = subprocess.run(NICE + cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise RuntimeError("FAILED: %s\n%s%s" % (" ".join(cmd), r.stdout, r.stderr))
    return r


def pack_a(entry_path, out_path):
    _run(["bash", PACK_A, entry_path, out_path])
    return os.path.getsize(out_path)


def pack_b(entry_path, weights_path, out_path):
    _run(["bash", PACK_B, entry_path, weights_path, out_path])
    return os.path.getsize(out_path)


def boot_smoke(artifact, timeout=90):
    """Run the artifact for real: 'uci' must answer 'uciok'.  Returns wall
    seconds (interpreter start + xz -d + full table build included)."""
    import time
    t0 = time.perf_counter()
    r = subprocess.run(NICE + ["bash", artifact],
                       input="uci\nquit\n", capture_output=True, text=True,
                       timeout=timeout, cwd=os.path.dirname(artifact))
    dt = time.perf_counter() - t0
    if "uciok" not in r.stdout:
        raise RuntimeError("boot smoke FAILED for %s:\n%s%s"
                           % (artifact, r.stdout[-2000:], r.stderr[-2000:]))
    return dt
