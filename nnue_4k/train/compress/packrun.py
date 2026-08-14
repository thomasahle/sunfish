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
import tempfile

_here = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
NICE = ["nice", "-n", "15"]

# The pack scripts are PINNED to a git rev, like the entry and for the
# same incident: the golf lane's --no-hoist-literals/shebang-strip pack.sh
# sat UNCOMMITTED in the working tree on 2026-08-14 and silently moved
# every measured artifact (old-blob baseline 3831 -> 3780, settled 3594 ->
# 3567).  A bake-off through an in-flight packer looks current and is
# not; when the change LANDS, HEAD moves and one re-run refreshes every
# number against the landed path.
PACK_REV = os.environ.get("BAKEOFF_PACK_REV", "HEAD")
_pinned = {}


def _pin(relpath):
    """Materialize REV:tools/build/<script> once; returns (path, prov)."""
    if relpath not in _pinned:
        spec = "%s:tools/build/%s" % (PACK_REV, relpath)
        src = subprocess.run(["git", "-C", REPO, "show", spec],
                             capture_output=True, text=True, check=True).stdout
        sha = subprocess.run(["git", "-C", REPO, "rev-parse", spec],
                             capture_output=True, text=True, check=True).stdout.strip()
        wt = os.path.join(REPO, "tools", "build", relpath)
        with open(wt) as f:
            if f.read() != src:
                print("  NOTE: working-tree %s differs from pinned %s (%s) -- "
                      "packing with the PIN" % (relpath, PACK_REV, sha[:12]),
                      flush=True)
        d = tempfile.mkdtemp(prefix="bakeoff_pack_")
        path = os.path.join(d, relpath)
        with open(path, "w") as f:
            f.write(src)
        _pinned[relpath] = (path, "%s (blob %s)" % (spec, sha[:12]))
    return _pinned[relpath]


def provenance():
    return {"pack_a": _pin("pack.sh")[1], "pack_b": _pin("pack_entry.sh")[1]}


def _run(cmd, **kw):
    r = subprocess.run(NICE + cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise RuntimeError("FAILED: %s\n%s%s" % (" ".join(cmd), r.stdout, r.stderr))
    return r


def pack_a(entry_path, out_path):
    _run(["bash", _pin("pack.sh")[0], entry_path, out_path])
    return os.path.getsize(out_path)


def pack_b(entry_path, weights_path, out_path):
    _run(["bash", _pin("pack_entry.sh")[0], entry_path, weights_path, out_path])
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
