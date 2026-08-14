"""Entry-source surgery: splice a payload, or replace the decode region.

The entry (replnet_proto.py) has ONE region every arm may rewrite: from
the `_w = 0` line through the end of the `_half` build.  Everything an
arm's decoder must define is what that region defines -- SHIFT, MGP, MGH,
ACC_BASE, _half -- and the shared templates below cover the common parts
so an arm usually only supplies the code that turns the payload integer
into the flat trit list `_T`.

Layout prologues (how `_w` is obtained):

  A  the entry's own base-90 string loop, string embedded in source;
  B  int.from_bytes over the raw tail of the artifact itself, located
     via the SF_A/SF_N environment the pack_entry.sh head exports.

Verification helpers exec the UNMINIFIED spliced source (pyminify is
semantics-preserving; the packed artifact additionally gets a boot smoke)
and hand back the module globals for the bit-exact gate.
"""
import os
import re
import subprocess
import time

PAYLOAD_RE = re.compile(r'^for _c in "(.*)":$', re.M)
REGION_START = "\n_w = 0\n"
_REGION_END_LINE = "\n    _half[_p] = _h\n"


def read_entry(spec):
    """Entry source, PINNED.  `spec` is REV:PATH (read from the git object
    store) or a plain filesystem path.  Returns (source, provenance).

    The pin exists because of a measured incident (2026-08-14): another
    lane was golfing replnet_proto.py in the working tree DURING a
    bake-off, and three same-morning measurements of "the entry" read
    3831/3728/3733 B.  A ranked table against a moving denominator is
    noise, so the default is HEAD's blob and the provenance lands in the
    results json."""
    if ":" in spec and not os.path.exists(spec):
        rev, path = spec.split(":", 1)
        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        src = subprocess.run(["git", "-C", repo, "show", spec],
                             capture_output=True, text=True, check=True).stdout
        sha = subprocess.run(["git", "-C", repo, "rev-parse", spec],
                             capture_output=True, text=True, check=True).stdout.strip()
        wt = os.path.join(repo, path)
        dirty = False
        if os.path.exists(wt):
            with open(wt) as f:
                dirty = f.read() != src
        if dirty:
            print("  NOTE: working-tree %s differs from pinned %s (%s) -- "
                  "measuring the PIN" % (path, rev, sha[:12]), flush=True)
        return src, "%s (blob %s)" % (spec, sha[:12])
    with open(spec) as f:
        return f.read(), "file %s (UNPINNED working tree)" % os.path.abspath(spec)


def splice_payload(src, s90):
    """export.py's splice: replace the payload string, touch nothing else
    (the baseline arm's layout A must reproduce the recorded bytes)."""
    m = PAYLOAD_RE.search(src)
    if not m:
        raise ValueError("no payload string in entry")
    return src[:m.start(1)] + s90 + src[m.end(1):]


# Everything this module splices and the bit-exact gate reads.  The entry
# belongs to the GOLF LANE and moves; when it moves past the seam, the
# harness must say so in one line instead of dying on str.index deep in a
# measurement (2026-08-14: golf round 2 inlined NN/LBITS/VBITS, renamed
# ACC_BASE to _B and restructured the _half build -- every arm broke at
# once, and the first symptom was a bare ValueError).
GATE_GLOBALS = ("NN", "LBITS", "VBITS", "ACC_BASE", "ROWS")


def seam_missing(src):
    """What an entry source no longer exposes, in words."""
    miss = []
    if not PAYLOAD_RE.search(src):
        miss.append('the payload string line (for _c in "...":)')
    if REGION_START not in src:
        miss.append("the decode region start (a bare `_w = 0` line)")
    if _REGION_END_LINE not in src:
        miss.append("the decode region end (`    _half[_p] = _h`)")
    gone = [n for n in GATE_GLOBALS if not re.search(r"\b%s\b" % n, src)]
    if gone:
        miss.append("module globals the bit-exact gate compares: " + ", ".join(gone))
    return miss


def require_seam(src, prov="this entry source"):
    """Refuse to measure against a drifted entry -- loudly, and without
    guessing at its new shape (re-seaming compress/ is a coordination job
    with the lane that owns replnet_proto.py, not an inference)."""
    miss = seam_missing(src)
    if miss:
        raise RuntimeError(
            "ENTRY SEAM DRIFT -- %s no longer exposes what the bake-off "
            "splices and gates:\n  * %s\nNothing is measured until compress/ "
            "is re-seamed against the entry's new shape." % (prov, "\n  * ".join(miss)))


def _region_span(src):
    """[start, end) of the replaceable decode region."""
    require_seam(src)
    i = src.index(REGION_START) + 1          # keep the preceding newline
    j = src.index(_REGION_END_LINE)
    j += len(_REGION_END_LINE)
    return i, j


def replace_region(src, region_src):
    i, j = _region_span(src)
    if not region_src.endswith("\n"):
        region_src += "\n"
    return src[:i] + region_src + src[j:]


# --------------------------------------------------------------- layouts

def prologue_a(s90):
    """The entry's own decode loop, arm's string spliced in."""
    return ('_w = 0\nfor _c in "%s":\n'
            " _d = ord(_c) - 35; _w = _w * 90 + _d - (_d > 4) - (_d > 56)\n" % s90)


PROLOGUE_B = ('_w = int.from_bytes(open(os.environ["SF_A"], "rb").read()'
              '[-int(os.environ["SF_N"]):], "big")\n')


# ------------------------------------------------------ shared templates
# Header pop: identical digits in identical order for every arm, so the
# payload is always header + 90**9 * body and arms only own the body.
SRC_HEADER = """\
_w, SHIFT = divmod(_w, 90)
_g, _B = [], 0
for _k in range(NN):
    _w, _d = divmod(_w, 90); _g.append(_d)
for _k in range(NN):
    _w, _d = divmod(_w, 90); _B += _d - 44 << LBITS * _k
MGP = sum(_g[_k] * 32 << LBITS * _k for _k in range(NN)) * _R2
MGH = MGP | MH
ACC_BASE = MLO + _B * _R2
"""

# _half from a flat trit list _T (feature-major, lane-minor, len 3072).
SRC_HALF_FROM_T = """\
_half = {}
_i = 0
for _p in _PIECES:
    _h = [0] * 120
    for _f in range(64):
        _r = 0
        for _k in range(NN):
            _r += _g[_k] * _T[_i] << LBITS * _k; _i += 1
        _h[21 + _f // 8 * 10 + _f % 8] = _r
    _half[_p] = _h
"""

# _half from 768 base-81 symbols in a list _S (feature-major).
SRC_HALF_FROM_S = """\
_half = {}
_i = 0
for _p in _PIECES:
    _h = [0] * 120
    for _f in range(64):
        _d = _S[_i]; _i += 1
        _r = 0
        for _k in range(NN):
            _d, _t = divmod(_d, 3); _r += _g[_k] * (_t - 1) << LBITS * _k
        _h[21 + _f // 8 * 10 + _f % 8] = _r
    _half[_p] = _h
"""


def build_region(prologue, body):
    """prologue (defines _w) + header pop/masks + arm body (defines _half,
    usually via _T or _S and a shared tail)."""
    return prologue + SRC_HEADER + body


# ---------------------------------------------------------- verification

def exec_entry(src, tail_bytes=None, tmpdir=None):
    """Exec unminified spliced source; returns (globals dict, seconds).
    For layout B sources, `tail_bytes` is written to a file and SF_A/SF_N
    point at it -- read()[-n:] only needs the tail to BE the file end."""
    g = {"__name__": "spliced_bakeoff_entry"}
    old = {k: os.environ.get(k) for k in ("SF_A", "SF_N")}
    tf = None
    try:
        if tail_bytes is not None:
            tf = os.path.join(tmpdir, "tail.bin")
            with open(tf, "wb") as f:
                f.write(tail_bytes)
            os.environ["SF_A"] = tf
            os.environ["SF_N"] = str(len(tail_bytes))
        t0 = time.perf_counter()
        exec(compile(src, "<spliced-entry>", "exec"), g)
        dt = time.perf_counter() - t0
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return g, dt
