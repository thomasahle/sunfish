"""The encode target, and the independent bit-exact mirror.

A trained ternary replnet is exactly (shift, g[N], bias_digits[N],
trits[768][N]) -- what export_replnet writes and the entry's decode loop
reconstructs.  Every encoder arm must round-trip to THIS tuple; the
module-data mirror below turns it into the entry's decoded globals
(SHIFT, MGP, MGH, ACC_BASE, ROWS) by an independent reimplementation of
the entry algebra, so an arm's spliced entry can be checked for
bit-exactness without trusting the arm's own decoder.
"""
import os
import pickle
from collections import namedtuple

# Entry constants (replnet_proto.py).  Mirrored here, and re-asserted
# against the spliced module at verify time so drift is loud.
NN, LBITS, VBITS = 4, 16, 15
HALF = NN * LBITS
M16 = (1 << LBITS) - 1
_U = ((1 << 2 * HALF) - 1) // M16
_R2 = 1 | 1 << HALF
MH, MLO = _U << VBITS, _U << 14
PIECES = "PNBRQKpnbrqk"

QNet = namedtuple("QNet", "name shift g bd trits clampcp Efloat struct")
QNet.__new__.__defaults__ = (None,)
# trits: tuple of 768 tuples of N ints in {-1,0,+1}, feature-major
# (feat = piece_index*64 + sq64), lane-minor -- export_replnet's order.
# Efloat: the pre-quantization float rows (768 x N) for predictor arms;
# NOT part of the encode target -- round-trips are against `trits`.
# struct: the TRAINED-STRUCTURE record (train/structures.py) when the net
# was trained through a parametrization -- codebook + assignments, or
# U/V/R.  Arms that need it declare NotApplicable on nets without it;
# every other arm ignores it and prices the same trits.


def load_qnet(path):
    """RUN_DIR | .pickle | .pickle.payload -> QNet, from the pickle's own
    trainer-side fields (E is quantized exactly as export_replnet does)."""
    if os.path.isdir(path):
        path = os.path.join(path, "best.pickle")
    if path.endswith(".payload"):
        path = path[:-len(".payload")]
    with open(path, "rb") as f:
        d = pickle.load(f)
    assert d["kind"] == "replnet-ternary", d["kind"]
    trits = tuple(
        tuple(max(-1, min(1, round(x * 32))) for x in row) for row in d["E"])
    assert len(trits) == 768 and all(len(r) == d["N"] for r in trits)
    if d.get("struct") is not None:
        # a structured net's table is ALREADY on the grid: the defensive
        # clamp above must have been a no-op, or the pickle and the
        # parametrization disagree (never hide it behind the clamp)
        assert all(round(x * 32) == t for row, trow in zip(d["E"], trits)
                   for x, t in zip(row, trow)), \
            "structured net %s: E is off the ternary grid" % path
    name = os.path.basename(path)
    for suf in (".pickle",):
        if name.endswith(suf):
            name = name[:-len(suf)]
    if name == "best":                     # run-dir convention: name the run
        name = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return QNet(name, d["shift"], list(d["g"]), list(d["bias_digits"]),
                trits, d["clampcp"], d["E"], d.get("struct"))


def header_digits(q):
    """The 1+N+N base-90 header digits in extraction (LSB-first) order."""
    return [q.shift] + list(q.g) + list(q.bd)


def header_int(q):
    """Header packed LSB-first: digit i at 90**i.  Every arm's payload is
    header_int + 90**9 * body_int, so the shared header-pop source works
    unchanged across arms."""
    n = 0
    for d in reversed(header_digits(q)):
        n = n * 90 + d
    return n


HEADER_RADIX = 90 ** 9  # 1 + N + N digits


def flat_trits(q):
    """Feature-major lane-minor flat list, len 3072 -- the canonical
    stream order (the entry consumes exactly this order)."""
    return [t for row in q.trits for t in row]


def symbols81(q):
    """One 0..80 symbol per feature: sum (t+1)*3^k -- the current char."""
    return [sum((t + 1) * 3 ** k for k, t in enumerate(row)) for row in q.trits]


# ---------------------------------------------------------------- enc90
# export.py's digit map (NOT codec.ALPHA: e=4 emits the apostrophe, which
# is legal inside the entry's double-quoted string; both maps invert to
# the same decoder, and the recorded 3831/3834 artifacts used this one).

def enc90_digit(e):
    d = e + (e >= 5)
    d += d >= 57
    return chr(35 + d)


def dec90_digit(c):
    d = ord(c) - 35
    return d - (d > 4) - (d > 56)


def int_to_s90(n):
    """Big int -> the entry's base-90 string (MSB char first)."""
    digs = []
    while n:
        n, d = divmod(n, 90)
        digs.append(d)
    s = "".join(enc90_digit(d) for d in reversed(digs)) or enc90_digit(0)
    assert "\\" not in s and '"' not in s
    return s


def s90_to_int(s):
    n = 0
    for c in s:
        n = n * 90 + dec90_digit(c)
    return n


def int_to_bytes(n):
    return n.to_bytes(max(1, (n.bit_length() + 7) // 8), "big")


# ------------------------------------------------- independent mirror

def sq120(f):
    return 21 + f // 8 * 10 + f % 8


def expected_module_data(q):
    """(SHIFT, MGP, MGH, ACC_BASE, ROWS) exactly as the entry builds them
    -- an independent reimplementation, verified once per net against a
    verify_export-blessed baseline splice, then the gate every arm's
    module must equal bit-for-bit.  Equality here implies identical
    evaluation: nn_cp reads nothing else but these and fixed constants."""
    B = 0
    for k in range(NN):
        B += q.bd[k] - 44 << LBITS * k
    MGP = sum(q.g[k] * 32 << LBITS * k for k in range(NN)) * _R2
    MGH = MGP | MH
    ACC_BASE = MLO + B * _R2
    half = {}
    for pi, p in enumerate(PIECES):
        h = [0] * 120
        for f in range(64):
            row = q.trits[pi * 64 + f]
            h[sq120(f)] = sum(q.g[k] * row[k] << LBITS * k for k in range(NN))
        half[p] = h
    rows0 = {p: [half[p][s] + (half[p.swapcase()][119 - s] << HALF)
                 for s in range(120)] for p in PIECES}
    rows1 = {p: [rows0[p.swapcase()][119 - s] for s in range(120)] for p in PIECES}
    return {"SHIFT": q.shift, "MGP": MGP, "MGH": MGH, "ACC_BASE": ACC_BASE,
            "ROWS": (rows0, rows1)}


def module_data_of(mod):
    """The same tuple pulled from an exec'd spliced entry module."""
    return {"SHIFT": mod["SHIFT"], "MGP": mod["MGP"], "MGH": mod["MGH"],
            "ACC_BASE": mod["ACC_BASE"], "ROWS": mod["ROWS"]}
