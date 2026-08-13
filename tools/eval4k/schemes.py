"""Eval-table encodings, each emitting a drop-in replacement for the entry's
eval region (it must define `piece`, `pst`, `K_MID`, `K_END`).

The premise being tested: numpy and 60 s of startup are free under the TCEC
rules, so evaluation data can be stored in whatever form is *smallest* and
expanded once at load time into the plain 120-square tables the search
already reads.  Nothing here changes the hot loop -- `value(move)` still does
two table lookups.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splice  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PIECE, RAW = splice.classic_tables(open(os.path.join(ROOT, splice.ENTRY)).read())
ORDER = "PNBRQK"

# ---------------------------------------------------------------------------
# base-90 printable digits: ASCII 35..126 minus the quote and the backslash,
# which cannot appear raw inside a Python string literal.
BAD = (39, 92)
ALPHA = [chr(c) for c in range(35, 127) if c not in BAD]
assert len(ALPHA) == 90


def enc(n):
    s = ""
    while n:
        n, d = divmod(n, 90)
        s = ALPHA[d] + s
    return s or ALPHA[0]


DEC = '_v=0\nfor _c in "%s":\n _d=ord(_c)-35;_v=_v*90+_d-(_d>4)-(_d>56)\n'
# d = ord(c)-35, so the two forbidden codes sit at d==4 (39) and d==57 (92);
# subtracting one per gap crossed maps the 90 live codes onto 0..89.


def _check_dec():
    """Positive control on the codec itself: it must round-trip, and it must
    NOT round-trip if the gap correction is removed."""
    for n in (0, 1, 89, 90, 12345678901234567890, 2 ** 400 + 7):
        s = enc(n)
        v = 0
        for c in s:
            d = ord(c) - 35
            v = v * 90 + d - (d > 4) - (d > 56)
        assert v == n, (n, s, v)


_check_dec()


def mixed(vals, radix):
    """Little-endian mixed-radix pack -- no bits wasted on a power of two."""
    n = 0
    for x in reversed(vals):
        n = n * radix + x
    return n


PIECE_SRC = 'piece = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}\n'
TAIL_SRC = (
    'K_MID, K_END = pst["K"], tuple(piece["K"] + 70\n'
    "   - 10 * (abs(2 * (i // 10) - 11) + abs(2 * (i % 10) - 9)) for i in range(120))\n"
)


def build64(tables, step=1):
    """Encode six 64-value tables; emit a decoder that rebuilds the padded pst."""
    lo = min(min(t) for t in tables.values())
    hi = max(max(t) for t in tables.values())
    off = (-lo + step - 1) // step * step if lo < 0 else 0
    lvl = (hi + off) // step + 1
    vals = []
    for k in ORDER:
        for x in tables[k]:
            q = (x + off + step // 2) // step
            vals.append(max(0, min(lvl - 1, q)))
    n = mixed(vals, lvl)
    src = PIECE_SRC + DEC % enc(n)
    src += "pst = {}\n"
    src += 'for _k in "PNBRQK":\n'
    src += " _t = [_v // %d ** _i %% %d * %d - %d + piece[_k] for _i in range(64)]\n" % (lvl, lvl, step, off)
    src += " _v //= %d ** 64\n" % lvl
    src += " pst[_k] = tuple([0] * 20 + sum(([0] + _t[_i * 8:_i * 8 + 8] + [0] for _i in range(8)), []) + [0] * 20)\n"
    src += TAIL_SRC
    return src, len(enc(n)), lvl


def quantise(step):
    return {k: [(x + step // 2) // step * step for x in v] for k, v in RAW.items()}


def mirror(tables):
    out = {}
    for k, t in tables.items():
        m = []
        for r in range(8):
            row = t[r * 8:r * 8 + 8]
            avg = [(row[f] + row[7 - f]) // 2 for f in range(8)]
            m += avg
        out[k] = m
    return out


def build32(tables, step=1):
    """File-mirrored: store 32 values per piece and unfold at load time."""
    tables = mirror(tables)
    half = {k: [tables[k][r * 8 + f] for r in range(8) for f in range(4)] for k in ORDER}
    lo = min(min(t) for t in half.values())
    hi = max(max(t) for t in half.values())
    off = (-lo + step - 1) // step * step if lo < 0 else 0
    lvl = (hi + off) // step + 1
    vals = []
    for k in ORDER:
        for x in half[k]:
            vals.append(max(0, min(lvl - 1, (x + off + step // 2) // step)))
    n = mixed(vals, lvl)
    src = PIECE_SRC + DEC % enc(n)
    src += "pst = {}\n"
    src += 'for _k in "PNBRQK":\n'
    src += " _h = [_v // %d ** _i %% %d * %d - %d + piece[_k] for _i in range(32)]\n" % (lvl, lvl, step, off)
    src += " _v //= %d ** 32\n" % lvl
    src += " _r = [_h[_i * 4:_i * 4 + 4] for _i in range(8)]\n"
    src += " pst[_k] = tuple([0] * 20 + sum(([0] + _q + _q[::-1] + [0] for _q in _r), []) + [0] * 20)\n"
    src += TAIL_SRC
    return src, len(enc(n)), lvl


# ---------------------------------------------------------------------------
def _dec_literal(tabs):
    """Reconstruct the pre-decoder form: 384 decimal numbers plus the pad
    loop. This is the thing every encoding is priced against."""
    src = PIECE_SRC + "pst = {\n"
    for k in ORDER:
        rows = ["    " + ", ".join("%4d" % x for x in tabs[k][i * 8:i * 8 + 8]) for i in range(8)]
        src += "    '%s': (\n%s),\n" % (k, ",\n".join(rows))
    src += "}\n"
    src += "for k, table in pst.items():\n"
    src += "    padrow = lambda row: (0,) + tuple(x + piece[k] for x in row) + (0,)\n"
    src += "    pst[k] = sum((padrow(table[i * 8 : i * 8 + 8]) for i in range(8)), ())\n"
    src += "    pst[k] = (0,) * 20 + pst[k] + (0,) * 20\n"
    return src + TAIL_SRC


def s_dec():
    """The pre-decoder baseline: classic's tables as decimal literals."""
    return _dec_literal(RAW)


def s_zeros():
    """Control: the decimal-literal form with all 384 numbers zero. The size
    delta against `s_dec` is the marginal packed cost of the DATA itself."""
    return _dec_literal({k: [0] * 64 for k in ORDER})


def s_b90_null():
    """Control: the decoder machinery with a 1-value payload. Isolates the
    fixed cost of the decode path from the cost of the data."""
    return build64({k: [0] * 64 for k in ORDER}, 1)[0]


def s_b90():
    return build64(RAW, 1)[0]


def s_b90_q2():
    return build64(quantise(2), 2)[0]


def s_b90_q4():
    return build64(quantise(4), 4)[0]


def s_b90_q8():
    return build64(quantise(8), 8)[0]


def s_b90_q16():
    return build64(quantise(16), 16)[0]


def s_mirror():
    return build32(RAW, 1)[0]


def s_mirror_q4():
    return build32(quantise(4), 4)[0]


def s_mirror_q8():
    return build32(quantise(8), 8)[0]


SCHEMES = {
    "dec": s_dec,
    "zeros": s_zeros,
    "b90_null": s_b90_null,
    "b90": s_b90,
    "b90_q2": s_b90_q2,
    "b90_q4": s_b90_q4,
    "b90_q8": s_b90_q8,
    "b90_q16": s_b90_q16,
    "mirror": s_mirror,
    "mirror_q4": s_mirror_q4,
    "mirror_q8": s_mirror_q8,
}
