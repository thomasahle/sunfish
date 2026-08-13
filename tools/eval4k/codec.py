"""The startup decoder: eval tables stored as one big integer, expanded once.

The reframe this lane is testing: TCEC 4k gives 60 s of startup and allows
numpy, so evaluation data should be stored in whatever form is *smallest*
and expanded once at load time into the plain 120-square tables the search
already reads.  Nothing enters the hot loop -- `value(move)` still does two
table lookups and the score stays O(1) incremental.

Measured on the real packer (see MEASUREMENTS): classic's 384 numbers cost
**502 packed bytes** as a decimal literal and **395** through this codec,
while the codec machinery itself costs **13**.  The saving is exact: the
decoded tables are bit-identical to the literal ones.

No numpy: the decoder is nine lines of integer arithmetic, so it runs under
pypy3 with no import and no fallback.  (pypy3 on this laptop *does* now have
numpy 2.4.6 -- the ledger's "our pypy3 has no numpy" note is stale -- but a
decoder that needs nothing is worth more than one that needs a wheel.)
"""

# base-90 printable digits: ASCII 35..126 minus the apostrophe (39) and the
# backslash (92), which cannot appear raw inside a Python string literal.
ALPHA = [chr(c) for c in range(35, 127) if c not in (39, 92)]
assert len(ALPHA) == 90

# d = ord(c)-35 puts the two forbidden codes at d==4 and d==57; subtracting
# one per gap crossed maps the 90 live codes back onto 0..89.
DEC = '_v=0\nfor _c in "%s":\n _d=ord(_c)-35;_v=_v*90+_d-(_d>4)-(_d>56)\n'

ORDER = "PNBRQK"


def enc(n):
    s = ""
    while n:
        n, d = divmod(n, 90)
        s = ALPHA[d] + s
    return s or ALPHA[0]


def dec(s):
    v = 0
    for c in s:
        d = ord(c) - 35
        v = v * 90 + d - (d > 4) - (d > 56)
    return v


def mixed(vals, radix):
    """Little-endian mixed-radix pack -- no bits wasted rounding up to a power
    of two, which is worth ~10% at 210 levels and more as levels shrink.

    int() on every input is load-bearing, not defensive noise: this accumulator
    is a 3000-bit Python integer, and a single numpy int64 anywhere in `vals`
    makes the whole product numpy and SILENTLY WRAPS at 64 bits. A caller that
    built its tables with numpy got a valid-looking source encoding garbage,
    announced only by a RuntimeWarning nobody reads."""
    n = 0
    for x in reversed(vals):
        n = n * int(radix) + int(x)
    return n


def _block(tabs, order, step, half, dest, init=False):
    """One decode loop: `order`'s tables at one (step, half) setting."""
    t2 = {k: list(tabs[k]) for k in order}
    if half:
        t2 = {k: [(t[r * 8 + f] + t[r * 8 + 7 - f]) // 2 for r in range(8) for f in range(4)]
              for k, t in t2.items()}
    lo = min(min(t) for t in t2.values())
    hi = max(max(t) for t in t2.values())
    off = (-lo + step - 1) // step * step if lo < 0 else 0
    lvl = (hi + off) // step + 1
    n_per = 32 if half else 64
    vals = [max(0, min(lvl - 1, (x + off + step // 2) // step)) for k in order for x in t2[k]]
    src = DEC % enc(mixed(vals, lvl))
    if init:
        src += "%s = {}\n" % dest
    src += 'for _k in "%s":\n' % order
    src += " _t = [_v // %d ** _i %% %d * %d - %d + piece[_k] for _i in range(%d)]\n" % (
        lvl, lvl, step, off, n_per)
    src += " _v //= %d ** %d\n" % (lvl, n_per)
    if half:
        src += " _r = [_t[_i * 4:_i * 4 + 4] for _i in range(8)]\n"
        src += " %s[_k] = tuple([0] * 20 + sum(([0] + _q + _q[::-1] + [0] for _q in _r), []) + [0] * 20)\n" % dest
    else:
        src += " %s[_k] = tuple([0] * 20 + sum(([0] + _t[_i * 8:_i * 8 + 8] + [0]" % dest
        src += " for _i in range(8)), []) + [0] * 20)\n"
    return src


def emit(piece, raw, step=1, half=False, exact=""):
    """Source that rebuilds `piece`, `pst`, `K_MID`, `K_END`.

    raw:   {piece: 64 ints}, rank-8-first, piece value NOT included
    step:  quantisation step (1 = exact)
    half:  store 4 files per rank and unfold by mirroring (192 values, not 384)
    exact: pieces held back at full resolution in a SECOND decode block.

    `exact` exists for one specific reason. Mirroring is not lossless -- every
    one of classic's tables is left-right asymmetric, the king by as much as
    111 cp -- and the king's asymmetry is the castling-side preference that the
    landed kend fix depends on. Compressing the OTHER five tables must not
    silently perturb it, or a screen measures the fit and the perturbation
    together and a negative result says nothing about either.
    """
    tabs = {k: list(v) for k, v in raw.items()}
    order = "".join(k for k in ORDER if k not in exact)
    # `piece` used to be accepted and then IGNORED: this line was a hard-coded
    # copy of classic's values, so a caller emitting a fit's OWN values got
    # classic's back unless it remembered to patch the first line out again.
    # All three callers did remember -- but the failure mode of forgetting is a
    # plausible-looking artifact carrying the wrong piece values, which is the
    # same silent-corruption class as the mirrored king and the numpy wrap.
    src = "piece = {%s}\n" % ", ".join('"%s": %d' % (k, int(piece[k])) for k in ORDER)
    src += _block(tabs, order, step, half, "pst", init=True)
    if exact:
        src += _block(tabs, "".join(k for k in ORDER if k in exact), 1, False, "pst")
    src += ('K_MID, K_END = pst["K"], tuple(piece["K"] + 70\n'
            "   - 10 * (abs(2 * (i // 10) - 11) + abs(2 * (i % 10) - 9)) for i in range(120))\n")
    return src


def _selftest():
    """Positive control on the codec, including a case it must FAIL: drop the
    gap correction and the round-trip must break."""
    for n in (0, 1, 89, 90, 2 ** 400 + 7):
        assert dec(enc(n)) == n, n
    bad = enc(2 ** 400 + 7)
    naive = 0
    for c in bad:
        naive = naive * 90 + ord(c) - 35
    assert naive != 2 ** 400 + 7, "negative control did not fail"
    vals = [3, 0, 7, 209, 1]
    assert dec(enc(mixed(vals, 210))) == mixed(vals, 210)
    # 64-bit wrap control: 384 values at radix 210 is a ~3000-bit integer, so
    # a numpy-typed input must produce the SAME number as the Python one.
    try:
        import numpy as _np
    except ImportError:
        return
    big = list(range(200)) * 2
    assert mixed([_np.int64(v) for v in big], 210) == mixed(big, 210), \
        "numpy inputs wrap the mixed-radix accumulator"


_selftest()
