"""Splice an alternative eval-table section into the 4k PST entry.

The entry's evaluation data lives in one contiguous region: the `piece` dict,
the `pst` literal, the padding loop, and the K_MID/K_END derivation.  Every
scheme this lane prices is a different way of producing *exactly those four
names*, so the honest way to price one is to swap the region and pack the
real file with the real packer.

Nothing here composes byte figures.  `measure.py` builds a file per scheme
and reads the size off disk.
"""
import re

ENTRY = "nnue_4k/pst_entry.py"

# The region runs from the `piece = {` line to the end of the K_MID/K_END
# statement.  Both anchors are unique in the file.
START = '\npiece = {"P": 100,'
END_ANCHOR = "\n###############################################################################\n\n# With xz compression"


def region(src):
    i = src.index(START)
    j = src.index(END_ANCHOR)
    return i, j


def split(src):
    i, j = region(src)
    return src[:i], src[i:j], src[j:]


def splice(src, replacement):
    head, _, tail = split(src)
    return head + "\n" + replacement.strip("\n") + "\n" + tail


def classic_tables(_ignored=None):
    """The 6x64 raw tables and the piece values, from classic -- the single
    source of truth the entry generator also reads.  Reading them back out of
    the *entry* stopped working the moment the entry stored them encoded, and
    an encoded entry is exactly when a self-referential reader would start
    silently comparing a thing to itself."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    classic = open(os.path.join(root, "sunfish.py")).read()
    m = re.search(r"\npiece = \{.*?\n\}\n", classic, re.S)
    ns = {}
    exec(m.group(0), ns)
    return ns["piece"], {k: list(v) for k, v in ns["pst"].items()}


def padded(piece, raw):
    """Reproduce the entry's padded 120-square tables, for equality checks."""
    out = {}
    for k, table in raw.items():
        rows = []
        for i in range(8):
            rows += [0] + [x + piece[k] for x in table[i * 8 : i * 8 + 8]] + [0]
        out[k] = tuple([0] * 20 + rows + [0] * 20)
    return out
