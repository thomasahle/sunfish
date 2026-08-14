"""The encoder zoo.  One module per family; every arm registers here.

Arm protocol -- an arm owns only the BODY of the payload:

    encode(q: QNet) -> (body_int, body_src, note)

  * body_int: the arm's whole stored state as one integer (mixed radix,
    LSB = first thing its decoder pops).  The harness prepends the shared
    9-digit header (shift, gains, biases), so the full payload integer is
    header + 90**9 * body, identical across arms and layouts.
  * body_src: module-level Python that consumes the global `_w` (already
    holding body_int when it runs) and defines `_half` -- usually by
    building `_T` (3072 flat trits) or `_S` (768 base-81 symbols) and
    appending a shared tail from entrysrc.  Its bytes are PART OF THE
    ARTIFACT and are what the decoder-cost column measures.
  * arms with native_a = True additionally promise that body_src decodes
    the SAME digit stream the stock entry's decode block does, so layout
    A can splice the string into the untouched entry (byte-identical to
    the recorded export path).

Encoders may be arbitrarily clever; decoders pay by the byte.  Nothing
here is skipped because it "obviously" loses -- the ledger's
estimate-vs-measured record (8+ misses) is why the zoo measures.

Some arms price a STRUCTURE the net must have been trained through
(train/structures.py -- q.struct): those raise NotApplicable on a plain
net, which the harness records as a SKIP, never as a pass and never as a
failure.  Their post-hoc cousins (cb8, lr_svd) stay in the zoo as the
controls: same decoder shape, structure fitted after the fact.
"""


class NotApplicable(Exception):
    """This arm cannot price THIS net (the structure it encodes is absent).
    A skip in the table -- distinct from an arm that FAILED."""


def mixed_pack(pairs):
    """pairs = (radix, digit) in DECODE POP ORDER -> the integer whose
    successive divmods by those radices yield those digits."""
    n = 0
    for r, d in reversed(pairs):
        assert 0 <= d < r, (r, d)
        n = n * r + d
    return n


def base90_pairs(value, ndigits):
    """value split into `ndigits` base-90 pops (LSB first)."""
    out = []
    for _ in range(ndigits):
        value, d = divmod(value, 90)
        out.append((90, d))
    assert value == 0
    return out


ARMS = []


def register(arm):
    ARMS.append(arm() if isinstance(arm, type) else arm)
    return arm


def all_arms():
    # import for side effect: each module registers its arms
    from . import base81, controls, reorder, lanes, mixedradix, rans, \
        codebook, sparse, lowrank, rle, trained  # noqa: F401
    return list(ARMS)
