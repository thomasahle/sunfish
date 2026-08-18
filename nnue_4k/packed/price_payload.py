"""Marginal payload price, MEASURED through tools/build/pack.sh.

Instrument, not an engine: we splice payload strings of a given LENGTH and a
given CHARACTER DISTRIBUTION into the real replnet_proto.py and record what
pack.sh says.  The code side is byte-identical across rows, so the DIFFERENCE
between two rows is the payload's price and nothing else.  The spliced builds
do not run (the decoder's shape is unchanged) -- that is fine and it is the
same method the ledger used to price N=5/N=6 with random weights.
"""
import os, random, re, subprocess, sys, tempfile

REPO = "/Users/ahle/repos/sunfish-packed"
PROTO = os.path.join(REPO, "nnue_4k/replnet_proto.py")
src = open(PROTO).read()

# the payload literal: the long string in the `for _c in "...":` decode line
m = re.search(r'for _c in "([^"]{500,})":', src)
assert m, "payload literal not found"
BASE_PAYLOAD = m.group(1)
print("baseline payload chars:", len(BASE_PAYLOAD))

# base-90 alphabet exactly as the decoder reads it: ord(c) - 35, with the two
# skipped codes (the decoder's `- (_d > 4) - (_d > 56)` accounts for `"` and
# `\`), so the legal characters are chr(35+d) minus those two.
ALPHA = [chr(35 + i) for i in range(94) if chr(35 + i) not in ('"', "\\")][:90]
assert len(ALPHA) == 90

def trit_chars(n, zero_frac, rng):
    """n chars, each holding 4 trits; each trit is 0 with prob zero_frac."""
    out = []
    for _ in range(n):
        d = 0
        for k in range(4):
            t = 0 if rng.random() < zero_frac else rng.choice((-1, 1))
            d += (t + 1) * 3 ** k
        out.append(ALPHA[d])
    return "".join(out)

def uniform_chars(n, rng):
    return "".join(rng.choice(ALPHA) for _ in range(n))

def pack(payload):
    body = src[:m.start(1)] + payload + src[m.end(1):]
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(body); path = f.name
    out = subprocess.run(["bash", os.path.join(REPO, "tools/build/pack.sh"),
                          path, path + ".packed"],
                         capture_output=True, text=True)
    n = os.path.getsize(path + ".packed")
    os.unlink(path); os.unlink(path + ".packed")
    return n

rng = random.Random(20260819)
rows = []
print("%-34s %6s %6s %8s" % ("payload", "chars", "bytes", "B/char"))
base_n = pack(BASE_PAYLOAD)
print("%-34s %6d %6d %8s" % ("SHIPPED N=4 trained payload", len(BASE_PAYLOAD), base_n, "-"))

# regime (a): trit-packed chars at three sparsities, three lengths each
for z in (0.50, 0.70, 0.85):
    prev = None
    for n in (768, 1536, 3072):
        b = pack(trit_chars(n, z, rng))
        d = "" if prev is None else "%.4f" % ((b - prev[1]) / (n - prev[0]))
        print("%-34s %6d %6d %8s" % ("trit4 zeros=%.2f" % z, n, b, d))
        prev = (n, b)

# regime (b): uniform base-90 digits (V, gains, bias)
prev = None
for n in (768, 1024, 1280):
    b = pack(uniform_chars(n, rng))
    d = "" if prev is None else "%.4f" % ((b - prev[1]) / (n - prev[0]))
    print("%-34s %6d %6d %8s" % ("uniform base-90", n, b, d))
    prev = (n, b)
