"""Price the payload container: bytes per base-90 digit, by stream statistics.

The factored-compression lane needs one number the ledger does not yet carry:
how many BYTES a payload digit costs as a function of its ALPHABET and its
zero-heaviness.  The full-table family only ever measured one point (four
trits per digit at ~55 % zeros, "1.67 bits/trit"), and a factored payload is a
different stream -- larger alphabets, no trit grouping, different sparsity --
so its byte price cannot be read off that point.

Method: splice a synthetic digit stream of the requested shape into the real
`replnet_proto.py` payload literal and run the real `tools/build/pack.sh`.
Every number is a size read off a built file.  The artifact is NOT required to
run (the digits are random); this prices the CONTAINER, not a net.

    python3 price_container.py                 # the standard ladder
    python3 price_container.py --digits 900 --alphabet 90 --zeros 0.0
"""
import argparse
import pathlib
import random
import re
import subprocess
import tempfile

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[2]
ENGINE = REPO / "nnue_4k" / "replnet_proto.py"
PACK = REPO / "tools" / "build" / "pack.sh"
# The entry's codec: digit d -> byte 35+d, skipping the backslash (92).
PAT = re.compile(r'b"[^"]*"')


def enc(d):
    return chr(35 + d + (35 + d >= 92))


def stream(n, alphabet, zeros, rng):
    """n digits over `alphabet` levels; `zeros` is P(digit == the zero code).

    The zero code is (alphabet-1)//2 for signed alphabets and 0 for trit
    groups -- both are just "the value lzma sees most often", which is all
    that matters for the container price.
    """
    zero = (alphabet - 1) // 2
    out = []
    for _ in range(n):
        if rng.random() < zeros:
            out.append(zero)
        else:
            out.append(rng.randrange(alphabet))
    return out


def trit_groups(n_trits, per_digit, zeros, rng):
    """The full-table family's stream: `per_digit` trits packed per digit."""
    trits = [0 if rng.random() < zeros else rng.choice((-1, 1)) for _ in range(n_trits)]
    out = []
    for i in range(0, len(trits), per_digit):
        grp = trits[i:i + per_digit]
        out.append(sum((t + 1) * 3 ** j for j, t in enumerate(grp)))
    return out


def pack(digits):
    """Splice `digits` into the engine's payload literal and pack it."""
    src = ENGINE.read_text()
    body = "".join(enc(d) for d in digits)
    assert '"' not in body and "\\" not in body
    # The payload is the LONGEST bytes literal in the file; the engine carries
    # short ones too (a docstring example, the UCI time-field pick).
    m = max(PAT.finditer(src), key=lambda m: len(m.group()))
    assert len(m.group()) > 400, "payload literal not found (longest is %d B)" % len(m.group())
    new = src[:m.start()] + 'b"%s"' % body + src[m.end():]
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "e.py"
        p.write_text(new)
        out = pathlib.Path(td) / "e.packed"
        r = subprocess.run(["bash", str(PACK), str(p), str(out)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return out.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digits", type=int)
    ap.add_argument("--alphabet", type=int, default=90)
    ap.add_argument("--zeros", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    base = pack([])
    print("code side, payload elided: %d B" % base)
    if args.digits:
        n = pack(stream(args.digits, args.alphabet, args.zeros, rng))
        print("%d digits, alphabet %d, zeros %.2f: %d B (payload %d B, %.4f B/digit)"
              % (args.digits, args.alphabet, args.zeros, n, n - base,
                 (n - base) / args.digits))
        return

    print("\n--- alphabet ladder, 1000 digits, uniform (a FACTOR payload) ---")
    print("%-10s %8s %8s %9s %9s" % ("alphabet", "total", "payload", "B/digit", "bits/dig"))
    for a in (3, 5, 9, 16, 27, 45, 64, 81, 90):
        n = pack(stream(1000, a, 0.0, rng))
        print("%-10d %8d %8d %9.4f %9.3f"
              % (a, n, n - base, (n - base) / 1000, (n - base) * 8 / 1000))

    print("\n--- zero-heaviness at alphabet 90, 1000 digits ---")
    for z in (0.0, 0.2, 0.4, 0.55, 0.7):
        n = pack(stream(1000, 90, z, rng))
        print("zeros %.2f: %8d  payload %6d  %.4f B/digit" % (z, n, n - base, (n - base) / 1000))

    print("\n--- the full-table family's own stream (4 trits/digit) ---")
    for z in (0.43, 0.5, 0.55):
        d = trit_groups(768 * 4, 4, z, rng)
        n = pack(d)
        print("N=4, %d trits @ %.0f%% zeros: %8d  payload %6d  %.4f B/digit  %.3f bits/trit"
              % (768 * 4, z * 100, n, n - base, (n - base) / len(d),
                 (n - base) * 8 / (768 * 4)))

    print("\n--- linearity check: is B/digit constant in stream length? ---")
    for k in (200, 500, 1000, 1500, 2000):
        n = pack(stream(k, 90, 0.0, rng))
        print("%5d digits: %8d  payload %6d  %.4f B/digit" % (k, n, n - base, (n - base) / k))


if __name__ == "__main__":
    main()
