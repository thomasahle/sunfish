#!/usr/bin/env python3
"""Emit the payload string for packed128_compose_proto.py: a REAL-SHAPED
stand-in for an N-lane ternary packed residual, sized for byte pricing.

The weights are random (this is a PRICING artifact -- training them is the
staged retrain's job), but the encoding is the real one a generator would
emit, and the sparsity is a knob the retrain recipe explicitly regularises
for, so lzma sees the true byte cost:

  * base-90 BYTES through the entry's own codec
    (d = c - 35 - (c > 92)): a bytes literal forbids only the quote (34) and
    the backslash (92), so [35,125] minus {92} is exactly 90 live codes and
    the decode pays ONE gap test instead of two,
  * one char per 4 trits (values 0..80 < 88): trit groups stay CHAR-ALIGNED,
    which is what lets lzma exploit zero-heavy weights (the ledger's
    "base-3 and lzma COMPOSE" measurement, commit 4850894),
  * extraction order, LSB first: shift, N gains, N biases, 768*N trits --
    and the string is EMITTED in that order too. The entry used to build a
    big integer and peel digits back off it, which is the identity on the
    digit sequence read backwards; dropping the big integer drops the
    reversal with it.

--feats > 768 sizes the payload for a LARGER capacity (Thomas's 1024-B
payload directive, 2026-08-14): the extra feature chars sit at the END of
the string now that it is emitted LSB-first, which the entry's decode never reads -- the
artifact stays runnable while lzma prices the full-capacity stream. The
real larger-net decode seam (n8 / kb deltas) is agreed with TRAINQUEUE
before any training run uses it.
"""
import argparse
import random


def enc(e):
    """Inverse of the entry codec's digit map (one gap, over the backslash)."""
    return chr(35 + e + (35 + e >= 92))


p = argparse.ArgumentParser()
p.add_argument("--N", type=int, default=4)
p.add_argument("--seed", type=int, default=20260814)
p.add_argument("--zeros", type=float, default=0.55)
p.add_argument("--feats", type=int, default=768)
p.add_argument("--u2", type=int, default=0,
               help="ml2 second-layer read-out: emit this many signed u2 "
                    "values (|u| <= 127) as offset-4050 base-90 digit PAIRS "
                    "between the biases and the feature chars (certify_ml2 "
                    "layout; 0 = single-layer replnet payload, unchanged)")
args = p.parse_args()

rng = random.Random(args.seed)
digits = [6]                                          # shift
digits += [rng.randint(20, 60) for _ in range(args.N)]    # gains C_k
digits += [rng.randint(0, 87) for _ in range(args.N)]     # biases b_k + 44
for _ in range(args.u2):                                  # ml2 u2, LSB pair first
    d = rng.randint(-127, 127) + 4050
    digits += [d % 90, d // 90]
trits = [0 if rng.random() < args.zeros else rng.choice((-1, 1))
         for _ in range(args.feats * args.N)]
for i in range(0, len(trits), 4):
    digits.append(sum((trits[i + j] + 1) * 3 ** j for j in range(4)))

s = "".join(enc(d) for d in digits)                   # first char = LSB
assert "\\" not in s and '"' not in s
print(s)
