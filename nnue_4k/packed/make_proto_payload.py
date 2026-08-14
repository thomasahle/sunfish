#!/usr/bin/env python3
"""Emit the payload string for packed128_compose_proto.py: a REAL-SHAPED
stand-in for an N-lane ternary packed residual, sized for byte pricing.

The weights are random (this is a PRICING artifact -- training them is the
staged retrain's job), but the encoding is the real one a generator would
emit, and the sparsity is a knob the retrain recipe explicitly regularises
for, so lzma sees the true byte cost:

  * base-90 characters through the entry's own codec
    (d = ord(c)-35; v = v*90 + d - (d>4) - (d>56)),
  * one char per 4 trits (values 0..80 < 88): trit groups stay CHAR-ALIGNED,
    which is what lets lzma exploit zero-heavy weights (the ledger's
    "base-3 and lzma COMPOSE" measurement, commit 4850894),
  * extraction order, LSB first: shift, N gains, N biases, 768*N trits.
"""
import argparse
import random


def enc(e):
    """Inverse of the entry codec's digit map (skips '\\' and ')')."""
    d = e + (e >= 5)
    d += d >= 57
    return chr(35 + d)


p = argparse.ArgumentParser()
p.add_argument("--N", type=int, default=4)
p.add_argument("--seed", type=int, default=20260814)
p.add_argument("--zeros", type=float, default=0.55)
args = p.parse_args()

rng = random.Random(args.seed)
digits = [6]                                          # shift
digits += [rng.randint(20, 60) for _ in range(args.N)]    # gains C_k
digits += [rng.randint(0, 87) for _ in range(args.N)]     # biases b_k + 44
trits = [0 if rng.random() < args.zeros else rng.choice((-1, 1))
         for _ in range(768 * args.N)]
for i in range(0, len(trits), 4):
    digits.append(sum((trits[i + j] + 1) * 3 ** j for j in range(4)))

s = "".join(enc(d) for d in reversed(digits))         # first char = MSB
assert "\\" not in s and '"' not in s
print(s)
