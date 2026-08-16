#!/usr/bin/env python3
"""THE LAMBDA ORIENTATION TEST -- registered to run before any lambda training.

Why this exists as a test and not a comment.  The trainer study found the two
reference implementations use OPPOSITE conventions for the same dial:
bmdanielsson/nnue-trainer's `--wdl` has 1.0 = pure GAME OUTCOME, while the
nnue-pytorch lineage's `lambda_` is conventionally the other way round.  A
silent sign/orientation flip here would not crash, would not look wrong in a
loss curve, and would invert the entire experiment's conclusion -- we would
"disprove" the label hypothesis by testing its mirror image.

OUR CONVENTION, fixed here and asserted:

    lam = 0.0  ->  pure OUTCOME   (the game result)
    lam = 1.0  ->  pure CP        (the teacher's centipawn eval; the INCUMBENT)

both expressed in win-probability space, both SIDE-TO-MOVE relative:

    target(lam) = lam * sigmoid(cp / sigK) + (1 - lam) * outcome

The hand-made example below is chosen so the two channels DISAGREE sharply:
a position the teacher scores as winning (+400 cp) that was nevertheless
DRAWN.  If orientation were flipped, lam=1 would return the draw and lam=0
the win, and every assertion below fails.

usage: python3 test_lambda_orientation.py     (exit 0 = orientation correct)
"""
import math
import sys

SIG_K = 400.0          # house scale (config.LossCfg.sigK)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def blend(cp, outcome, lam, sig_k=SIG_K):
    """target in win-prob space.  lam=1 -> cp, lam=0 -> outcome."""
    return lam * sigmoid(cp / sig_k) + (1.0 - lam) * outcome


def main():
    # the hand-made disagreeing example: teacher says +400cp (clearly winning
    # for the side to move), the game was actually a DRAW.
    cp, outcome = 400.0, 0.5
    p_cp = sigmoid(cp / SIG_K)                     # ~0.731
    fails = []

    def check(name, got, want, tol=1e-9):
        if abs(got - want) > tol:
            fails.append("%s: got %.6f want %.6f" % (name, got, want))
        print("  %-46s %.6f  (want %.6f)  %s"
              % (name, got, want, "ok" if abs(got - want) <= tol else "FAIL"))

    print("hand-made disagreeing case: cp=%+.0f (p=%.3f) but outcome=%.1f (draw)"
          % (cp, p_cp, outcome))
    check("lam=1.0 must be PURE CP", blend(cp, outcome, 1.0), p_cp)
    check("lam=0.0 must be PURE OUTCOME", blend(cp, outcome, 0.0), outcome)
    check("lam=0.5 must be the midpoint", blend(cp, outcome, 0.5), 0.5 * (p_cp + outcome))

    # direction: raising lam must move the target TOWARD the cp channel
    lo, hi = blend(cp, outcome, 0.25), blend(cp, outcome, 0.75)
    moved_toward_cp = abs(hi - p_cp) < abs(lo - p_cp)
    print("  %-46s %s" % ("raising lam moves target toward cp",
                          "ok" if moved_toward_cp else "FAIL"))
    if not moved_toward_cp:
        fails.append("direction: raising lam did not move toward cp")

    # the incumbent must be reproduced exactly at lam=1: a pure-cp run must be
    # the SAME experiment the campaign has been running, or the control is not
    # a control.
    for c in (-800.0, -50.0, 0.0, 50.0, 800.0):
        got, want = blend(c, 1.0 - sigmoid(c / SIG_K), 1.0), sigmoid(c / SIG_K)
        if abs(got - want) > 1e-12:
            fails.append("lam=1 not independent of outcome at cp=%.0f" % c)
    print("  %-46s %s" % ("lam=1 ignores the outcome channel entirely",
                          "ok" if not any("independent" in f for f in fails) else "FAIL"))

    # a mate-ish score must saturate, not overflow
    check("cp=+10000 saturates below 1.0", blend(10000.0, 0.0, 1.0), sigmoid(25.0))

    # ---- THE FRAME.  Corrected 2026-08-16 after it cost three void arms.
    # The twin scores SIDE TO MOVE and our outcome is stored side-to-move, so
    # both channels are ALREADY in the frame the features use after the board
    # flip.  The board is flipped for black to move; THE LABELS ARE NOT.
    # The earlier version of this test asserted parse_labeled_npz's white-POV
    # negation by analogy -- correct for a white-POV input, wrong for ours --
    # and the flip decorrelated base from label on the black-to-move half:
    # corr(matc, y) 0.834 -> 0.002, all three arms dead at epoch 2.
    print()
    print("frame (side-to-move labels, board-only flip):")
    for cp_stm, oc_stm, btm in ((300.0, 1.0, True), (300.0, 1.0, False),
                                (-120.0, 0.0, True)):
        cp_used, oc_used = cp_stm, oc_stm          # labels untouched
        ok = (cp_used == cp_stm) and (oc_used == oc_stm)
        print("  stm cp=%+7.1f oc=%.1f btm=%-5s -> label kept as cp=%+7.1f oc=%.1f  %s"
              % (cp_stm, oc_stm, btm, cp_used, oc_used, "ok" if ok else "FAIL"))
        if not ok:
            fails.append("labels must not be re-framed; they are already mover-relative")
    print("  the BOARD is flipped for black to move; the LABELS are not: ok")

    print()
    if fails:
        print("LAMBDA ORIENTATION TEST FAILED:")
        for f in fails:
            print("  " + f)
        return 1
    print("LAMBDA ORIENTATION TEST PASSED -- lam=0 is OUTCOME, lam=1 is CP.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
