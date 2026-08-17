"""Generate search variants of the 4k entry FROM ONE SOURCE, at screen time.

Five stale-copy failures in one session all had the same shape: a fix written
to a new file while the old file stayed live and reachable. The recorded
mitigation was "variants should be GENERATED at screen time from a single
source, not accumulated as files". This is that generator.

The single source is `nnue_4k/pst_entry.py`, which is itself CI-guarded
against its own generator (`tools/build/check_entry.sh`). Every mod asserts
its anchor text occurs exactly once, so a mod that silently stops applying is
a hard error rather than a variant that is quietly the baseline -- the failure
mode that would make an A/B measure nothing and look fine.

usage:  make_variants.py OUTDIR name[.mod.mod ...] ...
        make_variants.py --list
e.g.    make_variants.py /tmp/bin base cap nolmr nolmr.cap

Names are dot-joined mod lists; `base` means no mods. The output file is
`OUTDIR/e_<name-with-dots-stripped>.py`, and each carries a provenance
header naming the source sha and the mods applied.
"""
import hashlib
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "nnue_4k", "pst_entry.py")
sys.path.insert(0, os.path.join(REPO, "tools", "eval4k"))


def _eval_mod(step, half, exact, fits="tools/tune/candidates/fits.json", arm="flat"):
    """An EVAL mod: swap the whole eval region for a fitted table set.

    DEFERRED, and that is not a style choice. `MODS` is built at import, so an
    eval mod whose fit file is absent used to raise before the dict existed --
    taking every unrelated SEARCH mod down with it and breaking the generator
    for lanes that had nothing to do with the eval. The mod is a callable that
    reads its fit when it is BUILT, so a missing fit fails loudly for the one
    candidate that needs it and for nothing else.

    Every other mod here is a search change written as literal text. The eval
    candidates cannot be, because their tables come from the fit -- so the
    anchor is the entry's current eval region (unique by construction, and the
    occurs-exactly-once check still applies) and the replacement is whatever
    `codec.emit` produces at this encoding. Same single-source rule as the rest
    of the file: the candidate is GENERATED at screen time from the committed
    fit, never accumulated as a file that can go stale against the base.
    """
    import json

    import codec
    import splice
    P = "PNBRQK"
    doc = json.load(open(os.path.join(REPO, fits)))
    T = (doc["arms"][arm] if "arms" in doc else doc[arm])["tables"]
    vals = {p: int(T["_value_" + p]) for p in P}
    raw = {p: [int(v) for v in T[p]] for p in P}
    _, region, _ = splice.split(open(SRC).read())
    return (region, "\n" + codec.emit(vals, raw, step, half, exact=exact).strip("\n") + "\n")


def _deferred(*a, **kw):
    return lambda: _eval_mod(*a, **kw)


# Each mod is (anchor, replacement), or a LIST of such pairs for a mod that
# has to touch several places. Every anchor MUST appear exactly once.
MODS = {
    # ---- EVAL candidates (see nnue_4k/MEASUREMENTS.md, the fits entry) ------
    # Same 384-parameter Texel refit of classic's tables, at two encodings.
    # C1 mirrors and quantises to step 8 and holds the king table back exact,
    # so the landed kend fix is bit-identical; C2 stores the same fit at full
    # resolution. C1 vs C2 is therefore a clean measurement of what the
    # compression costs IN PLAY, which loss cannot answer.
    "c1": _deferred(8, True, "K"),
    "c2": _deferred(1, False, ""),
    # ---- DISTILLED candidates: the same 384-parameter model and the same
    # positions as C2, trained on OUR OWN SEARCH's converged value at 160,000
    # nodes instead of Stockfish depth 8. D1 is exact, so D1-vs-C2 changes the
    # TEACHER and nothing else; D8 is the shippable step-8 form, which is
    # quantisation-aware rather than rounded afterwards and gives bytes BACK.
    # The king is held exact at any step: the codec quantises every table it is
    # handed, and the landed kend fix is not a fit's to round.
    "d1": _deferred(1, False, "", "tools/tune/candidates/students.json", "linear"),
    "d8": _deferred(8, False, "K", "tools/tune/candidates/students.json", "q8"),
    # ---- PHASE-BALANCED candidates. Same teacher and the same LABELS as d1 --
    # these are drawn from d1's own set -- but a flat 2,198-per-band mix instead
    # of the natural 43/31/11/15. The size-matched natural-mix control
    # (nat8792) is stably 13.3% WORSE than classic at phase 18-24 over 40
    # splits while these are 3.5% better, so the mix is the mechanism and not
    # the halved position count. See the pre-registration entry.
    "b1": _deferred(1, False, "", "tools/tune/candidates/bal/students.json", "linear"),
    "b8": _deferred(8, False, "K", "tools/tune/candidates/bal/students.json", "q8"),
    # ---- H1 TAPERED ENDGAME TERMS (nnue_4k/MEASUREMENTS.md, the H1 -------
    # pre-registration). Both are HAND-DESIGNED, not fitted -- every fitted
    # table this lane produced played worse than its loss promised -- and both
    # are cost-class ZERO in the hot loop: they ride the queens-off root seam
    # the landed kend+fresh fix already pays for (one boolean per search, and
    # the from_board rebuild line after the swap handles the carried score).
    #
    # ===================================================================
    # TOMBSTONE: `pend` LANDED 2026-08-15 (61b1a51), CONFIRMED +21.31 (902d9a2).
    # Endgame pawn-advance table: at queens-off, every pawn's value grows
    # quadratically with advancement (0,2,8,18,32,50 by rank 2..7), read by
    # both colours through the 119-i mirror. It is now injected DIRECTLY by
    # tools/build/make_pst_entry.py (`_pend`, beside `_pooltm`), asserted
    # exactly once there the same way this file always asserted it.
    #
    # WHAT IT MEASURED (nnue_4k/MEASUREMENTS.md, "pend CONFIRMED at +21.31
    # and LANDED"): screen (SPRT, stops early) +36.71 +/- 16.20 over 722
    # games; confirmation (fixed 800, no early stopping) pend W336/L287/D177
    # = 53.06%, +21.31 +/- 15.73 -> [+5.58, +37.04], pentanomial. Landed cost
    # +32 packed bytes (3308 -> 3340) measured on its OWN base -- the +37
    # some earlier notes quote was measured against a base that no longer
    # applies; byte deltas never compose across landings.
    #
    # WHICH ANCHORS SURVIVE -- the load-bearing part of a tombstone:
    #
    #   * The SECOND anchor, `pst["K"] = K_MID if "Q" in pos.board and "q"
    #     in pos.board else K_END`, is GONE from the baseline -- the landed
    #     seam reads `end = "Q" not in pos.board or "q" not in pos.board`
    #     then `pst["K"] = K_END if end else K_MID` / `pst["P"] = P_END if
    #     end else P_MID` instead. Re-creating `pend` raises there today.
    #     SAFE, but only half the story.
    #
    #   * The FIRST anchor, `K_MID, K_END = pst["K"], tuple(piece["K"] + 70`,
    #     STILL OCCURS EXACTLY ONCE -- the landing inserts the P_MID/P_END
    #     definition BEFORE this line, not through it, so the line itself is
    #     untouched. A re-created `pend` would match it and insert a SECOND
    #     P_MID/P_END pair WHILE THE OCCURS-EXACTLY-ONCE CHECK ON THAT ANCHOR
    #     PASSES CLEANLY -- silent double-application on top of the already
    #     landed pend, caught only by the second anchor raising first. If the
    #     seam line ever drifts back toward the pre-landing text, that second
    #     anchor stops being the backstop. DO NOT re-create this mod; compose
    #     against the baseline instead, exactly as `pendkhold2` already does.
    #
    # SUBSUMPTION OBLIGATION, carried from the landing: `pend` is a
    # hand-written phase term, and it is DELETED (not composed) the day a
    # phase-capable net screens and subsumes it -- the comparison matrix must
    # include net-vs-net+pend at that point.
    #
    # ledger: nnue_4k/MEASUREMENTS.md, "pend CONFIRMED at +21.31 and LANDED"
    # (902d9a2) -- carries the arm sha256s and the packed sha of the artifact
    # that played the confirmation.
    # ===================================================================
    # `kact`: steeper K_END centralization. The gradient (10 cp per step of
    # centre manhattan distance) was inherited from classic and never swept
    # on this engine; 14/step makes an active king worth up to +126 across
    # the board instead of +90. Known tree-shape effect, accepted and
    # pre-registered: a diagonal centralizing king step's delta goes 40->56,
    # crossing QS=40, so those steps are firmly admitted at depth 0.
    # Composes with pend in either order (disjoint anchors).
    "kact": (
        "   - 10 * (abs(2 * (i // 10) - 11) + abs(2 * (i % 10) - 9)) for i in range(120))\n",
        "   - 14 * (abs(2 * (i // 10) - 11) + abs(2 * (i % 10) - 9)) for i in range(120))\n",
    ),
    # ---- THE MINIMAL TAPER: the queens-off CLIFF becomes a phase RAMP -----
    # The entry already carries two king tables and already picks between them
    # once per search. What it does not have is a taper: the choice is a step
    # function of one boolean, so a queenless middlegame with four rooks and
    # four minors on gets the SAME fully-centralized king table as a bare KRK
    # ending. `ktap` replaces the step with a linear interpolation on the
    # standard 24-point phase (N=B=1, R=2, Q=4), so K_MID holds at full
    # material and K_END arrives only as the material actually leaves.
    #
    # COST CLASS ZERO IN THE HOT LOOP, exactly like pend and kact: the blend
    # runs ONCE per search, at the same root seam the kend+fresh fix already
    # pays for, and the from_board rebuild two lines below re-derives the
    # carried score under whatever table this produces. `value(move)` and
    # every futility/QS margin keep reading one fixed `pst` for the whole
    # search -- there is no per-node multiply and no second accumulator,
    # which is what the 2026-08-13 "continuous phase blend does not fit"
    # pricing was really about (that build carried a second 384-value TABLE
    # SET; this one carries no new data at all).
    #
    # Phase is clamped at 24 so a second queen extrapolates PAST K_MID
    # nowhere: promotions can push the raw sum to 28+.
    #
    # `ktap` leaves the pawn seam alone -- pend is a landed, confirmed +21.31
    # and it is not this mod's to re-decide. `kptap` is the variant that
    # ramps both, and it is measured separately for exactly that reason.
    "ktap": (
        '        end = "Q" not in pos.board or "q" not in pos.board\n'
        '        pst["K"] = K_END if end else K_MID\n',
        '        end = "Q" not in pos.board or "q" not in pos.board\n'
        '        ph = min(24, sum(pos.board.count(c) * w\n'
        '                         for c, w in zip("NnBbRrQq", (1, 1, 1, 1, 2, 2, 4, 4))))\n'
        '        pst["K"] = tuple(e + (m - e) * ph // 24 for m, e in zip(K_MID, K_END))\n',
    ),
    # `kptap`: the same ramp on BOTH tapered tables. pend's endgame pawn bonus
    # currently switches on at the queen exchange and is full strength the
    # instant it does; here it grows with the same phase the king table uses.
    # This one can lose ground that pend already won, which is why it is a
    # separate arm and not a "while we are here" extension of ktap.
    "kptap": (
        '        end = "Q" not in pos.board or "q" not in pos.board\n'
        '        pst["K"] = K_END if end else K_MID\n'
        '        pst["P"] = P_END if end else P_MID\n',
        '        ph = min(24, sum(pos.board.count(c) * w\n'
        '                         for c, w in zip("NnBbRrQq", (1, 1, 1, 1, 2, 2, 4, 4))))\n'
        '        pst["K"] = tuple(e + (m - e) * ph // 24 for m, e in zip(K_MID, K_END))\n'
        '        pst["P"] = tuple(e + (m - e) * ph // 24 for m, e in zip(P_MID, P_END))\n',
    ),
    # ---- H2 KING-SAFETY TERMS (nnue_4k/MEASUREMENTS.md, the H2 -----------
    # pre-registration). The entry is MATED in a third of its losses
    # (classic: a fifth; control: a tenth), and the measured queen-regime
    # split of those mates -- 23 both-queens-on / 17 exactly-one / 9
    # queenless -- partitions the evidence between these two. Both are
    # cost-class ZERO like the H1 pair: a startup formula and the same
    # root-seam boolean the landed kend+fresh fix already pays for.
    #
    # `kmid`: steeper K_MID edge-vs-centre gradient. Classic's fitted
    # K_MID slopes ~60-100 cp home-vs-centre; at median depth 10 with no
    # king-ring term and no check concept that demonstrably does not keep
    # the king out of assembling attacks. Add a linear centre-manhattan
    # gradient, ZERO-CENTRED at the middle ring (sum runs 2..14, so
    # 6*sum - 48 gives corner +36, e1 0, g1 +24, centre -36): the material
    # mean is untouched, only the slope steepens, roughly doubling
    # classic's. Both kings read it via the 119-i mirror (the kend
    # symmetry argument). K_END is NOT touched -- this is kact's exact
    # mirror, and the two compose in either order (disjoint anchors: the
    # anchor here is the MATE_LOWER line, deliberately outside the
    # K_MID/K_END definition that pend and kact rewrite).
    "kmid": (
        "MATE_LOWER = 60000 - 13 * 929\n",
        "K_MID = tuple(x and x + 6 * (abs(2 * (i // 10) - 11) + abs(2 * (i % 10) - 9)) - 48\n"
        "   for i, x in enumerate(K_MID))\n"
        "MATE_LOWER = 60000 - 13 * 929\n",
    ),
    # ===================================================================
    # TOMBSTONE: `khold`, `khold2` -- FAMILY CLOSED 2026-08-15, neither
    # ever lands. Both rewrote the SAME seam line pend's landing (61b1a51)
    # also rewrote, so their shared anchor is gone the same way oldtm's
    # and steptm's went with pooltm's landing (5f16bae) -- this is that
    # same tombstone convention, not a new one.
    #
    # THE MATE-CONVERSION GATE DECIDED BETWEEN THEM FIRST (ad292ae, KQK
    # directive: "promote king->center as soon as either queen leaves").
    # `khold` FAILS kqk-approach (7/8: king a1->b1 then 18 moves of
    # shuffle at halfmove-clock 36, and converts kqk-mid slower, 9 vs 6) --
    # "khold drops to mechanism control and must never land." It never
    # reached an independent Elo screen because of this: built and
    # gate-checked only (6567bc4), "no Elo claimed; screen staged, not
    # armed".
    #
    # `khold2` PASSED the same gate (8/8, move-for-move identical to base)
    # and went on to TWO closed Elo screens:
    #   ALONE   first attempt died at ~828 games -- shared-scratchpad
    #           arena clobbered by another lane, numbers unrecoverable
    #           (ec70bd8). Clean rerun (bbc1969): 1000 games, +2.43 +/-
    #           7.24, LLR never left the middle -- UNDECIDED (LB -4.81, UB
    #           +9.67), no land, no drop. Paired with kmid's own null
    #           (+2.08 +/- 16.58, ec70bd8, kmid stays -- disjoint anchor,
    #           still builds, NOT retired here): "the H2 king-safety seam
    #           is not a source of Elo at this budget."
    #   ON pend the pre-registered hand-written `pendkhold2` mod (built
    #           because khold2 and pend share this same seam line and
    #           cannot dot-compose) SCREENED AND LOST: -10.78 +/- 6.96,
    #           95% [-17.75, -3.81], H0 accepted at 774 of a 1000-game cap
    #           (78ff222) -- khold2's marginal contribution on top of the
    #           landed pend is measurably NEGATIVE, not the expected
    #           straddle.
    # kact closed the same H1/H2 programme at -33.07 +/- 15.98, SPRT DROP
    # (ec70bd8) -- every arm this programme raised on the king is now
    # closed by measurement and none of them lands.
    #
    # LIVE ANCHOR, read before re-creating either: the shared anchor
    # `pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else
    # K_END` is GONE from the baseline -- pend's landing rewrote this
    # exact line to the `end = ...` form. Re-creating `khold` or `khold2`
    # raises there today: SAFE, the same designed failure as oldtm's and
    # steptm's.
    #
    # TO REBUILD FOR REPRODUCTION: check out `5457f27` (the commit
    # immediately before pend's landing at `61b1a51`) or earlier -- the
    # anchor is intact there and both mods build. Verified directly at
    # the two commits that actually measured `khold2` (`ad292ae`, its
    # build, and `bbc1969`, its clean rerun): both predate `61b1a51` and
    # both still have the anchor. Do not build either mod at or after
    # `61b1a51`.
    # ===================================================================
    # `pendkhold2`: THE HAND-WRITTEN COMBINATION, written because the pair
    # cannot dot-compose. `khold2` and `pend` both rewrite the queens-off seam
    # line, so `khold2.pend` and `pend.khold2` raise in either order and the
    # standing rule (ledger, the H1 pre-registration) is that a combined mod
    # must be WRITTEN and screened as its own arm. `pend` has since LANDED, so
    # this mod is that combination expressed as khold2's MARGINAL contribution
    # on top of the shipped pend seam -- build it as `pendkhold2` on `base`.
    #
    # IT IS EXACTLY pend + khold2, AND NOTHING ELSE. The two parents touch the
    # same line but not the same DECISION, and the composition keeps each
    # parent's own predicate:
    #
    #   `end` (pend's)   = at least one queen OFF   -> governs the PAWN table
    #   khold2's clause  = at least one queen ON, AND root non-pawn material
    #                      above piece["Q"]         -> governs the KING table
    #
    # These are different predicates, and that is the whole point of the pair:
    # `pend` never changed the king's CONDITION (it only re-expressed the base
    # condition through `end` so the pawn table could share the test), while
    # khold2 changes the king's condition and nothing else. So on the shared
    # line khold2 wins, `end` survives for the pawn line it was introduced for,
    # and every line below attributes to exactly one parent:
    #
    #   end = ...            pend      (unchanged from the landed seam)
    #   heavy = ...          khold2    (its first replacement line, verbatim)
    #   pst["K"] = ...       khold2    (its second replacement line, verbatim)
    #   pst["P"] = ...       pend      (unchanged, below the anchor)
    #
    # The anchor deliberately stops before the `pst["P"]` line so that line is
    # untouched text rather than text this mod re-asserts -- if `pend`'s pawn
    # line ever moves, this mod raises instead of silently reinstating an old
    # copy of it.
    #
    # SCREENED 2026-08-15 AND LOST: -10.78 +/- 6.96, 95% [-17.75, -3.81], H0
    # accepted at 774 of a 1000-game cap. The registered bar was a pentanomial
    # LOWER bound above zero; the measured UPPER bound is below zero, so this
    # is not the expected straddle -- khold2's marginal contribution on top of
    # the landed pend is measurably NEGATIVE. Kept as a built, priced, measured
    # arm (it is the control for any future attempt at this seam), NOT as a
    # candidate. See nnue_4k/MEASUREMENTS.md for the mechanism reading.
    "pendkhold2": (
        '        end = "Q" not in pos.board or "q" not in pos.board\n'
        '        pst["K"] = K_END if end else K_MID\n',
        '        end = "Q" not in pos.board or "q" not in pos.board\n'
        '        heavy = sum(piece[c] for c in pos.board.upper() if c in "NBRQ")\n'
        '        pst["K"] = K_MID if heavy > piece["Q"] and ("Q" in pos.board or "q" in pos.board) else K_END\n',
    ),
    # ---- PAWN-STRUCTURE MACHINERY, PRICED BY BUILDING ---------------------
    # The 2026-08-14 H1 registration recorded the passer delta-rule as
    # "DESIGNED and priced out (score/ps split returns + scan class)". That
    # was reasoning, not a build, and the number it turned on -- what a
    # per-file pawn accumulator costs in bytes and in nps -- was never
    # measured. `pdbl` measures it, on the cheapest structural term that is
    # genuinely O(1) incremental.
    #
    # WHY DOUBLED AND NOT PASSED. A per-file count integer is RANK-BLIND. A
    # doubled penalty is a function of the counts alone, so a pawn moving from
    # file a to file b changes it by exactly (c_b >= 1) - (c_a >= 2) -- two
    # nibble reads, no scan. A PASSER bonus is rank-weighted, so when an enemy
    # pawn is captured off file f+1 the bonus that appears depends on the rank
    # of our pawn on file f, which the counts do not record. Making passers
    # incremental needs a SECOND accumulator (most-advanced rank per file);
    # computing them on demand is the scan class the registration named. So
    # `pdbl` is the floor: whatever a passer term would cost, it costs at
    # least this, and this is the part that is cheap.
    #
    # THE FIELD IS TWO-SIDED AND SWAPS ON ROTATE. `w` packs our eight file
    # counts in nibbles 0-7 and theirs in nibbles 8-15; rotate() exchanges the
    # halves, which is what makes the mirrored board's opponent view correct
    # for free. Files are counted in the MOVER's orientation, so a file index
    # is (square % 10) - 1 for us and 8 - (square % 10) for them.
    #
    # The term rides in value(), NOT only in score: `value(move)` has to stay
    # an exact delta of `score` or move ordering, the QS admission gate and
    # the futility test all start reading a different quantity than the search
    # returns. That requirement is the whole reason this is expensive.
    "pdbl": [
        ('class Position(namedtuple("P", "board score wc bc ep kp r")):\n',
         'class Position(namedtuple("P", "board score wc bc ep kp r w")):\n'),
        ("            119 - self.kp if self.kp and not n else 0,\n"
         "            self.board,\n"
         "        )\n",
         "            119 - self.kp if self.kp and not n else 0,\n"
         "            self.board, self.w >> 32 | (self.w & 0xFFFFFFFF) << 32,\n"
         "        )\n"),
        ("        board, r = self.board, self.r\n"
         "        wc, bc, ep, kp = self.wc, self.bc, 0, 0\n"
         "        score = self.score + self.value(move)\n",
         "        board, r = self.board, self.r\n"
         "        wc, bc, ep, kp = self.wc, self.bc, 0, 0\n"
         "        score = self.score + self.value(move)\n"
         "        w = self.w\n"
         '        if p == "P":\n'
         "            w += (1 << 4 * (j % 10 - 1)) - (1 << 4 * (i % 10 - 1))\n"
         "            if A8 <= j <= H8: w -= 1 << 4 * (j % 10 - 1)\n"
         "            if j == self.ep: w -= 1 << 4 * (8 + 8 - (j + S) % 10)\n"
         '        if q == "p": w -= 1 << 4 * (8 + 8 - j % 10)\n'),
        ("        return Position(r, -score, bc, wc,\n"
         "                        119 - ep if ep else 0, 119 - kp if kp else 0, board)\n",
         "        return Position(r, -score, bc, wc,\n"
         "                        119 - ep if ep else 0, 119 - kp if kp else 0, board,\n"
         "                        w >> 32 | (w & 0xFFFFFFFF) << 32)\n"),
        ("    return Position(board, score, wc, bc, ep, kp, board[::-1].swapcase())\n",
         "    w = 0\n"
         "    for i, p in enumerate(board):\n"
         '        if p == "P": w += 1 << 4 * (i % 10 - 1)\n'
         '        elif p == "p": w += 1 << 4 * (8 + 8 - i % 10)\n'
         "    return Position(board, score + dbl(w), wc, bc, ep, kp,\n"
         "                    board[::-1].swapcase(), w)\n"),
        # the doubled term itself, and the O(1) delta value() adds
        ("def from_board(board, wc=(True, True), bc=(True, True), ep=0, kp=0):\n",
         "def dbl(w):\n"
         '    """-12 cp per doubled pawn, ours minus theirs. Only called from\n'
         "    from_board -- every other site updates the delta in O(1).\"\"\"\n"
         "    return -12 * sum(max(0, w >> 4 * f & 15) - 1 for f in range(8)\n"
         "                     if w >> 4 * f & 15) \\\n"
         "        + 12 * sum(max(0, w >> 4 * f & 15) - 1 for f in range(8, 16)\n"
         "                   if w >> 4 * f & 15)\n"
         "\n"
         "\n"
         "def from_board(board, wc=(True, True), bc=(True, True), ep=0, kp=0):\n"),
        ('        if p == "P":\n'
         "            if A8 <= j <= H8:\n"
         "                score += pst[prom][j] - pst[\"P\"][j]\n"
         "            if j == self.ep:\n"
         '                score += pst["P"][119 - (j + S)]\n'
         "        return score\n",
         '        if p == "P":\n'
         "            if A8 <= j <= H8:\n"
         "                score += pst[prom][j] - pst[\"P\"][j]\n"
         "                # the pawn LEAVES its file and does not arrive on one\n"
         "                score += 12 * (self.w >> 4 * (i % 10 - 1) & 15 >= 2)\n"
         "            elif i % 10 != j % 10:\n"
         "                # doubled delta, O(1): our count on the destination file\n"
         "                # rises and on the source file falls. GUARDED on the\n"
         "                # file actually changing -- a straight push has a == b,\n"
         "                # where the formula (c_b >= 1) - (c_a >= 2) reads 1 for a\n"
         "                # single pawn and invents a phantom doubled pawn on every\n"
         "                # push. Caught by the from-scratch rebuild check, not by\n"
         "                # the delta check: value() and move() agreed with each\n"
         "                # other perfectly while both were wrong.\n"
         "                score -= 12 * ((self.w >> 4 * (j % 10 - 1) & 15 >= 1)\n"
         "                               - (self.w >> 4 * (i % 10 - 1) & 15 >= 2))\n"
         "            if j == self.ep:\n"
         '                score += pst["P"][119 - (j + S)]\n'
         "                score -= 12 * (self.w >> 4 * (16 - (j + S) % 10) & 15 >= 2)\n"
         '        if q == "p":\n'
         "            score -= 12 * (self.w >> 4 * (16 - j % 10) & 15 >= 2)\n"
         "        return score\n"),
    ],
    # ---- SEARCH: the root gamma seed. THIS IS NOT AN EVAL MOD -------------
    # `search()` starts every search at gamma = 0 and bisects. The root stores a
    # move ONLY on a fail-high, so the node count of the first root fail-high --
    # the "first yield" -- is the earliest the engine can answer at all, and
    # both stop conditions are polled at `nodes % 2048 == 0`. A build whose
    # first yield exceeds 2048 answers `bestmove (none)`.
    #
    # Seeding BELOW the static score makes the first probe cheap and one-sided:
    # it is a fail-HIGH, which is the kind that produces a move. `pos.score`
    # alone is a trade, not a fix -- it helps every fit and makes the incumbent
    # WORSE (780 -> 2,920), because seeding at the true value makes the first
    # probe a coin flip. The -150 offset is what makes it one-sided.
    #
    # Eight builds measured: every fitted eval this lane has produced sits at
    # or over the 2,048 cliff, and this clears all of them. It is a SEARCH
    # change and lands in the search lane, not here.
    "seed": ("\n        gamma = 0\n", "\n        gamma = pos.score - 150\n"),
    # Cap the null-move score at static eval plus one score bucket, exactly as
    # classic does. Ours has never capped it: the score is both a looser cutoff
    # trigger AND this node's returned value, so an inflated pass estimate
    # propagates into the tt and the MTD bisection.
    "cap": (
        "                score = -self.bound(pos.rotate(n=True), 1 - gamma, depth - 3)\n",
        "                score = min(pos.score + EVAL_ROUGHNESS,\n"
        "                    -self.bound(pos.rotate(n=True), 1 - gamma, depth - 3))\n",
    ),
    # Late move reductions off. Threshold-triggered (val < LMR), so setting the
    # threshold to 0 disables it without touching the loop.
    "nolmr": ("\nLMR = 60\n", "\nLMR = 0\n"),
    # =====================================================================
    # ENTRY-NATIVE CONSTANT COHORT, round 1 (2026-08-16). Ten arms, each ONE
    # value change and NOTHING else -- no line added, no line removed, source
    # byte count IDENTICAL to the baseline's 46244 on every one of them.
    #
    # WHY THIS COHORT EXISTS. The entry's search constants are all INHERITED
    # from classic and none was ever tuned HERE: QS/QS_A/EVAL_ROUGHNESS are
    # byte-identical to sunfish.py's (40/140/15), the null guard is classic's
    # `abs(pos.score) < 500` and R=3, and `LMR = 60` was measured only
    # ON-vs-OFF (+38.9 +/- 19.1, 2026-08-13) -- never scanned. Meanwhile the
    # search around them changed completely: IIR replaced IID, LMR arrived,
    # `pend`/`kend` changed the eval's endgame, `pooltm` changed the clock,
    # and MTD-bi replaced classic's driver. Constants tuned for one pruning
    # envelope are not tuned for another.
    #
    # THE ZERO-BYTE RULE, and what it costs. Only values that pack at
    # <= 3405 B are admitted; the packed size is what the 4096 ceiling
    # measures, and lzma is context-sensitive, so a same-length digit swap
    # can still cost +1..+3 B. Measured over the WHOLE neighbourhood
    # (scan 2026-08-16): `EVAL_ROUGHNESS` 20/25/30 cost +1/+2/+2 while 10/12
    # are free and 40 GIVES a byte; every `QS` but 60 costs +2; every `QS_A`
    # but 160 costs +2; the null MIN-DEPTH (`depth > 2`) costs +3 at every
    # value and is therefore EXCLUDED from this cohort entirely. The zero-byte
    # rule is thus a real constraint on the grid, not a formality -- it is why
    # the down-steps here are short (ER cannot go below 10 in two digits) and
    # why the cohort is not a symmetric lattice. Recorded so a reader does not
    # mistake the gaps for a design choice.
    #
    # Each arm is standalone; they are NOT meant to compose (two of them touch
    # the same LMR line and would raise, as designed).
    #
    # --- EVAL_ROUGHNESS: the MTD-bi bisection stop width -----------------
    # `while lower < upper - EVAL_ROUGHNESS` is how many root probes a depth
    # costs. Coarser = fewer probes = more depth for the same 20000 nodes, and
    # -- unlike #205's fuel oracle -- it adds NO table-state dependence: the
    # width is a driver property, not part of any key. It is the cleanest test
    # available of the #205 finding that bought depth does not convert, since
    # here the depth is bought with no instability at all.
    "er40": ("\nEVAL_ROUGHNESS = 15\n", "\nEVAL_ROUGHNESS = 40\n"),
    "er10": ("\nEVAL_ROUGHNESS = 15\n", "\nEVAL_ROUGHNESS = 10\n"),
    # --- QS admission: val_lower = QS - depth * QS_A ----------------------
    # QS is the depth-0 intercept, so it governs the QSearch nodes that
    # dominate the budget: 60 admits only clearly-winning captures. 60 is the
    # ONLY zero-byte value on this knob.
    "qs60": ("\nQS = 40\nQS_A", "\nQS = 60\nQS_A"),
    # QS_A is the per-ply slope; larger makes val_lower MORE negative, i.e.
    # admits MORE quiet moves from depth 1 up (-100 -> -120 at depth 1).
    # 160 is the ONLY zero-byte value on this knob.
    "qsa160": ("\nQS_A = 140\n", "\nQS_A = 160\n"),
    # --- LMR threshold: reduce quiet moves with static value below it -----
    # 75 is the value classic tuned out of its 9,310-game campaign and that
    # the #205 port DELIBERATELY DROPPED ("the cheap decomposition follow-up
    # if this arm wins"). The port lost, but the constant was never the thing
    # tested -- it arrives here standalone, at ZERO bytes instead of the
    # port's +71, with the entry's own verified reduction left intact.
    "lmr75": ("\nLMR = 60\n", "\nLMR = 75\n"),
    # The opposite sign, and the control for it: fewer moves qualify.
    "lmr40": ("\nLMR = 60\n", "\nLMR = 40\n"),
    # --- the null-move static guard --------------------------------------
    # `abs(pos.score) < 500` blocks the null in lopsided positions. #205
    # deleted it outright and the composite failed, but the decomposition put
    # the damage on the fuel oracle (13 MTD crossings vs the gate's 4), not on
    # the guard. Widening it to 900 buys more null cuts and adds NO new
    # table-state dependence -- |pos.score| is a function of the position.
    "nlim900": ("abs(pos.score) < 500", "abs(pos.score) < 900"),
    # --- the null reduction, BOTH probe sites ----------------------------
    # R 3 -> 4 on the cut probe AND on the mate-band verification probe. They
    # move together on purpose: the verification certifies the SAME claim the
    # cut probe made, so leaving it at depth - 3 would verify a different
    # search than the one being trusted. Two sites, one deviation.
    "nred4": [
        ("                score = -self.bound(pos.rotate(n=True), 1 - gamma, depth - 3)\n",
         "                score = -self.bound(pos.rotate(n=True), 1 - gamma, depth - 4)\n"),
        ("                elif score < gamma or self.bound(pos.rotate(n=True),\n"
         "                        1 - MATE_LOWER, depth - 3) >= 1 - MATE_LOWER:\n",
         "                elif score < gamma or self.bound(pos.rotate(n=True),\n"
         "                        1 - MATE_LOWER, depth - 4) >= 1 - MATE_LOWER:\n"),
    ],
    # --- futility pruning depth ------------------------------------------
    # The largest pruning-envelope step available at zero bytes: extend the
    # sorted-descending futility break from depth <= 1 to depth <= 2, so a
    # whole extra ply discards its tail once pos.score + val < gamma. High
    # variance in both directions, which is exactly what a 50-game selector
    # CAN resolve.
    "fut2": ("if depth <= 1 and pos.score + val < gamma",
             "if depth <= 2 and pos.score + val < gamma"),
    # --- the LMR count gate ----------------------------------------------
    # `cnt > 2` -> `cnt > 0`: reduce from the SECOND move rather than the
    # fourth. This is #205's intrinsic-gate DIRECTION (reduce earlier) asked
    # in the entry's own count condition instead of importing classic's null
    # guard -- zero bytes against the port's +71, and no new table-state
    # dependence beyond the `cnt` term the baseline already has.
    "cnt0": ("red = LMR and depth > 2 and cnt > 2 and val < LMR",
             "red = LMR and depth > 2 and cnt > 0 and val < LMR"),
    # --- the rest of the zero-byte pool, kept for the liveness map --------
    # These were generated and probed (60 first-yield positions, 20000 nodes,
    # bestmove-divergence vs the base) BEFORE any arm was chosen, so that no
    # game is spent on a constant that cannot change a move. The map is in
    # nnue_4k/MEASUREMENTS.md under the round-1 pre-registration.
    "fut0": ("if depth <= 1 and pos.score + val < gamma",
             "if depth <= 0 and pos.score + val < gamma"),
    "fut3": ("if depth <= 1 and pos.score + val < gamma",
             "if depth <= 3 and pos.score + val < gamma"),
    "lred3": ("1 - gamma, depth - 2 if red else depth - 1",
              "1 - gamma, depth - 3 if red else depth - 1"),
    "lred4": ("1 - gamma, depth - 2 if red else depth - 1",
              "1 - gamma, depth - 4 if red else depth - 1"),
    "nred2": [
        ("                score = -self.bound(pos.rotate(n=True), 1 - gamma, depth - 3)\n",
         "                score = -self.bound(pos.rotate(n=True), 1 - gamma, depth - 2)\n"),
        ("                elif score < gamma or self.bound(pos.rotate(n=True),\n"
         "                        1 - MATE_LOWER, depth - 3) >= 1 - MATE_LOWER:\n",
         "                elif score < gamma or self.bound(pos.rotate(n=True),\n"
         "                        1 - MATE_LOWER, depth - 2) >= 1 - MATE_LOWER:\n"),
    ],
    "nred5": [
        ("                score = -self.bound(pos.rotate(n=True), 1 - gamma, depth - 3)\n",
         "                score = -self.bound(pos.rotate(n=True), 1 - gamma, depth - 5)\n"),
        ("                elif score < gamma or self.bound(pos.rotate(n=True),\n"
         "                        1 - MATE_LOWER, depth - 3) >= 1 - MATE_LOWER:\n",
         "                elif score < gamma or self.bound(pos.rotate(n=True),\n"
         "                        1 - MATE_LOWER, depth - 5) >= 1 - MATE_LOWER:\n"),
    ],
    "nlim250": ("abs(pos.score) < 500", "abs(pos.score) < 250"),
    "nlim300": ("abs(pos.score) < 500", "abs(pos.score) < 300"),
    "er12": ("\nEVAL_ROUGHNESS = 15\n", "\nEVAL_ROUGHNESS = 12\n"),
    "lmr30": ("\nLMR = 60\n", "\nLMR = 30\n"),
    "lmr50": ("\nLMR = 60\n", "\nLMR = 50\n"),
    # --- THE DEADLINE POLL, the one search constant a fixed-node instrument
    # CANNOT see. `self.nodes % 2048` decides how often the in-search clock is
    # read, so under `go nodes` it is exactly inert and round 1 was structurally
    # blind to it. It is also the only constant left that touches nps, which is
    # where both standing instruments say the entry's remaining edge lives.
    #
    # It SHIPS: unlike the node-cap poll on the line above it, this one is NOT
    # inside `# minifier-hide`, so it is in the artifact. Verified, not assumed.
    #
    # Free at 1024, 4096 and 8192 (all pack to 3405 B, measured). 4096 is the
    # conservative step and the only one played: a longer poll widens the
    # worst-case overrun past `searcher.deadline`, and this engine has a
    # forfeit history, so 8192 stays registered-not-run pending 4096's result.
    "poll4096": ("if self.nodes % 2048 == 0 and time.time() > self.deadline: raise Stop",
                 "if self.nodes % 4096 == 0 and time.time() > self.deadline: raise Stop"),
    "poll8192": ("if self.nodes % 2048 == 0 and time.time() > self.deadline: raise Stop",
                 "if self.nodes % 8192 == 0 and time.time() > self.deadline: raise Stop"),
    # --- INTERNAL ITERATIVE REDUCTION, the two constants round 1 missed ---
    # `if depth > 2 and killer is None: depth -= 1` carries a min-depth AND a
    # reduction amount, and neither was in the round-1 cohort -- an honest
    # omission, found while round 1 was already running and therefore held
    # back rather than slipped in. IIR is a MEASURED mechanism here (+22.3
    # +/- 16.0, `iirk.noiid`, 2026-08-13) whose own constants are untuned.
    #
    # BUT THE WHOLE IIR AXIS IS OFF THE ZERO-BYTE GRID. Measured, every
    # neighbour (packed, base 3405):
    #     depth -= 2                    3406  +1
    #     depth -= 3                    3406  +1
    #     depth > 1                     3406  +1
    #     depth > 3 / > 4               3407  +2
    #     depth > 3 and depth -= 2      3408  +3
    # Not one free value. These two are kept as BUILT-BUT-UNPLAYED candidates:
    # at 691 spare bytes a +1 arm is affordable, but affordable is a different
    # question from free and it is the coordinator's to answer, not this
    # generator's. Do not slip them into a zero-byte cohort.
    "iird2": ("if depth > 2 and killer is None: depth -= 1",
              "if depth > 2 and killer is None: depth -= 2"),
    "iirmin4": ("if depth > 2 and killer is None: depth -= 1",
                "if depth > 4 and killer is None: depth -= 1"),
    # =====================================================================
    # THE #205 PORT: classic's tuned null shaping, and its intrinsic LMR gate.
    #
    # SCREENED AND NOT LANDED, 2026-08-16. Fixed-node SPRT, 1000 games at the
    # cap: **UNDECIDED, +5.91 +/- 17.25, 95% [-11.33, +23.17]**, 0 illegal, 0
    # forfeits. The interval EXCLUDES classic's own +48.25 +/- 27.03, so the
    # transfer claim is refuted at this precision -- and at +71 packed bytes it
    # fails the exchange rate outright (0.08 Elo/byte at the point estimate
    # against LMR's ~1.8). The mod is KEPT, not deleted: it is the base for the
    # `nofuel`/`nogate` decomposition below, and a reader who re-runs it
    # unaware of this verdict would spend another 1000 games. Full entry and
    # the decomposition are in nnue_4k/MEASUREMENTS.md, 2026-08-16.
    #
    # Classic merged #205 ("Land tuned null shaping and intrinsic LMR", master
    # bf44c52) out of a 9,310-game tuning campaign and measured the search
    # change at +48.25 +/- 27.03. The entry's search forked from classic long
    # ago, so #205 is not a patch that applies -- each part is mapped to the
    # entry's own site here, and the parts that do NOT map are named rather
    # than quietly bundled.
    #
    # CARRIED:
    #  * The TWO-REGIME null. Classic splits the pass into a score candidate
    #    below depth 6 and a FUEL ORACLE from 6 on. The entry has only ever had
    #    the score candidate, at every depth > 2; the oracle is new here.
    #  * NULL_MARGIN = -200 with the depth - 7 probe, #205's tuned pair, ported
    #    unchanged. They arrive WITH the mechanism they tune, so there is no
    #    entry-side incumbent for them to overwrite.
    #  * The static guard drop. Classic deleted `abs(pos.score) < 500` from
    #    both regimes; the entry carries the same test and it goes here too.
    #  * The intrinsic LMR GATE. Classic reduces a low-value child whenever the
    #    null guard holds, with no count condition; the entry only reduced past
    #    the third move. The gate becomes the UNION, so every move the entry
    #    already reduced is still reduced and the guard adds the early ones.
    #
    # DROPPED, each for a stated reason:
    #  * LMR = 75. Not ported. 60 already governs measured entry behaviour
    #    (+38.9 +/- 19.1 fixed-node, ledger 2026-08-13) and classic tuned 75
    #    for a reduction classic did not previously have. Moving it would
    #    retune a measured mechanism under cover of a port; it is the cheap
    #    decomposition follow-up if this arm wins.
    #  * The IID removal. ALREADY DONE here, and better: `iirk.noiid` landed
    #    internal iterative REDUCTION in its place (+22.3 +/- 16.0, 2026-08-13),
    #    which answers the same question without spending a shallow search.
    #  * The unverified reduction. Classic's intrinsic LMR never re-searches;
    #    the entry re-searches a reduced child that fails high. That
    #    verification is an entry invariant, so the ported gate FEEDS the
    #    entry's verified reduction rather than replacing it.
    #  * score_move, the depth 2-3 static cap, the depth-1 tail widening, and
    #    classic's `proof` certificate. Those are #193's and the entry's own
    #    lineage respectively -- out of this port's scope.
    #
    # INVARIANTS. `d` shortens the RECURSION only: the node still keys and
    # stores under nominal `depth`, and the QS admission still reads nominal
    # `depth`, so no shallow value is ever filed under a deep key. The
    # mate-band verification probe on the score candidate is kept verbatim --
    # it is the entry's device, not classic's, and the zero-illegal bestmove
    # floor rests on it. Gate ladder green on both arms (mate-conversion 8/8,
    # legality 130/130 at both budget paths on laptop AND box, first-yield
    # MAX 676/2048 identical, empty-dir smoke).
    #
    # WHAT IS *NOT* HELD, measured rather than reasoned. The target window
    # `pos.score + NULL_MARGIN` is fixed by (pos, depth), but the probe that
    # reads it goes through `bound()`, which may satisfy itself from a table
    # entry -- so the ply `d` gives up is a function of TABLE STATE, not of
    # (pos, depth) alone. That is a genuine new break of one-value-per-key, on
    # top of the `cnt` term, and it shows: over the 60 first-yield positions at
    # the screen's own 20000-node budget the MTD driver's bracket-crossing
    # tripwire fires **1 time on the base and 13 times on this arm**. The
    # guards in search() exist for exactly this and clamp it, and the number is
    # pre-registered in the ledger so it cannot be re-read after the result.
    # Claiming position-determinism here (as the upstream comment does) would
    # be a model/code divergence, so it is not claimed.
    #
    # MECHANISM CHECK, same 60 positions and budget: the arm reaches a deeper
    # final depth on 53, shallower on 0, equal on 7 -- mean 9.93 -> 12.13 plies
    # for the same 20000 nodes. The port prunes as intended; whether the extra
    # depth is worth the extra instability is what the screen is for.
    "n205": [
        # 1. The tuned margin arrives with the oracle it belongs to.
        ("# Late move reduction: reduce quiet moves whose static value is below this,\n"
         "# once past the first few in the sorted list. 0 disables (classic parity).\n"
         "LMR = 60\n",
         "# Late move reduction: reduce quiet moves whose static value is below this,\n"
         "# once past the first few in the sorted list. 0 disables (classic parity).\n"
         "LMR = 60\n"
         "# Target margin of the deep-null fuel probe (depth >= 6): the pass must beat\n"
         "# pos.score + NULL_MARGIN for real moves to burn two plies instead of one.\n"
         "NULL_MARGIN = -200\n"),
        # 2. Free (minifier-hide strips it): the tuner sees the new knob.
        ("    EVAL_ROUGHNESS = (0, 50),\n",
         "    EVAL_ROUGHNESS = (0, 50),\n"
         "    NULL_MARGIN = (-300, 300),\n"),
        # 3. The score candidate keeps its body and loses its static guard and
        #    its deep half.
        ('            if not root and depth > 2 and abs(pos.score) < 500 and any(c in pos.board for c in "RBNQ"):\n',
         '            if not root and 2 < depth < 6 and any(c in pos.board for c in "RBNQ"):\n'),
        # 4. The oracle is appended after it, before the QSearch stand-pat.
        ("                elif score < gamma or self.bound(pos.rotate(n=True),\n"
         "                        1 - MATE_LOWER, depth - 3) >= 1 - MATE_LOWER:\n"
         "                    yield None, score\n",
         "                elif score < gamma or self.bound(pos.rotate(n=True),\n"
         "                        1 - MATE_LOWER, depth - 3) >= 1 - MATE_LOWER:\n"
         "                    yield None, score\n"
         "\n"
         "            # THE FUEL ORACLE. From depth 6 on the pass stops being a score\n"
         "            # candidate and becomes a question about DEPTH: one probe at the\n"
         "            # fixed target pos.score + NULL_MARGIN -- a window determined by\n"
         "            # (pos, depth) alone, so a fail-soft report is side-exact at it --\n"
         "            # decides whether real moves burn one ply or two. Nominal depth\n"
         "            # still keys the table and the QS admission; only the recursion\n"
         "            # shortens, so a deep null cut becomes a reduction and never a\n"
         "            # virtual score. Its guard doubles as the intrinsic LMR gate\n"
         "            # below, which is why the two landed as one change upstream.\n"
         "            d = depth\n"
         '            guard = depth >= 6 and any(c in pos.board for c in "RBNQ")\n'
         "            if guard:\n"
         "                target = pos.score + NULL_MARGIN\n"
         "                d -= -self.bound(pos.rotate(n=True), 1 - target, depth - 7) >= target\n"),
        # 5. One child rule for both streams. Factoring it is what lets the
        #    killer and its re-appearance in the sorted list agree on depth at a
        #    guard node, so the second one is a table hit rather than a second
        #    real search -- the cost-equality #194's lineage was built around.
        ("            if killer and pos.value(killer) >= val_lower:\n"
         "                yield killer, -self.bound(pos.move(killer), 1 - gamma, depth - 1)\n",
         "            # One child search for both streams, at one depth rule, so the\n"
         "            # killer and its later re-appearance cost the same and the second\n"
         "            # is a table hit. `late` is the entry's own count condition;\n"
         "            # `guard` is #205's intrinsic one. The killer is never late.\n"
         "            def child(move, val, late=False):\n"
         "                red = LMR and val < LMR and (guard or late)\n"
         "                score = -self.bound(pos.move(move), 1 - gamma, d - 2 if red else d - 1)\n"
         "                if red and score >= gamma:\n"
         "                    score = -self.bound(pos.move(move), 1 - gamma, d - 1)\n"
         "                return move, score\n"
         "\n"
         "            if killer and pos.value(killer) >= val_lower:\n"
         "                yield child(killer, pos.value(killer))\n"),
        # 6. The loop hands its count condition to the same rule.
        ("                red = LMR and depth > 2 and cnt > 2 and val < LMR\n"
         "                score = -self.bound(pos.move(move), 1 - gamma, depth - 2 if red else depth - 1)\n"
         "                if red and score >= gamma:\n"
         "                    score = -self.bound(pos.move(move), 1 - gamma, depth - 1)\n"
         "                yield move, score\n",
         "                yield child(move, val, depth > 2 and cnt > 2)\n"),
    ],
    # The two halves of `n205`, for decomposition. COMPOSE ONTO IT -- they are
    # `n205.nofuel` and `n205.nogate`, never standalone, and both anchors exist
    # only after `n205` has applied (so a wrong order raises, as designed).
    #
    # What they measured, 60 first-yield positions at 20000 nodes (mean final
    # depth / MTD bracket crossings; base is 9.93 / 1):
    #   n205         12.13 / 13      the full port
    #   n205.nogate  11.80 /  2      fuel oracle only  -- 1.87 of the 2.20 plies
    #   n205.nofuel  10.20 /  4      intrinsic gate only -- 0.27 plies
    # Depth is roughly additive; INSTABILITY IS SHARPLY SUPERADDITIVE. That is
    # the standing lead if anyone revisits this: `nogate` buys ~85% of the
    # depth for 2 crossings instead of 13. It has NOT been played.
    "nofuel": (
        "                d -= -self.bound(pos.rotate(n=True), 1 - target, depth - 7) >= target\n",
        "                self.bound(pos.rotate(n=True), 1 - target, depth - 7)\n"),
    "nogate": (
        "                red = LMR and val < LMR and (guard or late)\n",
        "                red = LMR and val < LMR and late\n"),
    # ===================================================================
    # TOMBSTONE: `pooltm` LANDED 2026-08-15, and `oldtm`/`steptm` went with
    # it. The pool manager is now the entry's DEFAULT time manager, applied
    # by tools/build/make_pst_entry.py (`_pooltm`, next to `_pend`).
    #
    # WHAT IT MEASURED, three regimes vs the shipped smooth budget on the
    # bench box, book3k, no adjudication (nnue_4k/MEASUREMENTS.md):
    #   60+0  +119.9 +/- 36.4   H1
    #   60+1  +136.6 +/- 35.2   H1, 262 games
    #   30+1  +124.50 +/- 38.79 H1, 288 games, Ptnml [10,14,31,45,44]
    # plus the pre-registered 1+0 hammer: 100 games, zero illegal, zero
    # (none), zero forfeits. Landed cost +65 packed bytes (3340 -> 3405)
    # measured on the post-pend base -- NOT the +57 it measured on the
    # pre-pend 3308 base, because lzma shares one dictionary and byte
    # deltas never compose across landings. The arm sha256s and the packed
    # sha of the artifact that played the deciding match are in the ledger
    # entry "POOL DECIDING MATCH 1".
    #
    # WHICH ANCHORS SURVIVE -- the load-bearing part of a tombstone:
    #
    #   * The smooth budget line (`think = min(wtime * (1000 + 20 * winc)
    #     ...`) IS GONE from the baseline. It was pooltm's first anchor and
    #     it was ALSO the shared anchor of `oldtm` and `steptm`, which is
    #     why those two retire here rather than merely being unused: their
    #     anchor stopped existing, so re-creating any of the three would
    #     RAISE loudly. That is the safe failure.
    #
    #   * `if "movetime" in times: think -= max(think * .05, .03)` STILL
    #     OCCURS EXACTLY ONCE. pooltm kept it and appended `soft = min(soft
    #     / 1000, think)` after it. Re-creating pooltm would therefore
    #     append a SECOND rescale of `soft` WHILE THE OCCURS-EXACTLY-ONCE
    #     CHECK PASSED CLEANLY -- soft in seconds divided by 1000 again, a
    #     ~0-second soft limit, i.e. an arm that looks correctly generated
    #     and plays depth-1 moves. This is the `iirk`/`fresh` failure mode
    #     and no automated check catches it. DO NOT re-create this mod.
    #
    #   * `best, cand, d0 = None, None, 1`, the `depth > d0` block and the
    #     `think * 0.8` break are all gone (rewritten); re-applying those
    #     three edits would raise.
    #
    # THE ONE MEASURED HOLE, recorded with the landing rather than buried:
    # at a 1-second sudden-death clock the pool scores -209.91 +/- 60.11,
    # because P = max(0, 1000 - 42*200) = 0 makes soft = 0 and the A/4
    # clamp unreachable. Mechanism, options and the open scoping decision
    # are in make_pst_entry.py's `_pooltm` comment and in the ledger.
    # ===================================================================
    # PHASE-M ARM. M falls with the move number instead of standing at 40:
    # Lc0's phase curve in its cheapest form, 46 at move one down to a floor
    # of 20 from ply 52 on, so spending rises through the middlegame where
    # depth buys the most. len(hist) is the driver's own counter (plies
    # played), the same input the classic twin's phase_m arm reads.
    #
    # ITS ANCHOR MOVED WITH THE LANDING, and this note is the reason the
    # tombstone above exists. `M = 40` used to be a line `pooltm` CREATED,
    # so `phasem` only composed as `pooltm.phasem` and raised in any other
    # form. The pool is now the baseline, so `M = 40` occurs exactly once in
    # the entry itself and `phasem` is a PLAIN mod: build it as `phasem`.
    # `pooltm.phasem` no longer exists and will raise on the pooltm name.
    "phasem": ("            M = 40\n", "            M = max(20, 46 - len(hist) / 2)\n"),
    # CORRECTION HISTORY, interior-only. A running average of (search value -
    # static value) over positions sharing a pawn skeleton, added to the
    # static score wherever the search trusts it -- here that is exactly one
    # consumer, the depth<=1 futility test.
    #
    # REBUILT 2026-08-13. The original prototype (e_pstcorrhist2.py, ledger
    # entry "corrhist: the key, not the correction") was never committed and
    # no copy survives on any machine, so this is a re-implementation from
    # that entry's written spec -- pawn-skeleton key via str.translate,
    # +/-120cp clamp, 7/8 decay, mates excluded, key NOT computed in QS. Its
    # node and nps numbers are therefore re-measured from scratch rather than
    # inherited; the old 0.70x/0.79x/0.89x column belongs to a build we no
    # longer have.
    #
    # Why interior-only is the whole design: computing the key at every node
    # cost 0.48x nps AND made the node count WORSE (0.80x vs 0.70x), because
    # a QS stand-pat score is largely the static eval itself, so correcting
    # it teaches the table its own output.
    #
    # THE KEY IS THE COST, so it is built from as little of the board as it
    # can be. Pawns only ever stand on ranks 2-7, which is board[31:89] in
    # the 120-char layout (rows 0-1 and 10-11 are padding, row 2 is rank 8,
    # row 9 is rank 1). Slicing first is exactly equivalent to translating
    # the whole board -- the discarded 62 characters cannot hold a pawn --
    # and it more than halves the per-node string work. `wkey` below restores
    # the full-board key so the difference can be priced rather than assumed.
    #
    # FUTILITY-BREAK SOUNDNESS (the constraint that cost -449 Elo once): the
    # loop breaks on the first move failing `static + val < gamma`, which is
    # only valid while the tested quantity descends with the sort key. `corr`
    # is computed ONCE per node, before the loop, and never changes inside
    # it, so the test is `val < gamma - pos.score - corr` -- a constant
    # threshold on the sort key `val`. The break stays sound.
    "corr": [
        ("TABLE_SIZE = 10**6\n",
         "TABLE_SIZE = 10**6\n"
         "# Correction history: clamp on the learned (search - static) offset,\n"
         "# and the translation that reduces a board to its pawn skeleton.\n"
         "CORR = 120\n"
         'PAWNS = str.maketrans("RNBQKrnbqk", "." * 10)\n'),
        ("        self.t, self.tp_move, self.h = {}, {}, set()\n",
         "        self.t, self.tp_move, self.h = {}, {}, set()\n"
         "        self.corr = {}\n"),
        ("        # Generator of moves to search in order.\n",
         "        # Correction history lookup. INTERIOR NODES ONLY: in QS the key\n"
         "        # costs more than the correction is worth, and corrects a score\n"
         "        # that is mostly the static eval it was learned from.\n"
         "        ckey, corr = None, 0\n"
         "        if depth:\n"
         "            ckey = pos.board[31:89].translate(PAWNS)\n"
         "            corr = self.corr.get(ckey, 0)\n"
         "\n"
         "        # Generator of moves to search in order.\n"),
        ("                if depth <= 1 and pos.score + val < gamma:\n",
         "                if depth <= 1 and pos.score + corr + val < gamma:\n"),
        ("                    yield (move, MATE_UPPER) if val >= MATE_LOWER else (None, pos.score + val)\n",
         "                    yield (move, MATE_UPPER) if val >= MATE_LOWER else (None, pos.score + corr + val)\n"),
        ("        # Table part 2. Every search decision is gamma-independent, so all\n",
         "        # Correction history update. Mates are excluded: a mate score is\n"
         "        # not an evaluation error, it is a different quantity, and one\n"
         "        # mate would saturate the entry for its whole pawn skeleton.\n"
         "        if ckey is not None and abs(best) < MATE_LOWER:\n"
         "            self.corr[ckey] = max(-CORR, min(CORR, (7 * corr + best - pos.score) // 8))\n"
         "            if len(self.corr) > TABLE_SIZE:\n"
         "                del self.corr[next(iter(self.corr))]\n"
         "\n"
         "        # Table part 2. Every search decision is gamma-independent, so all\n"),
    ],
    # HISTORY HEURISTIC, restored verbatim from 438ac49 (the removal commit),
    # not re-invented -- the sound form is the one that was already validated
    # against the -449 Elo soundness bug, and reconstructing it from memory is
    # how that bug would come back.
    #
    # Why it is being re-screened after being removed: the removal rested on a
    # NODE-RATIO PROXY (1.01 at completed depth 7 over 30 positions), and
    # ordering quality is not a node count -- it shows up in games. The
    # measurement also predates the king-table fix, the stale-carried-score
    # fix, LMR, and the MTD guards, every one of which changes what the search
    # does with a better-ordered list.
    #
    # SOUNDNESS, argued at EVERY consumer of iteration order, because the
    # consumer nobody re-checked is what cost -449:
    #   1. the depth<=1 futility BREAK discards the rest of the list, so it is
    #      only valid while iteration descends in static value. Hence
    #      `hh = self.hh if depth > 1 else {}`: frontier nodes sort by static
    #      value alone and the break tests exactly the quantity it sorts by.
    #   2. LMR's `cnt > 2 and val < LMR` is a NEW consumer that did not exist
    #      when this code was removed. It is a heuristic, not a contract: a
    #      reduced search that fails high is re-run at full depth, so a
    #      reordered list can cost or save work but cannot lose a cutoff.
    #   3. the ADMISSION test (`v >= val_lower`) and the futility `val` both
    #      stay the static pos.value. The sort may reorder, never re-admit.
    "hist": [
        ("        self.nodes, self.deadline = 0, 1 << 63\n",
         "        self.nodes, self.deadline = 0, 1 << 63\n"
         "        self.hh = {}                        # (piece,to) -> cutoff credit\n"),
        ("            # NOTE the iteration order is a soundness contract, not a\n"
         "            # heuristic: the futility break below discards the rest of the\n"
         "            # list, which is only valid when iteration descends in static\n"
         "            # value. A history-credit order key tried here scrambled that\n"
         "            # order and paid -449 Elo (ledger 5f5f34d); made sound, the\n"
         "            # history table measured a 1.01 node ratio -- worthless.\n"
         "            for cnt, (val, move) in enumerate(sorted(((v, m) for m in pos.gen_moves()\n"
         "                                     if (v:=pos.value(m)) >= val_lower), reverse=True)):\n",
         "            # Order key = static value + history credit; the ADMISSION test\n"
         "            # and the futility val stay the static pos.value (the sort may\n"
         "            # reorder, never re-admit).\n"
         "            # At frontier nodes (depth <= 1) the key MUST stay the static\n"
         "            # value alone: the futility break below discards the rest of\n"
         "            # the list, which is only sound when iteration is descending\n"
         "            # in val. A history-scrambled order let an early low-val move\n"
         "            # break away later non-futile moves -- the node failed low,\n"
         "            # the parent inflated, and the screen paid -449 Elo for it.\n"
         "            hh = self.hh if depth > 1 else {}\n"
         "            for cnt, (_, val, move) in enumerate(sorted(\n"
         "                                    ((v + hh.get((pos.board[m.i], m.j), 0), v, m)\n"
         "                                     for m in pos.gen_moves()\n"
         "                                     if (v:=pos.value(m)) >= val_lower), reverse=True)):\n"),
        ("                    self.tp_move[pos] = move\n",
         "                    k = (pos.board[move.i], move.j)\n"
         "                    self.hh[k] = self.hh.get(k, 0) + depth * depth\n"
         "                    self.tp_move[pos] = move\n"),
        ("        self.t.clear()\n",
         "        self.t.clear()\n"
         "        self.hh.clear()\n"),
    ],
    # TOMBSTONE: `iir`, `iirk`, `noiid`  --  LANDED 2026-08-13.
    #   what:      IIR (reduce a ply with no table move) replacing the IID
    #              probe, killer read once at the top of bound().
    #   measured:  +22.3 +/- 16.0 fixed-node, 1,000 games, raw 415-351.
    #              Entry 3475 -> 3472.
    #   now in:    tools/build/make_pst_entry.py
    #   LIVE ANCHORS -- read this before re-creating any of them:
    #     `noiid`  anchor GONE. Re-creating it raises. Safe.
    #     `iirk`   FIRST ANCHOR STILL LIVE: `depth = max(depth, 0)` occurs
    #              exactly once in the new baseline, so a re-created `iirk`
    #              would insert a SECOND killer read and a SECOND reduction
    #              and the occurs-exactly-once check would PASS. Silent
    #              double. Do not re-create it; compose against the baseline.
    #     `iir`    same hazard, same anchor.
    #   ledger:    "CONFIRMED: iirk.noiid is +22.3 +/- 16.0" (2026-08-13),
    #              which carries the arm sha256 and the packed sha of the
    #              artifact that played the games.
    # THE FRONTIER FUTILITY MARGIN, which is what corrhist turned out to be
    # about once its sign was read correctly.
    #
    # corrhist LOST 54.8 at fixed nodes while searching MORE nodes (1.04x to
    # depth 8, 1.15x to depth 9). Its only consumer is the depth<=1 futility
    # test and its censused table was systematically OPTIMISTIC (mean +10..+18
    # cp), so it made that test fire LESS. It searched more and played worse.
    # The frontier rule is therefore not too aggressive -- if anything it is
    # not aggressive enough -- and the direction with upside is a NEGATIVE
    # margin, which prunes MORE.
    #
    # The constants are reused rather than introduced: the names are already
    # in the lzma stream, which is the entire reason these cost 0-3 bytes.
    # The yielded estimate stays honest at pos.score + val -- the margin is a
    # cushion on the DECISION to stop looking, not a claim about the value.
    #
    # Soundness: a constant inside the break's test leaves it a constant
    # threshold on the sort key `val`, exactly as corrhist's per-node `corr`
    # did. The break still tests the quantity it sorts by.
    #
    # `futm40` is the candidate. `fut40` is its mirror, kept as ONE
    # confirmation arm: if the positive direction reproduces corrhist's loss
    # for 0 bytes, the diagnosis is confirmed and corrhist is closed for good.
    # THE NEGATIVE MARGIN MUST BE IN THE YIELD TOO, and the first version was
    # not. The legality gate caught it on 7 of 100 positions -- three of them
    # QUIET, with a full board of legal replies -- all answering `bestmove
    # (none)`, and no games were spent.
    #
    # The mechanism, exactly. The futility branch tests one quantity and
    # yields another:
    #
    #     if pos.score + val - QS < gamma:      # the TEST
    #         yield (None, pos.score + val)     # the YIELD
    #
    # With a POSITIVE margin the test is stricter than the yield, so a yielded
    # value is always below gamma and the virtual move can never fail high.
    # With a NEGATIVE margin the test is LOOSER, so `pos.score + val` can be
    # >= gamma while the branch fires -- a FAIL HIGH ON A VIRTUAL MOVE. That
    # breaks bound()'s contract that a root fail-high without a move is a
    # verified terminal, and go_loop believes the contract: it prints the
    # score and stops, with nothing in tp_move to play. Hence (none).
    #
    # This is the same lesson as the -449 futility-break bug and the LMP
    # tail-pruning defect, in a third costume: THE TEST AND THE THING IT
    # LICENSES MUST BE THE SAME QUANTITY. Here that means the margin belongs
    # in both places, so the yielded estimate is below gamma by construction.
    #
    # `-y` names the corrected form. The uncorrected ones are kept, unused,
    # as the positive control the legality gate is known to fail.
    "futm40y": [
        ("                if depth <= 1 and pos.score + val < gamma:\n",
         "                if depth <= 1 and pos.score + val - QS < gamma:\n"),
        ("                    yield (move, MATE_UPPER) if val >= MATE_LOWER else (None, pos.score + val)\n",
         "                    yield (move, MATE_UPPER) if val >= MATE_LOWER else (None, pos.score + val - QS)\n"),
    ],
    "futmy": [
        ("                if depth <= 1 and pos.score + val < gamma:\n",
         "                if depth <= 1 and pos.score + val - EVAL_ROUGHNESS < gamma:\n"),
        ("                    yield (move, MATE_UPPER) if val >= MATE_LOWER else (None, pos.score + val)\n",
         "                    yield (move, MATE_UPPER) if val >= MATE_LOWER else (None, pos.score + val - EVAL_ROUGHNESS)\n"),
    ],
    # BROKEN, kept as the gate's positive control -- 7/100 `bestmove (none)`.
    "futm40": ("                if depth <= 1 and pos.score + val < gamma:\n",
               "                if depth <= 1 and pos.score + val - QS < gamma:\n"),
    "futm": ("                if depth <= 1 and pos.score + val < gamma:\n",
             "                if depth <= 1 and pos.score + val - EVAL_ROUGHNESS < gamma:\n"),
    "fut": ("                if depth <= 1 and pos.score + val < gamma:\n",
            "                if depth <= 1 and pos.score + val + EVAL_ROUGHNESS < gamma:\n"),
    "fut40": ("                if depth <= 1 and pos.score + val < gamma:\n",
              "                if depth <= 1 and pos.score + val + QS < gamma:\n"),
    # Widen corrhist's key back to the whole board. Compose as `corr.wkey`:
    # same corrections, same node counts, strictly more string work per
    # interior node -- it exists to price the key, which is the half of
    # corrhist that decides whether the feature is affordable at all.
    "wkey": ("            ckey = pos.board[31:89].translate(PAWNS)\n",
             "            ckey = pos.board.translate(PAWNS)\n"),
    # `kend` and `fresh` USED TO LIVE HERE. They won their round-robin
    # (+107.5 +/- 31.6 vs classic over 4,000 games) and are now part of the
    # baseline, applied by tools/build/make_pst_entry.py -- so they are
    # deliberately NOT mods any more, and removing them was a correctness fix
    # rather than tidying:
    #
    #   - `kend`'s anchor no longer exists in the source, so it would raise.
    #     That is the designed failure and it would have been safe.
    #   - `fresh`'s anchor DOES still exist -- it is the line `kend` used to
    #     produce and the baseline now ships. Re-applying it would have
    #     appended a SECOND from_board() rebuild and the generator's
    #     occurs-exactly-once check would have passed, because the anchor
    #     really does occur exactly once. A silently doubled mod on a variant
    #     that looks correctly generated is precisely the failure this file
    #     was written to prevent.
    #
    # ===================================================================
    # CONVENTION FOR A LANDED MOD: RETIRE IN PLACE.
    #
    # An earlier version of this note said a landed mod must be DELETED.
    # That is half right -- the executable entry must go, or it can be
    # composed onto a baseline that already contains it -- but deleting
    # every trace throws away the one thing that stops the next person
    # re-creating it: the record of WHICH ANCHOR IS STILL LIVE.
    #
    # `iirk.noiid` made the case (landed 2026-08-13, +22.3 +/- 16.0).
    # `noiid`'s anchor was gone afterwards, so re-applying it would have
    # raised -- the designed failure, and safe. But `iirk`'s FIRST anchor,
    # `depth = max(depth, 0)`, still occurs exactly once in the new
    # baseline. Re-applying it would have inserted a second killer read and
    # a second reduction WHILE THE OCCURS-EXACTLY-ONCE CHECK PASSED
    # CLEANLY. A silently doubled mod on a variant that looks correctly
    # generated is the exact failure this file exists to prevent, and no
    # automated check catches it -- only a tombstone a human reads.
    #
    # So, when a mod lands:
    #   1. delete its (anchor, replacement) entry -- it must not stay
    #      composable onto a baseline that already contains it;
    #   2. leave a NAMED tombstone comment where it stood, saying what
    #      landed, what it measured, and -- the load-bearing part --
    #      whether any of its anchors SURVIVE in the new baseline, so the
    #      next reader knows whether re-creating it would raise loudly or
    #      double silently;
    #   3. point at the ledger entry holding the arm sha256 and the packed
    #      sha of the artifact that played.
    #
    # `kend`/`fresh` above and the `iir`/`iirk`/`noiid` tombstone are both
    # written in that form. Provenance also lives in git history and in
    # tools/screens/rr_hole.sh, a finished run's record that names arms
    # this generator can no longer build.
    # ===================================================================
}


def build(mods):
    src = open(SRC).read()
    for mod in mods:
        edits = MODS[mod]
        if callable(edits): edits = edits()          # deferred eval mods
        if isinstance(edits[0], str): edits = [edits]
        for anchor, repl in edits:
            n = src.count(anchor)
            if n != 1:
                raise SystemExit("mod %r: anchor %r occurs %d times, expected 1 -- "
                                 "the source moved under the generator"
                                 % (mod, anchor[:60], n))
            src = src.replace(anchor, repl, 1)
    return src


def main():
    if "--list" in sys.argv:
        print(" ".join(sorted(MODS)))
        return
    outdir, names = sys.argv[1], sys.argv[2:]
    sha = hashlib.sha256(open(SRC, "rb").read()).hexdigest()[:12]
    os.makedirs(outdir, exist_ok=True)
    for name in names:
        mods = [] if name == "base" else name.split(".")
        for mod in mods:
            if mod not in MODS:
                raise SystemExit("unknown mod %r (have: %s)" % (mod, " ".join(sorted(MODS))))
        src = build(mods)
        # Provenance goes after the polyglot header so the header stays line 1.
        i = src.index("\nimport os\n")
        src = (src[:i] + "\n# GENERATED by tools/build/make_variants.py from nnue_4k/pst_entry.py\n"
               "# source sha256[:12] = %s ; mods = %s\n" % (sha, ",".join(mods) or "none") + src[i:])
        out = os.path.join(outdir, "e_%s.py" % name.replace(".", ""))
        open(out, "w").write(src)
        os.chmod(out, 0o755)
        try:
            compile(src, out, "exec")
        except SyntaxError as e:
            raise SystemExit("SYNTAX ERROR in %s line %d: %s" % (out, e.lineno, e.text))
        print("%s  %d bytes  mods=%s" % (out, len(src), ",".join(mods) or "none"))


if __name__ == "__main__":
    main()
