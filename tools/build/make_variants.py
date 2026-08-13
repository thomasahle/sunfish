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
    # `pend`: endgame pawn-advance table. At queens-off, every pawn's value
    # grows quadratically with advancement -- 0,2,8,18,32,50 by rank 2..7
    # (72 on the promotion row, consistently discounting the promotion
    # delta). BOTH colours read it: the mover prices the opponent's runner
    # through the 119-i mirror, which is the taxonomy's pawn-race blindness
    # (41.h3?? c3!) mechanism. The quadratic is search-coupling discipline,
    # not numerology: per-move deltas are 2,6,10,14,18,22 -- all below
    # QS=40 and LMR=60, so the QS admission gate, the futility break and
    # the reduction trigger keep their measured tuning. `x and` keeps the
    # padding zeros zero.
    "pend": [
        ("K_MID, K_END = pst[\"K\"], tuple(piece[\"K\"] + 70\n",
         "P_MID, P_END = pst[\"P\"], tuple(x and x + (8 - i // 10) ** 2 * 2\n"
         "   for i, x in enumerate(pst[\"P\"]))\n"
         "K_MID, K_END = pst[\"K\"], tuple(piece[\"K\"] + 70\n"),
        ('        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END\n',
         '        end = "Q" not in pos.board or "q" not in pos.board\n'
         '        pst["K"] = K_END if end else K_MID\n'
         '        pst["P"] = P_END if end else P_MID\n'),
    ],
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
        "                score = -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3)\n",
        "                score = min(pos.score + EVAL_ROUGHNESS,\n"
        "                    -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3))\n",
    ),
    # Late move reductions off. Threshold-triggered (val < LMR), so setting the
    # threshold to 0 disables it without touching the loop.
    "nolmr": ("\nLMR = 60\n", "\nLMR = 0\n"),
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
        ("        self.tp_score, self.tp_move, self.history = {}, {}, set()\n",
         "        self.tp_score, self.tp_move, self.history = {}, {}, set()\n"
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
        ("        self.tp_score.clear()\n",
         "        self.tp_score.clear()\n"
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
