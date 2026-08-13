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

# Each mod is (anchor, replacement), or a LIST of such pairs for a mod that
# has to touch several places. Every anchor MUST appear exactly once.
MODS = {
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
    # INTERNAL ITERATIVE REDUCTION (ice4 prices it at 37). A node with no
    # table move has never been searched from here, so its ordering is static
    # value alone and searching it at full depth is the most expensive way to
    # discover that. Search it a ply shallower instead.
    #
    # Placed BEFORE the table probe and therefore before the store, so the
    # reduced depth is the key in BOTH directions: the node genuinely becomes
    # a depth-1-shallower node, rather than filing a shallow value under a
    # deep key -- which is the version of IIR that would break
    # one-value-per-key outright instead of merely bending it the way LMR
    # already does.
    #
    # Not a pruning rule: no move is discarded and no tail is cut, so the
    # pseudo-legal-movegen defect that killed LMP cannot reach it and the
    # `best > -MATE_UPPER` preamble does not apply. It is legality-gated
    # anyway -- that gate is cheap and the last three assumptions like this
    # one were wrong.
    "iir": ("        depth = max(depth, 0)\n",
            "        depth = max(depth, 0)\n"
            "\n"
            "        # INTERNAL ITERATIVE REDUCTION. No table move means this node has\n"
            "        # never been searched from here, so its ordering is static value\n"
            "        # alone and full depth is the dearest possible way to find that\n"
            "        # out. Search it a ply shallower. This sits BEFORE the table probe\n"
            "        # and therefore before the store, so the reduced depth is the key\n"
            "        # in both directions -- the node genuinely BECOMES a shallower\n"
            "        # node instead of filing a shallow value under a deep key.\n"
            "        if depth > 2 and pos not in self.tp_move: depth -= 1\n"),
    # Drop the IID probe. IIR is its cheaper answer to the same question -- no
    # table move, so do less work here -- and running both means paying for a
    # whole extra shallow search AND a reduction at the same node. Compose as
    # `iir.noiid`; on its own it prices what IID is currently worth.
    # IIR, single-lookup form. `iir` asks the table for this position twice --
    # once for `pos not in self.tp_move` and again for `killer =
    # self.tp_move.get(pos)` inside moves() -- and the interleaved probe
    # measured that duplicate hash at 7% of nps. This form reads the killer
    # ONCE, at the top, and lets the closure carry it into moves().
    #
    # Compose as `iirk.noiid`. It requires `noiid`, and not by convention: the
    # IID block ASSIGNS to `killer`, which would make the name local to
    # moves() and shadow the outer read into an UnboundLocalError. Removing
    # the IID block is what makes the closure legal, so the two mods travel
    # together and the generator's occurs-exactly-once check enforces it (the
    # assignment line is gone, so `iirk` alone still applies but the engine
    # would be built with IID intact and the closure broken -- hence the
    # explicit ordering note here rather than a silent trap).
    #
    # Behaviourally identical to `iir` by inspection: nothing mutates
    # `self.tp_move` between the top of bound() and the first execution of the
    # generator body. The one real difference is that the lookup is now paid
    # on nodes that return early from the score table, where `iir` paid none
    # -- so it trades one lookup on early-return nodes for one saved on every
    # searched node. NOT YET MEASURED; built now so it is ready if the arm
    # earns it, and it must be re-priced before it is believed.
    "iirk": [
        ("        depth = max(depth, 0)\n",
         "        depth = max(depth, 0)\n"
         "\n"
         "        # The killer is read ONCE here, not again inside moves(): IIR needs\n"
         "        # to know whether this position has a table move, and hashing the\n"
         "        # position twice to ask one question cost 7% of nps.\n"
         "        killer = self.tp_move.get(pos)\n"
         "\n"
         "        # INTERNAL ITERATIVE REDUCTION. No table move means this node has\n"
         "        # never been searched from here, so its ordering is static value\n"
         "        # alone and full depth is the dearest possible way to find that\n"
         "        # out. Search it a ply shallower. This sits BEFORE the table probe\n"
         "        # and therefore before the store, so the reduced depth is the key\n"
         "        # in both directions -- the node genuinely BECOMES a shallower\n"
         "        # node instead of filing a shallow value under a deep key.\n"
         "        if depth > 2 and killer is None: depth -= 1\n"),
        ("            # Look for the strongest move from earlier searches of this position.\n"
         "            # See https://chessprogramming.org/Killer_Move for details.\n"
         "            # We read this \"killer move\" before null-move in case it would get\n"
         "            # evicted from the table or replaced with something else worse.\n"
         "            killer = self.tp_move.get(pos)\n",
         "            # `killer` comes from the enclosing scope, read once at the top of\n"
         "            # bound(). It is still read before null-move, which is the property\n"
         "            # that mattered here: the entry could otherwise be evicted or\n"
         "            # overwritten with something worse while the null search runs.\n"),
    ],
    # The comment goes with the code: leaving eight lines describing an IID
    # probe above a file that no longer has one is the same defect class as
    # the null-move comment that claimed a cap this engine never had.
    "noiid": (
        "            # Back to killer moves: This heuristic is so good, that if there\n"
        "            # is no registered move, it's worth it to run a shallow search to find one.\n"
        "            # See https://chessprogramming.org/Internal_Iterative_Deepening for detais.\n"
        "            # This is known as Internal Iterative Deepening (IID). The probe\n"
        "            # runs as a driver probe (root=True): no null cutoff that would\n"
        "            # end it without storing a move, no repetition truncation, and\n"
        "            # no table entry under deviant semantics.\n"
        "            if not killer and depth > 2:\n"
        "                self.bound(pos, gamma, depth - 3, root=True)\n"
        "                killer = self.tp_move.get(pos)\n",
        "            # NO IID. The probe that used to stand here ran a whole extra\n"
        "            # shallow search whenever there was no table move; `iir` answers\n"
        "            # the same question by reducing this node instead, and the two\n"
        "            # together would pay twice for one observation.\n"),
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
    # A landed mod must therefore be DELETED from here, not left for
    # provenance. Provenance lives in git history and in the ledger entry
    # "THE HOLE ROUND-ROBIN, COMPLETE" (2026-08-13), which records the arm
    # sha256s; tools/screens/rr_hole.sh is that finished run's record and
    # names arms this generator can no longer build.
}


def build(mods):
    src = open(SRC).read()
    for mod in mods:
        edits = MODS[mod]
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
