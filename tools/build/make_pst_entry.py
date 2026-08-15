"""Build the PST 4k entry: our engine + classic's tables, no NNUE.

Not a sum of two measurements -- an actual file, packed by the real
packer, that must play with no other file beside it. `3208 + 579` was
composed, and lzma shares one dictionary across the whole stream, so the
incremental cost of the tables inside THIS source is not the cost of the
same tables inside classic.

Transformations, all mechanical:
  - drop the .sfnn loader, the SWAR constants, nn_cp and _ext
  - paste classic's pst/K_MID/K_END literals in their place
  - drop the accumulator fields (acc, pf, kb) from Position and every
    site that threads them, so score == ps and the eval is the table
usage: build_pst_entry.py OUT.py
"""
import re
import sys

import pathlib

# Derive the repo root from THIS file, never a hard-coded path: the
# absolute one broke CI on every machine but the author's, and inside a
# worktree it silently regenerated from the OTHER checkout -- so
# check_entry.sh verified a file it had never read.
REPO = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO + "/tools/eval4k")
import codec  # noqa: E402

OUT = sys.argv[1]
src = open(REPO + "/nnue_4k/sunfish_nnue.py").read()
classic = open(REPO + "/sunfish.py").read()

# ---- classic's evaluation tables, verbatim -------------------------------
m = re.search(r"\npiece = \{.*?\n(?=\n*###)", classic, re.S)
if not m:
    m = re.search(r"\npst = \{.*?\n(?=\n*###)", classic, re.S)
assert m, "classic tables not found"
tables = m.group(0)
# classic pads its tables at import; keep whatever padding code follows
pad = re.search(r"\npst = \{.*?\n\}", classic, re.S)
assert pad, "pst literal not found"

# ---- store the tables through the startup decoder ------------------------
# Same 384 numbers, bit-identical after decode, 97 fewer bytes in the packed
# artifact (3475 -> 3378) measured by pack.sh on the real file. Not 94: that
# was the saving on the pre-kend/fresh entry, and lzma shares one dictionary
# across the stream, so byte deltas do not compose across landings.
_ns = {}
exec(tables[: tables.index("# Pad tables")], _ns)
tables = codec.emit(_ns["piece"], _ns["pst"])
_chk = {}
exec(tables, _chk)
_ref = {}
exec(m.group(0), _ref)
assert all(tuple(_chk["pst"][k]) == tuple(_ref["pst"][k]) for k in "PNBRQK"), \
    "decoder does not reproduce classic's tables"
assert _chk["K_MID"] == _ref["K_MID"] and _chk["K_END"] == _ref["K_END"]

# ---- excise the NNUE region ---------------------------------------------
i = src.find("import json as _json")
j = src.find("###############################################################################\n"
             "# Piece-Square tables")
assert i > 0 and j > i, "NNUE region not found"
src = src[:i] + tables.strip("\n") + "\n\n" + src[j:]

# the packed engine's own table section now duplicates classic's: drop it
k = src.find("###############################################################################\n"
             "# Piece-Square tables")
k2 = src.find("###############################################################################", k + 10)
assert k2 > k
src = src[:k] + src[k2:]

# ---- strip the accumulator from Position --------------------------------
src = src.replace(
    'class Position(namedtuple("Position", "board score ps wc bc ep kp acc pf kb")):',
    'class Position(namedtuple("Position", "board score wc bc ep kp")):', 1)

src = src.replace("""        return Position(
            self.board[::-1].swapcase(), -self.score, -self.ps, self.bc, self.wc,
            119 - self.ep if self.ep and not nullmove else 0,
            119 - self.kp if self.kp and not nullmove else 0,
            self.acc, self.pf ^ 1, self.kb,
        )""",
"""        return Position(
            self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
            119 - self.ep if self.ep and not nullmove else 0,
            119 - self.kp if self.kp and not nullmove else 0,
        )""", 1)

src = src.replace("""        ps = self.ps + self.value(move)""",
                  """        score = self.score + self.value(move)""", 1)
src = re.sub(r"\n *# Every board mutation below.*?acc -= row\[q\]\[j\]\n", "\n", src, flags=re.S)
src = re.sub(r"\n *if B > 1:\n *# Mover's own bucket.*?kb = nb \* B \+ kb % B if self\.pf == 0 else kb - ob \+ nb\n",
             "\n", src, flags=re.S)
src = re.sub(r"\n *acc \+= row\[\"R\"\]\[kp\] - row\[\"R\"\]\[r\]", "", src)
src = re.sub(r"\n *acc \+= row\[prom\]\[j\] - row\[\"P\"\]\[j\]", "", src)
src = re.sub(r"\n *acc -= row\[\"p\"\]\[j \+ S\]", "", src)
src = re.sub(r"\n *# A king move across a bucket boundary.*?acc \+= row\[c\]\[s\]\n", "\n", src, flags=re.S)
src = src.replace("""        # We rotate the returned position, so it's ready for the next player
        pf = self.pf ^ 1
        return Position(board[::-1].swapcase(), -ps + nn_cp(acc, pf, board), -ps,
                        bc, wc, 119 - ep if ep else 0, 119 - kp if kp else 0,
                        acc, pf, kb)""",
"""        # We rotate the returned position, so it's ready for the next player
        return Position(board[::-1].swapcase(), -score, bc, wc,
                        119 - ep if ep else 0, 119 - kp if kp else 0)""", 1)

src = src.replace('''def from_board(board, wc=(True, True), bc=(True, True), ep=0, kp=0, pf=0):
    """Build a position (and its accumulator) from scratch. `board` is
    already in the side-to-move's orientation."""
    ps = sum(pst[p][i] if p.isupper() else -pst[p.upper()][119 - i]
             for i, p in enumerate(board) if p.isalpha())
    kb = 0
    if B > 1:
        own, opp = kbucket(board.index("K")), kbucket(119 - board.index("k"))
        kb = own * B + opp if pf == 0 else opp * B + own
    acc = ACC_BASE
    row = ROWS[pf][kb]
    for i, p in enumerate(board):
        if p in _PIECES:
            acc += row[p][i]
    return Position(board, ps + nn_cp(acc, pf, board), ps, wc, bc, ep, kp,
                    acc, pf, kb)''',
'''def from_board(board, wc=(True, True), bc=(True, True), ep=0, kp=0, pf=0):
    """Build a position from scratch; `board` is in the mover's orientation."""
    score = sum(pst[p][i] if p.isupper() else -pst[p.upper()][119 - i]
                for i, p in enumerate(board) if p.isalpha())
    return Position(board, score, wc, bc, ep, kp)''', 1)

# ps was the packed engine's clip-free score; with no net they are the same
src = src.replace("pos.ps", "pos.score").replace("self.ps", "self.score")
# ---- classic's tables need classic's king-table phase rule ---------------
# The packed engine switches to K_END on a bare king. That is right for ITS
# K_END -- a trained bare-king mop-up table -- but this entry pastes CLASSIC's
# K_END, the centralization gradient classic calls out as "important to win
# KRK/KQK endings", and classic keys it on queens-off. Over 37,374 positions
# from real games the two rules disagree on 62.1%. Measured +52.3 +/- 21.1
# fixed-node, +30.5 +/- 24.4 timed head-to-head, and it SAVES 11 packed bytes.
#
# Swapping the table also invalidates the incrementally carried `pos.score`,
# which was accumulated under the old table and which rotate() then sign-flips
# every ply -- an oscillating phantom tempo on every stand-pat and futility
# margin for the whole search. Rebuild the root under the table this search
# will actually use (+3 bytes; net -8, entry 3483 -> 3475).
old = ('        # Bare-king endings: swap in the centralization gradient (packed\'s\n'
       '        # own measured condition; classic keys on queens-off instead).\n'
       '        # Both directions every search because a game reaches bare-king\n'
       '        # DURING play with one Searcher at the wheel (a new game gets a\n'
       '        # new Searcher via ucinewgame; this was never about state outliving\n'
       '        # a game). The history scores this search inherits were accumulated\n'
       '        # under the previous table -- off by a constant after a swap, which\n'
       '        # classic measured harmless on the C twin at fixed nodes (668 games,\n'
       '        # +0.52 +/- 6.37; nnue_4k/MEASUREMENTS.md 2026-08-14, PR #184). The\n'
       '        # 4k entry re-derives instead (pst_entry.py\'s kend+fresh pair).\n'
       '        bare = sum(c.isupper() for c in pos.board) == 1 or sum(c.islower() for c in pos.board) == 1\n'
       '        pst["K"] = K_END if bare else K_MID\n')
new = ('        # Classic\'s K_END is a centralization gradient, and classic keys it\n'
       '        # on queens-off. Both directions every search because the queens\n'
       '        # leave DURING a game and one Searcher plays every move of it (a\n'
       '        # new game gets a new Searcher; this was never about state\n'
       '        # outliving a game).\n'
       '        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END\n'
       '        # The carried score was accumulated under the OTHER table: re-derive\n'
       '        # it fresh so the swap leaves no stale constant behind (the entry\'s\n'
       '        # kend+fresh pair; classic measured the constant harmless on the C\n'
       '        # twin, +0.52 +/- 6.37 over 668 fixed-node games -- see\n'
       '        # nnue_4k/MEASUREMENTS.md 2026-08-14, PR #184).\n'
       '        pos = self.root = from_board(pos.board, pos.wc, pos.bc, pos.ep, pos.kp)\n')
assert src.count(old) == 1, "king-table block not found verbatim in sunfish_nnue.py"
src = src.replace(old, new, 1)

# ---- IIR replaces IID ----------------------------------------------------
# Measured on THIS entry: +22.3 +/- 16.0 fixed-node over 1,000 games against
# the pre-IIR entry (raw 415-351, zero forfeits, zero illegal), and the entry
# gets SMALLER -- 3475 -> 3472. Stronger and smaller, the first time this lane
# has had both from one change.
#
# ENTRY-ONLY, deliberately. The trigger reads no evaluation, so the
# (feature, eval) rule does not force a re-measure -- but sunfish_nnue.py is
# the lichess bot's engine and another lane's artifact, and nothing here has
# played a game with the net. It goes in the entry's transform list, where
# kend/fresh live, and transfers to the NNUE engine only if someone measures
# it there.
#
# Three edits, each anchored to text that must occur exactly once.
_iir_edits = [
    # 1. Read the killer ONCE, at the top, and reduce a ply when there is no
    #    table move. Placed BEFORE the score-table probe and therefore before
    #    the store, so the reduced depth is the key in both directions: the
    #    node genuinely becomes a shallower node rather than filing a shallow
    #    value under a deep key. Reading the killer here instead of inside
    #    moves() is what keeps it to ONE hash of the position -- asking the
    #    same dict twice cost 7% of nps (0.908x vs 0.950x) for no behaviour.
    ("        depth = max(depth, 0)\n",
     "        depth = max(depth, 0)\n"
     "\n"
     "        # The killer is read ONCE here, not again inside moves(): the reduction\n"
     "        # below needs to know whether this position has a table move, and\n"
     "        # hashing the position twice to ask one question cost 7% of nps.\n"
     "        killer = self.tp_move.get(pos)\n"
     "\n"
     "        # INTERNAL ITERATIVE REDUCTION. No table move means this node has never\n"
     "        # been searched from here, so its ordering is static value alone and\n"
     "        # full depth is the dearest possible way to discover that. Search it a\n"
     "        # ply shallower instead. This REPLACED the IID probe, which answered\n"
     "        # the same question by running a whole extra shallow search; keeping\n"
     "        # both would pay twice for one observation.\n"
     "        if depth > 2 and killer is None: depth -= 1\n"),
    # 2. moves() takes the killer from the enclosing scope. This is only legal
    #    because edit 3 removes the IID block: that block ASSIGNED to `killer`,
    #    which would make the name local to moves() and turn the outer read
    #    into an UnboundLocalError. The two edits are one change.
    ('            # Look for the strongest move from earlier searches of this position.\n'
     '            # See https://chessprogramming.org/Killer_Move for details.\n'
     '            # We read this "killer move" before null-move in case it would get\n'
     '            # evicted from the table or replaced with something else worse.\n'
     '            killer = self.tp_move.get(pos)\n',
     "            # `killer` comes from the enclosing scope, read once at the top of\n"
     "            # bound(). It is still read before null-move, which is the property\n"
     "            # that mattered: the entry could otherwise be evicted or overwritten\n"
     "            # with something worse while the null search runs.\n"),
    # 3. The IID probe goes, and its comment goes with it. A comment describing
    #    a probe the file no longer has is the same defect as the null-move
    #    comment that claimed a cap this engine never had.
    ("            # Back to killer moves: This heuristic is so good, that if there\n"
     "            # is no registered move, it's worth it to run a shallow search to find one.\n"
     "            # See https://chessprogramming.org/Internal_Iterative_Deepening for detais.\n"
     "            # This is known as Internal Iterative Deepening (IID). The probe\n"
     "            # runs as a driver probe (root=True): no null cutoff that would\n"
     "            # end it without storing a move, no repetition truncation, and\n"
     "            # no table entry under deviant semantics.\n"
     "            if not killer and depth > 2:\n"
     "                self.bound(pos, gamma, depth - 3, root=True)\n"
     "                killer = self.tp_move.get(pos)\n",
     ""),
]
# ---- the entry must not describe a net it does not contain ----------------
# Found by the golf survey, and it is the Position-docstring sibling of the
# null-move comment that claimed a cap this engine never had: the shipped
# entry opened with a section header called "Packed big-integer NNUE residual"
# and a Position docstring listing `ps`, `acc`, `pf` and `kb` -- four fields
# this class does not have. Comments are stripped by the packer so none of
# this costs a byte, but the standing rule is that the model matches the code,
# and a reader of the artifact's source was being told it evaluates with a net.
#
# The dead `pf=0` parameter on from_board() is the same defect in executable
# form: it is the NNUE perspective flag, no caller in this entry or in
# sunfish_ui/uci.py passes it, and unlike the comments it is not free.
_truth_edits = [
    ("###############################################################################\n"
     "# Packed big-integer NNUE residual\n"
     "###############################################################################\n"
     "# The evaluation is\n"
     "#     score = pst(pos)  +  clip(nn(pos), -CLAMP, CLAMP)\n"
     "# where pst() is classic sunfish's exact incremental piece-square score (so\n"
     "# `value(move)` stays exact for move ordering, the QS gate and futility) and\n"
     "# nn() is a 768 -> N -> 1 net whose whole accumulator and whole head live in\n"
     "# ONE Python int.  See packed/pnet.py for the lane layout and why the head\n"
     "# needs no per-lane multiply.\n",
     "###############################################################################\n"
     "# Evaluation: classic sunfish's piece-square tables, and nothing else\n"
     "###############################################################################\n"
     "# THERE IS NO NET HERE. This file is generated from the packed-NNUE engine by\n"
     "# tools/build/make_pst_entry.py, which excises the loader, the accumulator and\n"
     "# the residual, and pastes classic's tables in their place. The evaluation is\n"
     "#     score = pst(pos)\n"
     "# kept exactly incremental, so `value(move)` stays an exact delta of it for\n"
     "# move ordering, the QS admission gate and the futility test.\n"),
    ("    score -- the board evaluation: ps + the clipped net residual\n"
     "    ps -- the piece-square part of the score alone, kept exactly incremental\n"
     "          so that value(move) below stays an exact delta of it\n",
     "    score -- the piece-square evaluation, kept exactly incremental so that\n"
     "             value(move) below stays an exact delta of it\n"),
    ("    ep - the en passant square\n"
     "    kp - the king passant square\n"
     "    acc -- the packed NNUE accumulator (one big int, 2N + 2*nb lanes)\n"
     "    pf -- perspective flag: which of the two lane blocks is the mover's\n"
     "    kb -- combined king-bucket index B*bucket(white) + bucket(black), in\n"
     "          ABSOLUTE colours (0 for plain B == 1 nets)\n"
     "\n"
     "    score/ps/acc/pf/kb are all functions of the other fields, so\n"
     "    identity -- what the transposition table, the killer table and the\n"
     "    repetition set key on -- deliberately ignores them.  Keeping the\n"
     "    accumulator out of __hash__ also keeps hashing off the big int, which\n"
     "    would otherwise cost more than the evaluation it feeds.\n",
     "    ep - the en passant square\n"
     "    kp - the king passant square\n"
     "\n"
     "    `score` is a function of the other fields, so identity -- what the\n"
     "    transposition table, the killer table and the repetition set key on --\n"
     "    deliberately ignores it. That is LOAD-BEARING, not tidiness: pst[\"K\"]\n"
     "    is swapped between K_MID and K_END per search, so the same board can\n"
     "    carry two different scores across a table change, and a repetition set\n"
     "    or a killer table that compared scores would stop recognising it.\n"),
    ("# MATE values derive from the classic piece values (K=60000, Q=929);\n"
     "# the tables themselves ride in the net file (see the loader above).\n",
     "# MATE values derive from the classic piece values (K=60000, Q=929).\n"),
    # The dead NNUE perspective flag. No caller passes it -- not in this file,
    # not in sunfish_ui/uci.py, which calls from_board(board, wc, bc, ep, 0).
    ("def from_board(board, wc=(True, True), bc=(True, True), ep=0, kp=0, pf=0):\n",
     "def from_board(board, wc=(True, True), bc=(True, True), ep=0, kp=0):\n"),
]
for _anchor, _repl in _truth_edits:
    assert src.count(_anchor) == 1, (
        "truth-edit anchor occurs %d times, expected 1: %r"
        % (src.count(_anchor), _anchor[:60]))
    src = src.replace(_anchor, _repl, 1)

# ---- golf: the artifact's info/PV line ------------------------------------
# -22 packed bytes, and the only cut on the golf menu that costs no capability
# anything in this repo uses. `info` is optional in UCI, `cand` is still
# computed (it is what gets played), and the DEVELOPMENT path is untouched:
# every screen runs through sunfish_ui/uci.py, which prints full info lines
# with depth, time, nodes, nps and pv. What is lost is PV output from the
# PACKED artifact in production, which no gate, no test and no harness reads.
#
# Verified after the cut rather than assumed: the artifact still streams its
# handshake and its bestmove LIVE to a pipe with stdin held open. That check
# matters more once these lines are gone, because they were the only output
# between `go` and `bestmove` -- and the first version of the check was itself
# wrong (select() on a buffered text stream reported a phantom stall), so it
# was redone with a reader thread before anything was believed.
_info_anchor = (
    '                        cand = render(i) + render(j) + move.prom.lower()\n'
    '                        print("info depth", depth, "score cp", score, "pv", cand)\n')
assert src.count(_info_anchor) == 1, "info-print anchor not found verbatim"
src = src.replace(
    _info_anchor,
    '                        cand = render(i) + render(j) + move.prom.lower()\n', 1)

for _anchor, _repl in _iir_edits:
    assert src.count(_anchor) == 1, (
        "IIR anchor occurs %d times, expected 1 -- sunfish_nnue.py moved under "
        "this generator: %r" % (src.count(_anchor), _anchor[:60]))
    src = src.replace(_anchor, _repl, 1)

# ---- golf: rename what nothing outside this file dereferences -------------
# pyminify renames globals and locals, but ATTRIBUTE and METHOD names survive
# minification verbatim -- they are the only long identifiers left in the
# packed stream. They cannot ALL go: the dev driver (sunfish_ui/uci.py)
# reads, by name, pos.board/.score/.kp/.move()/.gen_moves()/.rotate()/
# .value()/.prom, searcher.bound()/.search()/.tp_move/.nodes/.deadline/
# .node_cap, the `root=` kwarg of bound() (uci.py:216), and the module
# globals in ENGINE_API -- and agree.py and the variant screens drive entry
# SOURCES through that driver, so renaming any of those breaks the lane's
# own instruments. What is renamed here is exactly the set no external
# caller touches. The ledger's -103 estimate assumed the full rename; the
# driver-visible names are why the measured saving is smaller.
_golf_renames = [
    # (regex, replacement, expected count incl. comments -- drift is loud)
    # __version__ is a dunder, so pyminify keeps it verbatim; the entry's only
    # reader is the `version` concatenation right below it (ENGINE_API needs
    # `version`, nothing needs `__version__`). Fold the indirection.
    (r'__version__ = "([^"]+)"\nversion = "sunfish " \+ __version__',
     r'version = "sunfish \1"', 1),
    # 6 = 4 historical + the bestmove-floor call and its comment mention
    (r"\bking_capture\b", "k", 6),
    (r"\btp_score\b", "t", 9),
    (r"self\.history\b", "self.h", 4),
    (r"self\.root\b", "self.r", 3),
    (r"\bnullmove\b", "n", 7),
    (r"entry\.lower\b", "entry.l", 3),
    (r"entry\.upper\b", "entry.u", 3),
    (r'"lower upper"', '"l u"', 1),
    # namedtuple TYPENAME strings only feed repr(); the module globals the
    # driver's ENGINE_API checks (Move, Position) are untouched.
    (r'namedtuple\("Move"', 'namedtuple("M"', 1),
    (r'namedtuple\("Position"', 'namedtuple("P"', 1),
    (r'namedtuple\("Entry"', 'namedtuple("E"', 1),
]
# ---- P_END: the pawn table's queens-off variant (LANDED 2026-08-15) -------
# Screened +36.71 and CONFIRMED +21.31 [+5.58, +37.04] over a fixed 800 games.
# The queens-off seam already switches the KING table (K_MID/K_END); this adds
# the pawn one at the SAME test, so it costs one tuple and no new branch: with
# queens off, a pawn is worth (8 - rank)^2 * 2 more, steeply rewarding advanced
# passers exactly when promotion is the winning plan.
#
# Applied HERE rather than in sunfish.py because it is a 4k-entry change, not a
# classic one: `tables` is codec.emit's output and `src` is the engine body, so
# both anchors are asserted to occur exactly once and a drift in either is a
# hard build error rather than a silently unmodified entry.
_pend = [
    ('K_MID, K_END = pst["K"], tuple(piece["K"] + 70\n',
     'P_MID, P_END = pst["P"], tuple(x and x + (8 - i // 10) ** 2 * 2\n'
     '   for i, x in enumerate(pst["P"]))\n'
     'K_MID, K_END = pst["K"], tuple(piece["K"] + 70\n'),
    ('        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END\n',
     '        end = "Q" not in pos.board or "q" not in pos.board\n'
     '        pst["K"] = K_END if end else K_MID\n'
     '        pst["P"] = P_END if end else P_MID\n'),
]
for _a, _b in _pend:
    assert src.count(_a) == 1, "pend anchor %r occurs %d times" % (_a[:40], src.count(_a))
    src = src.replace(_a, _b, 1)

for _pat, _repl, _n in _golf_renames:
    src, _c = re.subn(_pat, _repl, src)
    assert _c == _n, "golf rename %r matched %d times, expected %d" % (_pat, _c, _n)

open(OUT, "w").write(src)
try:
    compile(src, OUT, "exec")
    print("built %s (%d source bytes) -- compiles" % (OUT, len(src)))
except SyntaxError as e:
    print("SYNTAX ERROR line %d: %s" % (e.lineno, src.split("\n")[e.lineno - 1][:80]))
    sys.exit(1)
