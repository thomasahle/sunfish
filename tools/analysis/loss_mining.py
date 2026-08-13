#!/usr/bin/env python3
"""Loss taxonomy miner. Implements tools/analysis/loss_rules.md EXACTLY.

    nice -n 15 python3 tools/analysis/loss_mining.py --pgn-dir DIR

PGN parsing only (python-chess is a move parser here, not an engine). The
hand-labeled positive controls from the rules file are asserted first; any
mismatch aborts before a single taxonomy number is printed. Output is a
deterministic markdown report on stdout.

DIR layout (snapshot copies, see the corpus ledger in nnue_4k/LOSS_TAXONOMY.md):
    pyleague_games.pgn
    eval-c1-20260813/{c1screen,c2screen,d1screen}.pgn
    elo-noiid-20260813/match.pgn
"""
import argparse
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field

import chess
import chess.pgn

SWING_CP = -150        # candidate swing: own-eval drop of at least 150 cp
HYSTERESIS_CP = 50     # voided if eval later returns within 50 cp of pre-drop
CREEP_ANCHOR_CP = -100  # creeping anchor: first own eval <= -100 never recovering
MATE_MAG_CP = 30_000   # |cp| at or above this is a mate-magnitude score
DEPTH_WINDOW = 5       # own moves strictly before the swing/anchor
DEPTH_BAND = 1.0       # LOW <= -1.0, HIGH >= +1.0 vs game median
OPTIMISM_WINDOW = 6    # own moves strictly before the swing/anchor
MIN_SCORED = 4         # fewer scored own moves => UNSCORED

NPM = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
COMMENT_RE = re.compile(r"([+-]?M?\d+(?:\.\d+)?)/(\d+)\s")

CORPORA = [
    # (corpus key, relative path, tracked engine names)
    ("pyleague", "pyleague_games.pgn", ("sunfish4k", "classic")),
    ("c1screen", "eval-c1-20260813/c1screen.pgn", ("base",)),
    ("c2screen", "eval-c1-20260813/c2screen.pgn", ("base",)),
    ("d1screen", "eval-c1-20260813/d1screen.pgn", ("base", "d1")),
    ("elo-noiid", "elo-noiid-20260813/match.pgn", ("master-697d69a",)),
]

# (corpus, game index, White, Black, Result, loser) ->
#   (class, swing/anchor fullmove, phase, depth signal, termination)
CONTROLS = {
    ("pyleague", 1, "d-house", "sunfish4k", "1-0", "sunfish4k"):
        ("SELF-DETECTED", 17, "MIDDLEGAME", "NORMAL", "MATE"),
    ("pyleague", 2, "sunfish4k", "d-house", "0-1", "sunfish4k"):
        ("SELF-DETECTED", 39, "ENDGAME", "HIGH", "ADJUD"),
    ("pyleague", 4, "sunfish4k", "classic", "0-1", "sunfish4k"):
        ("SELF-DETECTED", 42, "ENDGAME", "NORMAL", "ADJUD"),
    ("elo-noiid", 2, "master-697d69a", "noiid-53b35eb", "0-1", "master-697d69a"):
        ("CREEPING", 18, "MIDDLEGAME", "NORMAL", "MATE"),
    ("d1screen", 4, "base", "d1", "1-0", "d1"):
        ("SELF-DETECTED", 29, "MIDDLEGAME", "NORMAL", "ADJUD"),
}


@dataclass
class OwnMove:
    fullmove: int
    cp: int            # own-perspective centipawns (mates folded to +/-50000)
    mate: bool
    depth: int
    npm_before: int    # both-sides non-pawn material before this move
    fen_before: str    # position in which this move was chosen
    san: str           # the move itself
    opp_cp: int | None  # opponent's next recorded eval (opp perspective)
    opp_mate: bool


@dataclass
class Loss:
    corpus: str
    index: int
    loser: str
    opponent: str
    color: str                 # W/B of the loser
    swing_class: str           # SELF-DETECTED / CREEPING / UNSCORED
    swing_move: int | None
    phase: str | None
    depth_signal: str | None
    termination: str
    npm_at_swing: int | None = None
    optimism: float | None = None
    plycount: int = 0
    swing_fen: str | None = None   # position before the swing/anchor move
    swing_san: str | None = None
    pre_cp: int | None = None      # own eval before/at the swing
    post_cp: int | None = None
    extras: dict = field(default_factory=dict)


def parse_cp(token):
    """'+0.34' -> 34 cp; '+M5'/'-M3' -> +/-50000. Returns (cp, is_mate)."""
    if "M" in token:
        return (50_000 if not token.startswith("-") else -50_000), True
    cp = round(float(token) * 100)
    return cp, abs(cp) >= MATE_MAG_CP


def npm_of(board):
    return sum(v * (len(board.pieces(p, chess.WHITE)) + len(board.pieces(p, chess.BLACK)))
               for p, v in NPM.items())


def termination_class(headers, last_comment):
    lc = last_comment.lower()
    if "mates" in lc:
        return "MATE"
    term = headers.get("Termination", "")
    if term == "time forfeit" or "on time" in lc:
        return "TIME"
    if term == "abandoned":
        return "ABANDONED"
    if term == "adjudication":
        return "ADJUD"
    return "OTHER"


def extract_own_moves(game, loser_color):
    """One OwnMove per scored own move, plus the final comment."""
    own, last_comment = [], ""
    pending = None  # last own move awaiting the opponent's reply eval
    board = game.board()
    for node in game.mainline():
        pre = board
        comment = (node.comment or "").strip()
        if comment:
            last_comment = comment
        m = COMMENT_RE.match(comment)
        parsed = None
        if m:
            cp, mate = parse_cp(m.group(1))
            parsed = (cp, mate, int(m.group(2)))
        if pre.turn == loser_color:
            if parsed:
                cp, mate, depth = parsed
                pending = OwnMove(pre.fullmove_number, cp, mate, depth,
                                  npm_of(pre), pre.fen(), pre.san(node.move),
                                  None, False)
                own.append(pending)
            else:
                pending = None
        elif pending is not None and parsed:
            pending.opp_cp, pending.opp_mate = parsed[0], parsed[1]
            pending = None
        board = pre.copy(stack=False)
        board.push(node.move)
    return own, last_comment


def find_swing(own):
    """Rules-file swing: first -150cp drop surviving hysteresis + mate rule.

    Returns (kind, index into own) where kind is SELF-DETECTED or CREEPING.
    """
    for i in range(1, len(own)):
        drop = own[i].cp - own[i - 1].cp
        if drop > SWING_CP:
            continue
        if any(own[j].cp > own[i - 1].cp - HYSTERESIS_CP for j in range(i + 1, len(own))):
            continue  # recovered within hysteresis: voided
        if own[i].mate and own[i - 1].cp <= CREEP_ANCHOR_CP:
            break     # mate announcement from an already-lost position
        return "SELF-DETECTED", i
    # creeping anchor: first own eval <= -100 with no later recovery above -100
    for i, om in enumerate(own):
        if om.cp <= CREEP_ANCHOR_CP and all(o.cp <= -100 for o in own[i:]):
            return "CREEPING", i
    return "CREEPING", len(own) - 1


def depth_signal(own, idx):
    depths = [o.depth for o in own]
    window = depths[max(0, idx - DEPTH_WINDOW):idx]
    if len(window) < 2:
        return "UNKNOWN"
    delta = statistics.mean(window) - statistics.median(depths)
    if delta <= -DEPTH_BAND:
        return "LOW"
    if delta >= DEPTH_BAND:
        return "HIGH"
    return "NORMAL"


def optimism(own, idx):
    pairs = [o.cp + o.opp_cp for o in own[max(0, idx - OPTIMISM_WINDOW):idx]
             if o.opp_cp is not None and not o.mate and not o.opp_mate]
    return round(statistics.mean(pairs), 1) if len(pairs) >= 2 else None


def phase_of(om):
    if om.npm_before <= 12:
        return "ENDGAME"
    if om.fullmove <= 14 and om.npm_before >= 24:
        return "OPENING"
    return "MIDDLEGAME"


def classify_loss(corpus, index, game, loser, opponent, loser_color):
    own, last_comment = extract_own_moves(game, loser_color)
    term = termination_class(game.headers, last_comment)
    loss = Loss(corpus, index, loser, opponent, "W" if loser_color else "B",
                "UNSCORED", None, None, None, term,
                plycount=int(game.headers.get("PlyCount", 0)))
    if len(own) < MIN_SCORED:
        return loss
    kind, idx = find_swing(own)
    om = own[idx]
    loss.swing_class = kind
    loss.swing_move = om.fullmove
    loss.phase = phase_of(om)
    loss.depth_signal = depth_signal(own, idx)
    loss.npm_at_swing = om.npm_before
    loss.optimism = optimism(own, idx)
    loss.swing_fen, loss.swing_san = om.fen_before, om.san
    loss.pre_cp = own[idx - 1].cp if idx else None
    loss.post_cp = om.cp
    return loss


def mine(pgn_dir):
    losses, ledger = [], []
    for corpus, rel, tracked in CORPORA:
        path = f"{pgn_dir}/{rel}"
        per_name = {name: Counter() for name in tracked}
        skipped = 0
        with open(path, encoding="utf-8", errors="replace") as f:
            index = 0
            while (game := chess.pgn.read_game(f)) is not None:
                index += 1
                h = game.headers
                res = h.get("Result", "*")
                if res not in ("1-0", "0-1"):
                    skipped += 1
                    continue
                for name in tracked:
                    if h.get("White") == name:
                        color, opp = chess.WHITE, h.get("Black")
                    elif h.get("Black") == name:
                        color, opp = chess.BLACK, h.get("White")
                    else:
                        continue
                    lost = (res == "0-1") if color == chess.WHITE else (res == "1-0")
                    per_name[name]["losses" if lost else "wins"] += 1
                    if lost:
                        losses.append(classify_loss(corpus, index, game, name, opp, color))
            ledger.append((corpus, rel, index, skipped, per_name))
    return losses, ledger


def check_controls(losses):
    by_key = {}
    for L in losses:
        w, b = (L.loser, L.opponent) if L.color == "W" else (L.opponent, L.loser)
        res = "0-1" if L.color == "W" else "1-0"
        by_key[(L.corpus, L.index, w, b, res, L.loser)] = L
    failures = []
    for key, exp in CONTROLS.items():
        L = by_key.get(key)
        got = (None if L is None else
               (L.swing_class, L.swing_move, L.phase, L.depth_signal, L.termination))
        if got != exp:
            failures.append(f"CONTROL FAIL {key}: expected {exp}, got {got}")
    return failures


def table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def taxonomy_rows(group):
    rows = []
    for phase in ("OPENING", "MIDDLEGAME", "ENDGAME"):
        for cls in ("SELF-DETECTED", "CREEPING"):
            sel = [L for L in group if L.phase == phase and L.swing_class == cls]
            if not sel:
                continue
            c = Counter(L.depth_signal for L in sel)
            opt = [L.optimism for L in sel if L.optimism is not None]
            rows.append((phase, cls, len(sel), c["LOW"], c["NORMAL"], c["HIGH"],
                         c["UNKNOWN"],
                         round(statistics.mean(opt), 1) if opt else "-",
                         round(statistics.median(L.swing_move for L in sel), 1)))
    uns = [L for L in group if L.swing_class == "UNSCORED"]
    if uns:
        rows.append(("-", "UNSCORED", len(uns), "-", "-", "-", "-", "-", "-"))
    return rows


HEADER = ("phase", "class", "n", "d:LOW", "d:NORM", "d:HIGH", "d:?", "optimism", "med.move")


def report(losses, ledger):
    print("# Loss-mining output (deterministic; rules: tools/analysis/loss_rules.md)\n")
    print("## Corpus ledger\n")
    rows = []
    for c, rel, games, skipped, per_name in ledger:
        for name, cnt in per_name.items():
            rows.append((c, rel, games, skipped, name, cnt["wins"], cnt["losses"]))
    print(table(rows, ("corpus", "file", "games", "undecided", "tracked",
                       "decisive-wins", "decisive-losses")))
    print()
    for corpus, _, tracked in CORPORA:
        for name in tracked:
            group = sorted((L for L in losses if L.corpus == corpus and L.loser == name),
                           key=lambda L: L.index)
            if not group:
                continue
            print(f"\n## {corpus}: losses by {name} (n={len(group)})\n")
            print(table(taxonomy_rows(group), HEADER))
            term = Counter(L.termination for L in group)
            print("\nterminations: " + ", ".join(f"{k}={v}" for k, v in sorted(term.items())))
            if corpus == "pyleague":
                print("\nby opponent:")
                opps = sorted({L.opponent for L in group})
                rows = []
                for opp in opps:
                    sel = [L for L in group if L.opponent == opp]
                    c = Counter(L.swing_class for L in sel)
                    p = Counter(L.phase for L in sel)
                    o = [L.optimism for L in sel if L.optimism is not None]
                    rows.append((opp, len(sel), c["SELF-DETECTED"], c["CREEPING"],
                                 p["OPENING"], p["MIDDLEGAME"], p["ENDGAME"],
                                 round(statistics.mean(o), 1) if o else "-"))
                print(table(rows, ("opponent", "losses", "self-det", "creep",
                                   "open", "mid", "end", "optimism")))


def dump_swings(losses, path):
    """EPD of every swing/anchor position: input for the box-slot re-analysis probe."""
    with open(path, "w") as f:
        for L in sorted(losses, key=lambda L: (L.corpus, L.loser, L.index)):
            if L.swing_fen is None:
                continue
            f.write(f"{L.swing_fen} c0 \"{L.corpus}#{L.index} {L.loser} {L.swing_class}"
                    f" {L.phase} {L.depth_signal} played={L.swing_san}"
                    f" pre={L.pre_cp} post={L.post_cp}\";\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn-dir", required=True)
    ap.add_argument("--dump-swings", metavar="EPD",
                    help="write swing/anchor positions (probe input) to this EPD file")
    args = ap.parse_args()
    losses, ledger = mine(args.pgn_dir)
    failures = check_controls(losses)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        print("Controls failed; refusing to emit taxonomy.", file=sys.stderr)
        return 1
    print(f"controls: all {len(CONTROLS)} hand-labeled controls reproduced", file=sys.stderr)
    if args.dump_swings:
        dump_swings(losses, args.dump_swings)
    report(losses, ledger)
    return 0


if __name__ == "__main__":
    sys.exit(main())
