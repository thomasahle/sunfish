/-
Discharging the `Bounded` hypothesis for sunfish's CONCRETE evaluation.

`Bounded G` (Sunfish/Tricks.lean) has been a named hypothesis: every
static eval lies in [-MATE_UPPER, MATE_UPPER]. This file proves the
concrete numeric fact behind it, from the actual piece values and
piece-square tables of sunfish.py (transcribed below, with the padding
fold's piece values already added, exactly as the code builds them).

The honest statement needs two side conditions, one hiding a subtlety:

* EXACTLY ONE KING PER SIDE. In the king-capture model a transiently
  kingless position can score ~ K + 15 pieces ~ 70500 > MATE_UPPER =
  69290 -- the naive unconditional claim is FALSE. It is sound in use
  because bound() short-circuits king-gone positions (pos.score <=
  -MATE_LOWER returns -MATE_UPPER) before any stand-pat consults the
  eval; only both-kings positions reach it.
* AT MOST 15 NON-KING PIECES PER SIDE: true of every reachable game --
  each side starts with 15, captures only remove, promotion preserves
  the count.

Under those conditions the mover-view score
  (kEntry_own + sum of own non-king square values)
- (kEntry_opp + sum of opp non-king square values)
is bounded by (kMax - kMin) + 15 * nkMax, because every non-king
square value is nonnegative (nk_nonneg below), so the opponent's sum
is at least kMin. The headline theorem shows the bound sits BELOW
MATE_LOWER -- strictly stronger than `Bounded` needs: no static eval
can even touch the mate band, so evals never collide with mate or
king-capture sentinels.

The link from "sunfish board string" to "piece multiset" is not
modeled here (the hand model has no boards); closing that last gap
against the real code is the lean-surfaces sf_pst track. The mop-up
endgame king table (PR #140, formulaic) is covered by kEndSpread_lt.
-/

import Sunfish.GameTree

set_option maxRecDepth 4096

namespace Sunfish
namespace EvalBounds

/-- piece['P'] + pst['P'], all 64 squares (sunfish.py's padding fold). -/
def sqP : List Int := [100, 100, 100, 100, 100, 100, 100, 100, 170, 175, 177, 166, 211, 174, 177, 181, 106, 126, 119, 140, 136, 128, 140, 106, 85, 114, 98, 114, 113, 100, 114, 88, 77, 103, 109, 108, 105, 101, 100, 79, 80, 108, 105, 90, 91, 98, 103, 83, 72, 107, 94, 67, 68, 87, 103, 72, 100, 100, 100, 100, 100, 100, 100, 100]

/-- piece['N'] + pst['N'], all 64 squares (sunfish.py's padding fold). -/
def sqN : List Int := [207, 222, 197, 197, 269, 219, 216, 203, 277, 273, 390, 240, 284, 348, 276, 265, 291, 354, 281, 361, 360, 310, 348, 278, 306, 306, 330, 321, 316, 325, 308, 299, 279, 286, 314, 303, 304, 319, 282, 280, 260, 291, 294, 304, 300, 297, 292, 265, 255, 263, 282, 280, 282, 280, 255, 258, 199, 255, 251, 254, 259, 241, 256, 204]

/-- piece['B'] + pst['B'], all 64 squares (sunfish.py's padding fold). -/
def sqB : List Int := [255, 234, 230, 236, 295, 202, 279, 265, 308, 342, 359, 274, 277, 354, 322, 296, 310, 363, 285, 365, 377, 309, 351, 305, 348, 339, 342, 357, 349, 348, 337, 331, 334, 331, 339, 345, 339, 338, 320, 328, 335, 348, 346, 337, 329, 348, 342, 337, 341, 342, 332, 327, 328, 327, 342, 338, 312, 322, 303, 307, 305, 303, 309, 309]

/-- piece['R'] + pst['R'], all 64 squares (sunfish.py's padding fold). -/
def sqR : List Int := [504, 499, 502, 482, 505, 502, 518, 514, 518, 499, 518, 526, 518, 522, 503, 521, 492, 504, 499, 502, 511, 498, 497, 490, 479, 483, 490, 488, 492, 476, 473, 475, 459, 454, 468, 464, 470, 459, 447, 458, 450, 459, 450, 461, 461, 454, 461, 447, 442, 452, 457, 461, 459, 449, 448, 442, 458, 462, 466, 483, 478, 466, 457, 457]

/-- piece['Q'] + pst['Q'], all 64 squares (sunfish.py's padding fold). -/
def sqQ : List Int := [936, 930, 920, 815, 1005, 955, 1026, 958, 944, 964, 995, 918, 951, 1013, 992, 955, 927, 976, 964, 995, 1008, 998, 976, 931, 930, 911, 953, 948, 957, 951, 915, 922, 914, 912, 927, 923, 928, 918, 907, 905, 896, 922, 915, 917, 911, 917, 911, 899, 889, 909, 929, 908, 912, 912, 906, 887, 886, 896, 895, 915, 895, 889, 892, 883]

/-- piece['K'] + pst['K'], all 64 squares (sunfish.py's padding fold). -/
def sqK : List Int := [60004, 60049, 60042, 59911, 59911, 60054, 60075, 59944, 59971, 60009, 60050, 60050, 60050, 60050, 60009, 60003, 59944, 60011, 59949, 60040, 59940, 60025, 60033, 59972, 59950, 60045, 60010, 59996, 59983, 60012, 60000, 59956, 59950, 59961, 59953, 59975, 59954, 59958, 59993, 59955, 59958, 59962, 59961, 59929, 59942, 59971, 59974, 59971, 59996, 60003, 59987, 59955, 59949, 59984, 60012, 60004, 60015, 60027, 59997, 59987, 60005, 59999, 60036, 60016]

def tmax (t : List Int) : Int := t.foldr max (-1000000)
def tmin (t : List Int) : Int := t.foldr min 1000000

/-- Largest square value any non-king piece can contribute. -/
def nkMax : Int := max (tmax sqP) (max (tmax sqN) (max (tmax sqB) (max (tmax sqR) (tmax sqQ))))

/-- Non-king square values are never negative: each piece value
dominates its most negative table entry, so a side's non-king sum is
monotone in its piece multiset and a bare king is the minimum. -/
theorem nk_nonneg :
    (0 <= tmin sqP && 0 <= tmin sqN && 0 <= tmin sqB &&
     0 <= tmin sqR && 0 <= tmin sqQ) = true := by decide

/-- The eval bound: king-table spread plus 15 maximal non-king pieces. -/
def evalBound : Int := (tmax sqK - tmin sqK) + 15 * nkMax

/-- **The discharge**: the concrete bound sits below MATE_LOWER --
strictly stronger than `Bounded`'s [-MATE_UPPER, MATE_UPPER] band. -/
theorem evalBound_lt_MATE_LOWER : evalBound < MATE_LOWER := by decide

theorem evalBound_lt_MATE_UPPER : evalBound < MATE_UPPER := by decide

/-- The mop-up endgame king table (70 - 10 * center manhattan distance,
PR #140): its spread also keeps the bound below MATE_LOWER, so swapping
it in cannot break `Bounded`. -/
def kEndVals : List Int := [-56, -40, -24, -8, -8, -24, -40, -56, -40, -24, -8, 8, 8, -8, -24, -40, -24, -8, 8, 24, 24, 8, -8, -24, -8, 8, 24, 40, 40, 24, 8, -8, -8, 8, 24, 40, 40, 24, 8, -8, -24, -8, 8, 24, 24, 8, -8, -24, -40, -24, -8, 8, 8, -8, -24, -40, -56, -40, -24, -8, -8, -24, -40, -56]

theorem kEndSpread_lt : (tmax kEndVals - tmin kEndVals) + 15 * nkMax < MATE_LOWER := by
  decide


/-! ### The MATE_LOWER margin leak (machine-checked finding)

MATE_LOWER = K - 10 * Q was meant to mean "capturing the king while up
to ten queens behind still scores above MATE_LOWER" -- equivalently,
the king-gone check `pos.score <= -MATE_LOWER` catches every kingless
position. The margin is SHORT: a full promoted army (nine queens plus
2R+2B+2N, with PST bonuses) sums past ten bare queen values. -/

/-- Largest non-king army one side can ever field: 8 promotions all to
queens + the original queen, plus 2R, 2B, 2N, each on its best square.
(Composition-exact, unlike the loose 15 * nkMax bound above.) -/
def armyMax : Int := 9 * tmax sqQ + 2 * tmax sqR + 2 * tmax sqB + 2 * tmax sqN

theorem armyMax_exceeds_margin : armyMax > 10 * 929 := by decide

/-- **The leak**: a kingless side facing a maximal army has mover-view
score above -MATE_LOWER, so the king-gone short-circuit misses it.
(Worst case: opponent keeps king on its cheapest square, kingless side
holds armyMax.) -/
theorem kingGone_check_leaked : armyMax - tmin sqK > -(60000 - 10 * 929) := by decide

/-- The repair (now shipped): `MATE_LOWER = K - 13 * Q = 47923` closes
the leak - every kingless position scores at or below -MATE_LOWER,
while static evals keep their margin (`evalBound_lt_MATE_LOWER`). -/
theorem margin_covers : MATE_LOWER <= tmin sqK - armyMax := by decide


/-! ### The move-value floor (backing for `ValFloor`, Sunfish/Stalemate.lean)

`pos.value(move)` (sunfish.py lines 269-290) is the mover's own table
delta `pst[p][j] - pst[p][i]` plus additive terms that are all
NONNEGATIVE given the shipped tables:

* capture: `+ pst[q][119-j]` -- every table is nonnegative
  (`capture_terms_nonneg`, extending `nk_nonneg` with the king);
* the kp "castling check detection" bonus: `+ pst[K][119-j]` ≥ 59911;
* promotion: `+ pst[prom][j] - pst[P][j]` -- on every promotion square
  each of N, B, R, Q beats the pawn (`promotion_terms_nonneg`);
* en passant: `+ pst[P][...]` ≥ 63;
* castling rook relocation: corner → D1 gains 25, corner → F1 gains 9
  (`castle_rook_deltas`; the rotation means only the rank-1 corners of
  the white-view table are ever used).

So the floor of `pos.value` is the worst table delta, `-quietDropMax =
-211` (attained by the queen, 815 - 1026).  This is the concrete number
behind the abstract `ValFloor` hypothesis: the link from the board
string to these tables is not modeled (same caveat as `Bounded` above),
but every numeric fact is machine-checked here. -/

/-- Largest-minus-smallest square value of a table: the worst delta a
quiet move of that piece can score. -/
def spread (t : List Int) : Int := tmax t - tmin t

/-- The move-value floor is `-quietDropMax`: no quiet move drops more
than the queen's worst-case 211. -/
def quietDropMax : Int :=
  max (spread sqP) (max (spread sqN) (max (spread sqB)
    (max (spread sqR) (max (spread sqQ) (spread sqK)))))

theorem quietDropMax_eq : quietDropMax = 211 := by decide

/-- Every table (king included) is nonnegative, so capture and kp-bonus
terms of `pos.value` only add. -/
theorem capture_terms_nonneg :
    (0 <= tmin sqP && 0 <= tmin sqN && 0 <= tmin sqB &&
     0 <= tmin sqR && 0 <= tmin sqQ && 0 <= tmin sqK) = true := by decide

/-- On each of the eight promotion squares, promoting to any of N, B, R,
Q gains over the pawn value there (`pst[prom][j] - pst[P][j] ≥ 0`). -/
theorem promotion_terms_nonneg :
    (((List.zipWith (· - ·) (sqN.take 8) (sqP.take 8)).all (fun x => decide (0 ≤ x))) &&
     ((List.zipWith (· - ·) (sqB.take 8) (sqP.take 8)).all (fun x => decide (0 ≤ x))) &&
     ((List.zipWith (· - ·) (sqR.take 8) (sqP.take 8)).all (fun x => decide (0 ≤ x))) &&
     ((List.zipWith (· - ·) (sqQ.take 8) (sqP.take 8)).all (fun x => decide (0 ≤ x)))) = true := by
  decide

/-- The two castling rook relocations gain value: a1 → d1 is +25,
h1 → f1 is +9 (64-square indices 56/59 and 63/61). -/
theorem castle_rook_deltas :
    sqR.getD 59 0 - sqR.getD 56 0 = 25 ∧ sqR.getD 61 0 - sqR.getD 63 0 = 9 := by
  decide

/-- A king capture's value is dominated by the captured king's square
value (≥ tmin sqK = 59911), which clears `MATE_LOWER` even after the
worst mover drop -- the concrete backing for `KingCaptureValHigh`
(Sunfish/Stalemate.lean): king captures always pass the QS val-filter. -/
theorem kingCapture_val_above : MATE_LOWER + quietDropMax < tmin sqK := by decide

end EvalBounds
end Sunfish
