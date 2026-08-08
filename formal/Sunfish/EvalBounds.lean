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
def sqP : List Int := [100, 100, 100, 100, 100, 100, 100, 100, 178, 183, 186, 173, 202, 182, 185, 190, 107, 129, 121, 144, 140, 131, 144, 107, 83, 116, 98, 115, 114, 100, 115, 87, 74, 103, 110, 109, 106, 101, 100, 77, 78, 109, 105, 89, 90, 98, 103, 81, 69, 108, 93, 63, 64, 86, 103, 69, 100, 100, 100, 100, 100, 100, 100, 100]

/-- piece['N'] + pst['N'], all 64 squares (sunfish.py's padding fold). -/
def sqN : List Int := [214, 227, 205, 205, 270, 225, 222, 210, 277, 274, 380, 244, 284, 342, 276, 266, 290, 347, 281, 354, 353, 307, 342, 278, 304, 304, 325, 317, 313, 321, 305, 297, 279, 285, 311, 301, 302, 315, 282, 280, 262, 290, 293, 302, 298, 295, 291, 266, 257, 265, 282, 280, 282, 280, 257, 260, 206, 257, 254, 256, 261, 245, 258, 211]

/-- piece['B'] + pst['B'], all 64 squares (sunfish.py's padding fold). -/
def sqB : List Int := [261, 242, 238, 244, 297, 213, 283, 270, 309, 340, 355, 278, 281, 351, 322, 298, 311, 359, 288, 361, 372, 310, 348, 306, 345, 337, 340, 354, 346, 345, 335, 330, 333, 330, 337, 343, 337, 336, 320, 327, 334, 345, 344, 335, 328, 345, 340, 335, 339, 340, 331, 326, 327, 326, 340, 336, 313, 322, 305, 308, 306, 305, 310, 310]

/-- piece['R'] + pst['R'], all 64 squares (sunfish.py's padding fold). -/
def sqR : List Int := [514, 508, 512, 483, 516, 512, 535, 529, 534, 508, 535, 546, 534, 541, 513, 539, 498, 514, 507, 512, 524, 506, 504, 494, 479, 484, 495, 492, 497, 475, 470, 473, 451, 444, 463, 458, 466, 450, 433, 449, 437, 451, 437, 454, 454, 444, 453, 433, 426, 441, 448, 453, 450, 436, 435, 426, 449, 455, 461, 484, 477, 461, 448, 447]

/-- piece['Q'] + pst['Q'], all 64 squares (sunfish.py's padding fold). -/
def sqQ : List Int := [935, 930, 921, 825, 998, 953, 1017, 955, 943, 961, 989, 919, 949, 1005, 986, 953, 927, 972, 961, 989, 1001, 992, 972, 931, 930, 913, 951, 946, 954, 949, 916, 923, 915, 914, 927, 924, 928, 919, 909, 907, 899, 923, 916, 918, 913, 918, 913, 902, 893, 911, 929, 910, 914, 914, 908, 891, 890, 899, 898, 916, 898, 893, 895, 887]

/-- piece['K'] + pst['K'], all 64 squares (sunfish.py's padding fold). -/
def sqK : List Int := [60004, 60054, 60047, 59901, 59901, 60060, 60083, 59938, 59968, 60010, 60055, 60056, 60056, 60055, 60010, 60003, 59938, 60012, 59943, 60044, 59933, 60028, 60037, 59969, 59945, 60050, 60011, 59996, 59981, 60013, 60000, 59951, 59945, 59957, 59948, 59972, 59949, 59953, 59992, 59950, 59953, 59958, 59957, 59921, 59936, 59968, 59971, 59968, 59996, 60003, 59986, 59950, 59943, 59982, 60013, 60004, 60017, 60030, 59997, 59986, 60006, 59999, 60040, 60018]

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
def kEndVals : List Int := [-70, -50, -30, -10, -10, -30, -50, -70, -50, -30, -10, 10, 10, -10, -30, -50, -30, -10, 10, 30, 30, 10, -10, -30, -10, 10, 30, 50, 50, 30, 10, -10, -10, 10, 30, 50, 50, 30, 10, -10, -30, -10, 10, 30, 30, 10, -10, -30, -50, -30, -10, 10, 10, -10, -30, -50, -70, -50, -30, -10, -10, -30, -50, -70]

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
* the kp "castling check detection" bonus: `+ pst[K][119-j]` ≥ 59901;
* promotion: `+ pst[prom][j] - pst[P][j]` -- on every promotion square
  each of N, B, R, Q beats the pawn (`promotion_terms_nonneg`);
* en passant: `+ pst[P][...]` ≥ 63;
* castling rook relocation: corner → D1 gains 35, corner → F1 gains 14
  (`castle_rook_deltas`; the rotation means only the rank-1 corners of
  the white-view table are ever used).

So the floor of `pos.value` is the worst table delta, `-quietDropMax =
-192` (attained by the queen, 825 - 1017).  This is the concrete number
behind the abstract `ValFloor` hypothesis: the link from the board
string to these tables is not modeled (same caveat as `Bounded` above),
but every numeric fact is machine-checked here. -/

/-- Largest-minus-smallest square value of a table: the worst delta a
quiet move of that piece can score. -/
def spread (t : List Int) : Int := tmax t - tmin t

/-- The move-value floor is `-quietDropMax`: no quiet move drops more
than the queen's worst-case 192. -/
def quietDropMax : Int :=
  max (spread sqP) (max (spread sqN) (max (spread sqB)
    (max (spread sqR) (max (spread sqQ) (spread sqK)))))

theorem quietDropMax_eq : quietDropMax = 192 := by decide

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

/-- The two castling rook relocations gain value: a1 → d1 is +35,
h1 → f1 is +14 (64-square indices 56/59 and 63/61). -/
theorem castle_rook_deltas :
    sqR.getD 59 0 - sqR.getD 56 0 = 35 ∧ sqR.getD 61 0 - sqR.getD 63 0 = 14 := by
  decide

/-- A king capture's value is dominated by the captured king's square
value (≥ tmin sqK = 59901), which clears `MATE_LOWER` even after the
worst mover drop -- the concrete backing for `KingCaptureValHigh`
(Sunfish/Stalemate.lean): king captures always pass the QS val-filter. -/
theorem kingCapture_val_above : MATE_LOWER + quietDropMax < tmin sqK := by decide

end EvalBounds
end Sunfish
