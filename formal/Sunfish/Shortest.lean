/-
Distance-to-mate optimality: the search does not merely find a mate, it
finds the SHORTEST one -- and the constant that makes that true is
`EVAL_ROUGHNESS`, for a reason that is pure parity.

`Liveness.lean` ends at `forcedMate_play_mates`: from a root with a
forced mate in `k` plies, the engine's own iterated choice reaches a
checkmated position within `k` plies -- **the `k` the spec handed it**.
Nothing there says `k` was the best available.  This file replaces that
`k` with the LEAST one, which is the claim sunfish.py's constant block
has made since 2014:

    among winning lines the search takes the SHORTEST, and the losing
    side drags the mate out as long as it can (issue #11)

Four steps, of which the second is the one worth reading.

1. **`LeastMate`** -- the least `k` with `ForcedMate G k p`.  It exists
   because `ForcedMate` is upward closed (`forcedMate_mono`) and `Nat`
   is well ordered; the file takes it as a predicate so no choice
   principle is spent.

2. **Parity** (`leastMate_odd`).  The least mate budget is ODD.  This is
   not a chess fact and not an arithmetic accident: it is the shape of
   `ForcedMate` itself.  `mate` costs one ply, `step` costs two, and
   `step`'s reply quantifier is non-vacuous (`hnt` names a legal reply
   via `legal_of_allIllegalB_false`), so an EVEN budget always has one
   ply of slack -- `forcedMate_pred_of_even` hands it back.  Hence two
   distinct achievable mate distances differ by at least TWO plies.

3. **Block exactness** (`leastMate_value_block`).  No new induction: the
   forward spine `forcedMate_complete` puts the declared value at or
   above `MATE_LOWER + (D - k) * EVAL_ROUGHNESS`, and the
   distance-carrying converse `forcedMate_of_value_dist` puts it
   STRICTLY BELOW the next rung, because a value one rung higher would
   exhibit a mate in `k - 1` and contradict leastness.  So the value's
   rung index is exactly `D - k`.

4. **Separation** (`leastMate_value_separation`).  Two positions whose
   least mate distances differ have declared values more than
   `EVAL_ROUGHNESS` apart.  Steps 2 and 3 compose: distinct distances
   are two plies apart, two plies are two rungs, and two rungs leave a
   full rung of clear air between the blocks.

Why that last number is the whole point.  `search` stops bisecting at
`upper - lower <= EVAL_ROUGHNESS`, so the shipped root is only ever
guaranteed a move within `EVAL_ROUGHNESS` of the maximum -- the
idealisation `MaximalChoice` records and `formal/README.md` flags.
Separation says the gap between a faster and a slower mate is STRICTLY
MORE than `EVAL_ROUGHNESS`, so that tolerance cannot cross it:
`NearMaximalChoice` below is `MaximalChoice` weakened by exactly the
driver's own stopping tolerance, and `forcedMate_play_mates` still
holds under it (`forcedMate_play_shortest`).

So `EVAL_ROUGHNESS`-per-ply is not a margin someone measured and
rounded up.  It is the smallest step for which the theorem is true, and
it is true only because the achievable distances have a fixed parity.
At one point per ply (the pre-#172 alternative) the gap would be 2 and
the tolerance 15, and the shipped driver could take the slower mate.

**The defender half, step 5.**  "The losing side drags the mate out as
long as it can" is the same ordering read the other way, and it needs
no second move rule to model.  The engine minimises the value of the
position it moves to, always; at a lost node the positions it moves to
are attacker-to-move and their values are the attacker's POSITIVE mate
values `MATE_LOWER + (d - n) * EVAL_ROUGHNESS`, so minimising the value
IS maximising the attacker's remaining distance.  There is no
`NearMinimalChoice`: in negamax the defender's rule is literally the
attacker's rule, and the duality lives in the theorem rather than in
the choice.

`defence_resistance_step` is the local step, in SPEC form rather than
the distance form of `defence_maximal_resistance`: if the engine's own
defence lets the attacker mate in `n`, then the position was ALREADY
mated within `n` against every defence, so the engine gave nothing
away.  `defence_resists` iterates it by strong recursion into
`ResistsFor`, the inductive dual of `MatesWithin`.  Parity pays for the
driver's tolerance once, inside the local step: near-maximality admits
alternatives one rung worse, `forcedMate_of_value_dist` reads that rung
as a mate in `n + 1`, and `n + 1` is EVEN because `n` is odd -- so
`forcedMate_odd_le` hands the ply straight back, exactly as it does on
the attacker side.

**The `3` in the null reduction is load-bearing here.**  Parity is
preserved along every path because both of the search's depth steps are
odd: a real move spends one ply of depth per negation, and the null
option spends three (`nullValueD2`'s `d + 1 - 3`) per negation.  An even
null reduction would let a single line reach a mate value of the wrong
rung parity and the separation argument would collapse to a gap of
`EVAL_ROUGHNESS`, exactly the width the driver cannot resolve.  Nothing
in this file mentions the null term -- the parity lives in `ForcedMate`,
whose `step` constructor is two plies -- but a change to that constant
is a change to this theorem.
-/

import Sunfish.Liveness

namespace Sunfish

/-! ### Leastness -/

/-- There is no forced mate in zero plies: `mate` costs one ply and
`step` costs two, so every derivation has a positive index. -/
theorem not_forcedMate_zero (G : QSGame) (p : G.Pos) : ¬ ForcedMate G 0 p := by
  intro h
  cases h

/-- **Parity, the load-bearing lemma.**  An EVEN mate budget is never
tight: one ply can always be handed back.

`mate` is stated at every index, so it re-derives one lower for free.
`step` is the interesting case: its budget is `k + 2` with every legal
reply mating in `k`, and `hnt : allIllegalB G m = false` NAMES a legal
reply (`legal_of_allIllegalB_false`), so `ForcedMate G k` is actually
inhabited and `k ≠ 0`.  An even `k` is therefore at least `2`, the
induction hypothesis tightens every reply to `k - 1`, and `step`
re-applies at `k - 1 + 2 = k + 1`. -/
theorem forcedMate_pred_of_even (G : QSGame) :
    ∀ {n : Nat} {p : G.Pos}, ForcedMate G n p → n % 2 = 0 →
      ForcedMate G (n - 1) p := by
  intro n p h
  induction h with
  | @mate k p m hkg hm hleg hmate =>
    intro hev
    obtain ⟨k', rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
    have : ForcedMate G (k' + 1) p := ForcedMate.mate hkg hm hleg hmate
    simpa using this
  | @step k p m hkg hm hleg hnt hrep ih =>
    intro hev
    obtain ⟨m0, hm0, hleg0⟩ := legal_of_allIllegalB_false hnt
    have hk0 : ForcedMate G k m0 := hrep m0 hm0 hleg0
    have hkne : k ≠ 0 := by
      intro h0
      exact not_forcedMate_zero G m0 (h0 ▸ hk0)
    have hkev : k % 2 = 0 := by omega
    obtain ⟨k', rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
    have : ForcedMate G (k' + 2) p :=
      ForcedMate.step hkg hm hleg hnt
        (fun m' hm' hl' => by simpa using ih m' hm' hl' hkev)
    simpa using this

/-- **`LeastMate G k p`**: `k` is the exact distance to mate for the
side to move at `p`.  Stated as a predicate rather than a function so
that no well-ordering instance and no choice principle is spent; it is
unique where it holds. -/
def LeastMate (G : QSGame) (k : Nat) (p : G.Pos) : Prop :=
  ForcedMate G k p ∧ ∀ j, ForcedMate G j p → k ≤ j

theorem leastMate_unique (G : QSGame) {j k : Nat} {p : G.Pos}
    (hj : LeastMate G j p) (hk : LeastMate G k p) : j = k :=
  Nat.le_antisymm (hj.2 k hk.1) (hk.2 j hj.1)

/-- The least distance is at least one ply. -/
theorem leastMate_pos (G : QSGame) {k : Nat} {p : G.Pos}
    (h : LeastMate G k p) : 1 ≤ k := by
  rcases Nat.eq_zero_or_pos k with hk | hk
  · exact absurd (hk ▸ h.1) (not_forcedMate_zero G p)
  · exact hk

/-- **The least mate distance is odd.**  Immediate from
`forcedMate_pred_of_even`: an even least budget would admit a strictly
smaller one. -/
theorem leastMate_odd (G : QSGame) {k : Nat} {p : G.Pos}
    (h : LeastMate G k p) : k % 2 = 1 := by
  have hk1 := leastMate_pos G h
  by_cases hev : k % 2 = 0
  · exact absurd (h.2 (k - 1) (forcedMate_pred_of_even G h.1 hev)) (by omega)
  · omega

/-- **Distinct mate distances are two plies apart.**  Parity, spent.
This is the whole reason a ply may be priced at one `EVAL_ROUGHNESS`
rather than two. -/
theorem leastMate_gap (G : QSGame) {j k : Nat} {p q : G.Pos}
    (hj : LeastMate G j p) (hk : LeastMate G k q) (hlt : j < k) :
    j + 2 ≤ k := by
  have hjo := leastMate_odd G hj
  have hko := leastMate_odd G hk
  omega

/-! ### Block exactness -/

/-- **The declared value's rung index is exactly `D - k`.**

Lower bound: the forward spine (`forcedMate_complete`) at the least
`k`.  Upper bound: leastness.  If the value reached the NEXT rung,
`MATE_LOWER + (D - k + 1) * EVAL_ROUGHNESS`, then
`forcedMate_of_value_dist` would exhibit some `n` with
`n + (D - k + 1) <= D`, i.e. `n <= k - 1`, a forced mate strictly
nearer than the least one.

Both bounds together say the value lies in the half-open rung
`[MATE_LOWER + (D-k)*ER, MATE_LOWER + (D-k+1)*ER)`.  Note what is NOT
claimed: the value is not asserted to sit exactly ON a rung.  It does
not have to.  Everything downstream only needs the rung index, and
proving the block is free -- it is two existing theorems and leastness,
with no new induction over the tree. -/
theorem leastMate_value_block (G : QSGame) (guard : G.Pos → Bool)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G) (hZ : NoZugzwang G guard)
    {k D : Nat} {p : G.Pos}
    (hcapf : hasKingCapture G.toNullGame.toGame p = false)
    (hLM : LeastMate G k p) (hkD : k + 1 ≤ D)
    (hspan : (D : Int) * EVAL_ROUGHNESS ≤ 21366) :
    MATE_LOWER + ((D : Int) - (k : Int)) * EVAL_ROUGHNESS
        ≤ nullValueD2 G guard D p ∧
      nullValueD2 G guard D p
        < MATE_LOWER + ((D : Int) - (k : Int) + 1) * EVAL_ROUGHNESS := by
  have hML : MATE_LOWER = 47923 := rfl
  have hER : EVAL_ROUGHNESS = 15 := rfl
  have hkD' : (k : Int) + 1 ≤ (D : Int) := by exact_mod_cast hkD
  constructor
  · have h := forcedMate_complete G guard hF hZ hLM.1 D hkD
    simp only [mateFloor] at h
    simp only [hER] at hspan ⊢
    simp only [hER] at h
    have hnn : 0 ≤ ((D : Int) - (k : Int)) * 15 := by
      have : (0 : Int) ≤ (D : Int) - (k : Int) := by omega
      omega
    omega
  · by_cases hge : nullValueD2 G guard D p
        < MATE_LOWER + ((D : Int) - (k : Int) + 1) * EVAL_ROUGHNESS
    · exact hge
    · exfalso
      -- the value reaches rung `D - k + 1`, so that is a legal `t`
      have ht : MATE_LOWER + ((D + 1 - k : Nat) : Int) * EVAL_ROUGHNESS
          ≤ nullValueD2 G guard D p := by
        have hc : ((D + 1 - k : Nat) : Int) = (D : Int) - (k : Int) + 1 := by omega
        rw [hc]
        simp only [hML, hER] at hge ⊢
        omega
      obtain ⟨n, hn1, hn2, hn3⟩ :=
        forcedMate_of_value_dist G guard hF hQ hNM D (D + 1 - k) p hcapf ht
      have := hLM.2 n hn3
      omega

/-- **Separation: a faster mate outscores a slower one by MORE than the
driver's stopping tolerance.**

`j < k` are two least mate distances at the same remaining depth `D`.
Parity makes them two plies apart (`leastMate_gap`), so their rung
indices `D - j` and `D - k` are two apart, so the block of `p` starts a
full rung above where the block of `q` ends.

The conclusion is stated as a STRICT `EVAL_ROUGHNESS <` gap because
that is exactly the shape the driver needs: `search` stops at
`upper - lower <= EVAL_ROUGHNESS`, and a difference strictly greater
than `EVAL_ROUGHNESS` cannot fit inside such a window.  Halve the
per-ply price and this theorem is false. -/
theorem leastMate_value_separation (G : QSGame) (guard : G.Pos → Bool)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G) (hZ : NoZugzwang G guard)
    {j k D : Nat} {p q : G.Pos}
    (hcapp : hasKingCapture G.toNullGame.toGame p = false)
    (hcapq : hasKingCapture G.toNullGame.toGame q = false)
    (hjp : LeastMate G j p) (hkq : LeastMate G k q) (hlt : j < k)
    (hjD : j + 1 ≤ D) (hkD : k + 1 ≤ D)
    (hspan : (D : Int) * EVAL_ROUGHNESS ≤ 21366) :
    EVAL_ROUGHNESS
      < nullValueD2 G guard D p - nullValueD2 G guard D q := by
  have hER : EVAL_ROUGHNESS = 15 := rfl
  have hgap := leastMate_gap G hjp hkq hlt
  have hgap' : (j : Int) + 2 ≤ (k : Int) := by exact_mod_cast hgap
  have hlo := (leastMate_value_block G guard hF hQ hNM hZ hcapp hjp hjD hspan).1
  have hhi := (leastMate_value_block G guard hF hQ hNM hZ hcapq hkq hkD hspan).2
  simp only [hER] at hlo hhi ⊢
  omega

/-! ### The driver's own tolerance is enough -/

/-- **`MaximalChoice`, weakened by exactly the driver's stopping
tolerance.**  `search` bisects only until
`upper - lower <= EVAL_ROUGHNESS`, so the move left in `tp_move` is
guaranteed only to be within `EVAL_ROUGHNESS` of the best -- which is
what `MaximalChoice` idealises away and `formal/README.md` flags as an
idealisation.  This is the honest version: `ch p` minimises the child's
declared value up to `EVAL_ROUGHNESS`.

`MaximalChoice` implies it (`nearMaximalChoice_of_maximal`), and every
theorem below is stated for the weak form. -/
def NearMaximalChoice (G : QSGame) (guard : G.Pos → Bool) (d : Nat)
    (ch : G.Pos → G.Pos) : Prop :=
  ∀ p, allIllegalB G p = false →
    ch p ∈ movesAbove G (val_lower (d + 1)) p ∧
      ∀ m ∈ movesAbove G (val_lower (d + 1)) p,
        nullValueD2 G guard d (ch p) ≤ nullValueD2 G guard d m + EVAL_ROUGHNESS

theorem nearMaximalChoice_of_maximal (G : QSGame) (guard : G.Pos → Bool)
    (d : Nat) (ch : G.Pos → G.Pos) (h : MaximalChoice G guard d ch) :
    NearMaximalChoice G guard d ch := by
  intro p hai
  obtain ⟨hmem, hmax⟩ := h p hai
  refine ⟨hmem, fun m hm => ?_⟩
  have := hmax m hm
  have hER : (0 : Int) ≤ EVAL_ROUGHNESS := by decide
  omega

/-- Every forced mate has an ODD witness no larger than itself: the
tightening step, `forcedMate_pred_of_even` packaged for the recursion. -/
theorem forcedMate_odd_le (G : QSGame) {n : Nat} {p : G.Pos}
    (h : ForcedMate G n p) :
    ∃ n', n' ≤ n ∧ n' % 2 = 1 ∧ ForcedMate G n' p := by
  by_cases hev : n % 2 = 0
  · have hne : n ≠ 0 := fun h0 => not_forcedMate_zero G p (h0 ▸ h)
    exact ⟨n - 1, by omega, by omega, forcedMate_pred_of_even G h hev⟩
  · exact ⟨n, Nat.le_refl _, by omega, h⟩

/-! ### The attacker half, under the driver's real stopping rule -/

/-- **`forcedMate_play_mates`, with `MaximalChoice` replaced by
`NearMaximalChoice`.**  The engine's own move choice -- now only
required to be within `EVAL_ROUGHNESS` of the best, which is all the
shipped bisection guarantees -- still mates within `k` plies.

The proof is `forcedMate_play_mates`'s, with one slack ply everywhere
and parity to pay for it.  Near-maximality loses exactly one rung:
where the exact argmax hands the reached position a value at or below
`-(MATE_LOWER + (D-k)*ER)`, the tolerant choice can only be pinned at
`-(MATE_LOWER + (D-k-1)*ER)`.  Pushed through
`forcedlyMated_of_value_dist` that yields replies with a forced mate in
`n <= k - 1` rather than `n <= k - 2`, one ply too many.

`forcedMate_odd_le` recovers it.  `k` is odd, so `k - 1` is EVEN, so an
even budget is never tight and every reply's odd witness is at most
`k - 2` after all.  The lost rung was never reachable.

This is what the separation lemma buys, in the place it matters:
`EVAL_ROUGHNESS` of tolerance is precisely affordable, and it is
affordable only because achievable distances have a fixed parity. -/
theorem forcedMate_play_shortest_odd (G : QSGame) (guard : G.Pos → Bool)
    (ch : G.Pos → G.Pos) (d : Nat)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G) (hZ : NoZugzwang G guard)
    (hch : NearMaximalChoice G guard d ch) :
    ∀ (k : Nat) (p : G.Pos),
      k % 2 = 1 → k + 1 ≤ d + 1 → (d : Int) * EVAL_ROUGHNESS ≤ 21366 →
      hasKingCapture G.toNullGame.toGame p = false →
      ForcedMate G k p →
      MatesWithin G ch k p := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  intro k
  induction k using Nat.strongRecOn with
  | _ k ih =>
    intro p hkodd hkd hspan hcapf hFM
    have hk1 : 1 ≤ k := by omega
    have hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true) := by simp [hcapf]
    -- 1. the root's declared value carries the distance
    have hval := forcedMate_complete G guard hF hZ hFM (d + 1) hkd
    simp only [EVAL_ROUGHNESS] at hspan
    have hvalv : MATE_LOWER + (((d + 1 - k : Nat)) : Int) * 15
        ≤ nullValueD2 G guard (d + 1) p := by
      simp only [mateFloor, EVAL_ROUGHNESS] at hval
      have hc : (((d + 1 - k : Nat)) : Int) = (d : Int) + 1 - (k : Int) := by omega
      rw [hc]
      omega
    have hnn : (0 : Int) ≤ (((d + 1 - k : Nat)) : Int) := Int.ofNat_nonneg _
    -- 2. the root is neither kingless nor terminal
    have hkg : ¬ (G.eval p ≤ -MATE_LOWER) := by
      intro hh
      rw [nullValueD2_kingGone G guard (d + 1) p hh] at hvalv
      omega
    have hai : allIllegalB G p = false := by
      cases hb : allIllegalB G p with
      | false => rfl
      | true =>
        exfalso
        rw [nullValueD2_of_allIllegal G guard d p hkg hcap hb] at hvalv
        have h2 := (terminalValue_bounds G (d + 1) p).2
        omega
    -- 3. the chosen move is within one bracket of the fold's witness
    obtain ⟨hmem, hnear⟩ := hch p hai
    have hvfold := hvalv
    rw [nullValueD2_of_fold G guard d p hkg hcap hai] at hvfold
    have hnn15 : (0 : Int) ≤ (((d + 1 - k : Nat)) : Int) * 15 := by omega
    obtain ⟨mw, hmw, hmwv⟩ :=
      foldMax_failHigh_witness (fun x => -(nullValueD2 G guard d x))
        (movesAbove G (val_lower (d + 1)) p) (nullTermD2 G guard d p)
        (by have := nullTermD2_lt_ML G guard d p; omega) hvfold
    -- ONE RUNG of slack: this is where the tolerance is spent.
    have hchv : nullValueD2 G guard d (ch p)
        ≤ -(MATE_LOWER + (((d - k : Nat)) : Int) * 15) := by
      have h1 := hnear mw hmw
      simp only [EVAL_ROUGHNESS] at h1
      have hc : (((d + 1 - k : Nat)) : Int) = (((d - k : Nat)) : Int) + 1 := by omega
      omega
    have hnn2 : (0 : Int) ≤ (((d - k : Nat)) : Int) := Int.ofNat_nonneg _
    -- 4. the reached position is legal, with its king on the board
    have hmm : ch p ∈ G.moves p := movesAbove_subset G _ p (ch p) hmem
    have hkgc : ¬ (G.eval (ch p) ≤ -MATE_LOWER) := fun hh =>
      hcap ((hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨ch p, hmm, hh⟩)
    have hcapc : hasKingCapture G.toNullGame.toGame (ch p) = false := by
      cases hc : hasKingCapture G.toNullGame.toGame (ch p) with
      | false => rfl
      | true =>
        exfalso
        rw [nullValueD2_of_capture G guard d (ch p) hkgc hc] at hchv
        omega
    -- 5. the distance-carrying converse at the reached position
    obtain ⟨d0, rfl⟩ : ∃ d0, d = d0 + 1 := ⟨d - 1, by omega⟩
    have hcv : nullValueD2 G guard (d0 + 1) (ch p)
        ≤ -(MATE_LOWER + (((d0 + 1 - k : Nat)) : Int) * EVAL_ROUGHNESS) := by
      simp only [EVAL_ROUGHNESS]
      exact hchv
    have hFL := forcedlyMated_of_value_dist G guard hF hQ hNM d0
      (d0 + 1 - k) (ch p) hcapc hkgc hcv
    -- 6. read off the play
    cases hFL with
    | inl hcm =>
      obtain ⟨n, hn⟩ : ∃ n, k = n + 1 := ⟨k - 1, by omega⟩
      rw [hn]
      exact MatesWithin.mate hcm
    | inr hrest =>
      obtain ⟨hnt, n, hn1, hn2, hall⟩ := hrest
      -- `n <= k - 1` is one ply too many; PARITY tightens it to `k - 2`.
      have hnk : n + 1 ≤ k := by omega
      refine matesWithin_mono G ch (MatesWithin.step (n := k - 2) hnt ?_) k (by omega)
      intro m hm hleg
      obtain ⟨n', hn'le, hn'odd, hn'FM⟩ := forcedMate_odd_le G (hall m hm hleg)
      have hn'k : n' + 2 ≤ k := by omega
      have := ih n' (by omega) m hn'odd (by omega)
        (by simp only [EVAL_ROUGHNESS]; omega) hleg hn'FM
      exact matesWithin_mono G ch this (k - 2) (by omega)

/-- **DISTANCE-TO-MATE OPTIMALITY, attacker half.**  From a root whose
EXACT distance to mate is `k`, the engine's own move choice -- required
only to be within the driver's own `EVAL_ROUGHNESS` stopping tolerance
of the best -- mates within `k` plies against every legal defence.

`k` is the least forced-mate distance, so no strategy whatsoever mates
faster against best defence: the engine attains the game-theoretic
optimum, which is the "shortest PV" half of sunfish.py's constant-block
claim.  Parity (`leastMate_odd`) is what connects the two statements --
it is what makes a least distance an ODD one, and the odd distances are
exactly the ones the driver's final bracket can tell apart. -/
theorem leastMate_play_shortest (G : QSGame) (guard : G.Pos → Bool)
    (ch : G.Pos → G.Pos) (d : Nat)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G) (hZ : NoZugzwang G guard)
    (hch : NearMaximalChoice G guard d ch)
    {k : Nat} {p : G.Pos}
    (hLM : LeastMate G k p) (hkd : k + 1 ≤ d + 1)
    (hspan : (d : Int) * EVAL_ROUGHNESS ≤ 21366)
    (hcapf : hasKingCapture G.toNullGame.toGame p = false) :
    MatesWithin G ch k p :=
  forcedMate_play_shortest_odd G guard ch d hF hQ hNM hZ hch k p
    (leastMate_odd G hLM) hkd hspan hcapf hLM.1

/-! ### The defender half: maximal resistance -/

/-- The mated side's dual parity step.  `ForcedlyMated`'s budget is the
ATTACKER's, one ply past the defender's reply, so the two disjuncts read
`0` (mate is here) and `k + 1` plies.  An even attacker budget is never
tight, for the same reason as `forcedMate_pred_of_even` and by direct
appeal to it at every legal reply. -/
theorem forcedlyMated_pred_of_even (G : QSGame) {k : Nat} {q : G.Pos}
    (h : ForcedlyMated G k q) (hev : k % 2 = 0) :
    ForcedlyMated G (k - 1) q := by
  cases h with
  | inl hcm => exact Or.inl hcm
  | inr hrest =>
    obtain ⟨hai, hall⟩ := hrest
    exact Or.inr ⟨hai, fun m hm hleg =>
      forcedMate_pred_of_even G (hall m hm hleg) hev⟩

/-- The least attacker budget at a mated node, dual to `LeastMate`. -/
def LeastMated (G : QSGame) (k : Nat) (q : G.Pos) : Prop :=
  ForcedlyMated G k q ∧ ∀ j, ForcedlyMated G j q → k ≤ j

/-- **A mated node's distance is EVEN.**  Either mate is here (`k = 0`,
zero plies) or the attacker's least remaining budget is odd, so the
defender's own distance `k + 1` is even.  The dual of `leastMate_odd`,
and the reason the same two-rung separation holds on the losing side. -/
theorem leastMated_odd_or_zero (G : QSGame) {k : Nat} {q : G.Pos}
    (h : LeastMated G k q) : k = 0 ∨ k % 2 = 1 := by
  by_cases hz : k = 0
  · exact Or.inl hz
  · refine Or.inr ?_
    by_cases hev : k % 2 = 0
    · exact absurd (h.2 (k - 1) (forcedlyMated_pred_of_even G h.1 hev)) (by omega)
    · omega

/-- **MAXIMAL RESISTANCE, the defender half's local step.**  At a lost
position the engine's own choice is a distance-MAXIMAL defence: no legal
reply `m` leaves the attacker a mate nearer than the one the engine's
move `ch q` leaves.

`MaximalChoice` minimises the reached position's declared value, and at
a lost node those values are the attacker's positive mate values
`MATE_LOWER + (d - n) * EVAL_ROUGHNESS` -- so minimising the value is
MAXIMISING `n`.  That is the whole of "the losing side drags the mate
out as long as it can", read off the same ordering the attacker half
reads the other way.

Parity refunds the tolerance here too, and in exactly the same place:
the block bounds plus one `EVAL_ROUGHNESS` of slack give `i <= j + 1`,
and `i` and `j` are both odd (`leastMate_odd`), so `i <= j`.  The
engine may be one rung short of the true argmax and still cannot be
talked into a faster loss. -/
theorem defence_maximal_resistance (G : QSGame) (guard : G.Pos → Bool)
    (ch : G.Pos → G.Pos) (d : Nat)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G) (hZ : NoZugzwang G guard)
    (hch : NearMaximalChoice G guard d ch)
    {q m : G.Pos} {i j : Nat}
    (hai : allIllegalB G q = false)
    (hm : m ∈ movesAbove G (val_lower (d + 1)) q)
    (hcapc : hasKingCapture G.toNullGame.toGame (ch q) = false)
    (hcapm : hasKingCapture G.toNullGame.toGame m = false)
    (hLMj : LeastMate G j (ch q)) (hLMi : LeastMate G i m)
    (hjd : j + 1 ≤ d) (hid : i + 1 ≤ d)
    (hspan : (d : Int) * EVAL_ROUGHNESS ≤ 21366) :
    i ≤ j := by
  have hER : EVAL_ROUGHNESS = 15 := rfl
  obtain ⟨_, hnear⟩ := hch q hai
  have h1 := hnear m hm
  have hlo := (leastMate_value_block G guard hF hQ hNM hZ hcapc hLMj hjd hspan).1
  have hhi := (leastMate_value_block G guard hF hQ hNM hZ hcapm hLMi hid hspan).2
  have hjo := leastMate_odd G hLMj
  have hio := leastMate_odd G hLMi
  have hjd' : (j : Int) + 1 ≤ (d : Int) := by exact_mod_cast hjd
  have hid' : (i : Int) + 1 ≤ (d : Int) := by exact_mod_cast hid
  simp only [hER] at hlo hhi h1
  omega

/-! ### The defender half: the game the engine defends -/

/-- **`ResistsFor G ch n q`.**  The inductive dual of `MatesWithin`,
with the two quantifiers swapped and the bound turned around.

In `MatesWithin G ch n p` the ATTACKER plays `ch`, the defender answers
with any legal move, and mate ARRIVES within `n` plies.  Here the
DEFENDER plays `ch`, the attacker answers with any legal move, and mate
does NOT arrive before `n` plies have been spent.  Mate lands on a
defender-to-move node in both, so the index counts plies from the same
places: `p` and `m` are attacker-to-move in `MatesWithin`, `q` and `m`
are defender-to-move here.

The leaves are dual too.  `MatesWithin.mate` is stated at every index
`n + 1`, which is the "at most" reading: once the mate has landed, any
remaining budget is met.  `zero` and `safe` are the mirror image, the
"at least" reading: while the mate has NOT landed, any budget of at
most one ply is met -- `zero` because nothing is claimed, `safe`
because one ply of survival is exactly "not mated now".

`draw` is the one constructor with no mirror image, and the asymmetry
is real rather than cosmetic.  `MatesWithin.step` carries `hnt` to stop
a moveless defender from satisfying its reply quantifier vacuously: a
stalemate is a draw and must not be counted as a mate.  That same
corner is a WIN for resistance -- a defender with no legal move who is
not in check is never mated at all -- so here it is a leaf that meets
every budget.  A guard on one side and a leaf on the other, for one
reason: a draw refutes the attacker's claim and establishes the
defender's.

What `ResistsFor` does NOT say, exactly as `MatesWithin` does not, is
that `ch q` is a legal move.  Both are statements about a play tree
relative to a given `ch`; legality is supplied where the tree is built,
by `NearMaximalChoice` (`ch q ∈ movesAbove ... ⊆ G.moves q`). -/
inductive ResistsFor (G : QSGame) (ch : G.Pos → G.Pos) : Nat → G.Pos → Prop where
  | zero {q : G.Pos} : ResistsFor G ch 0 q
  | safe {q : G.Pos} (hsafe : ¬ Checkmated G q) : ResistsFor G ch 1 q
  | draw {n : Nat} {q : G.Pos}
      (hterm : allIllegalB G q = true) (hsafe : ¬ Checkmated G q) :
      ResistsFor G ch n q
  | step {n : Nat} {q : G.Pos}
      (hsafe : ¬ Checkmated G q)
      (hrep : ∀ m ∈ G.moves (ch q),
        hasKingCapture G.toNullGame.toGame m = false → ResistsFor G ch n m) :
      ResistsFor G ch (n + 2) q

/-- Resistance is DOWNWARD closed where `matesWithin_mono` is upward:
a lower bound that holds for `n` plies holds for fewer.  The two leaves
are what make the odd indices reachable, so the "at least `n` plies"
reading is honest at every `n`, not only the even ones the theorem
produces. -/
theorem resistsFor_anti (G : QSGame) (ch : G.Pos → G.Pos) :
    ∀ {n : Nat} {q : G.Pos}, ResistsFor G ch n q →
      ∀ j, j ≤ n → ResistsFor G ch j q := by
  intro n q h
  induction h with
  | @zero q =>
    intro j hj
    obtain rfl : j = 0 := by omega
    exact ResistsFor.zero
  | @safe q hsafe =>
    intro j hj
    rcases j with _ | _ | j'
    · exact ResistsFor.zero
    · exact ResistsFor.safe hsafe
    · exact absurd hj (by omega)
  | @draw n q hterm hsafe =>
    intro j _
    exact ResistsFor.draw hterm hsafe
  | @step n q hsafe _hrep ih =>
    intro j hj
    rcases j with _ | _ | j'
    · exact ResistsFor.zero
    · exact ResistsFor.safe hsafe
    · exact ResistsFor.step hsafe (fun m hm hleg => ih m hm hleg j' (by omega))

/-- **A checkmated node resists for nothing.**  Every constructor but
`zero` refuses the mate that has already landed, so the index cannot be
read as vacuous: `ResistsFor G ch n q` at `n ≥ 1` really does deny that
mate is here. -/
theorem not_resistsFor_of_checkmated (G : QSGame) (ch : G.Pos → G.Pos)
    {n : Nat} {q : G.Pos} (hcm : Checkmated G q) (hn : 1 ≤ n) :
    ¬ ResistsFor G ch n q := by
  intro h
  cases h with
  | zero => omega
  | safe hsafe => exact hsafe hcm
  | draw _ hsafe => exact hsafe hcm
  | step hsafe _ => exact hsafe hcm

/-! ### The local step, in spec form -/

/-- **The engine's defence gives nothing away.**  If the move the
engine plays at `q` lets the attacker mate in `n` plies, then EVERY
legal move at `q` does: the position was already `ForcedlyMated G n q`.

This is `defence_maximal_resistance` restated where the recursion can
use it.  That one compares two exact distances through
`leastMate_value_block`; this one carries a `ForcedMate` to a
`ForcedlyMated` and so composes with itself.

Three lines of value, and a ply of parity.  `forcedMate_complete` puts
the chosen move's value at or above `MATE_LOWER + (d - n) * ER`.
Near-maximality is a MINIMUM over the reached values, so every
alternative -- and by `ValFloor` every legal move is admitted at this
depth -- sits at or above one rung lower, `MATE_LOWER + (d - n - 1) *
ER`.  `forcedMate_of_value_dist` reads that rung as a forced mate in
`n + 1`: one ply too many, the rung the driver's tolerance costs.
`forcedMate_odd_le` refunds it, because `n` is odd and so `n + 1` is an
EVEN budget, which is never tight.

Note what is NOT needed: nothing about `q`'s own value, its king, or
its terminal status beyond `hai` naming a legal move.  The argument is
entirely about the positions the engine can move to. -/
theorem defence_resistance_step (G : QSGame) (guard : G.Pos → Bool)
    (ch : G.Pos → G.Pos) (d : Nat)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G) (hZ : NoZugzwang G guard)
    (hch : NearMaximalChoice G guard d ch)
    {n : Nat} {q : G.Pos}
    (hai : allIllegalB G q = false)
    (hnodd : n % 2 = 1) (hnd : n + 1 ≤ d)
    (hspan : (d : Int) * EVAL_ROUGHNESS ≤ 21366)
    (hFM : ForcedMate G n (ch q)) :
    ForcedlyMated G n q := by
  have hML : MATE_LOWER = 47923 := rfl
  obtain ⟨_hmem, hnear⟩ := hch q hai
  have hval := forcedMate_complete G guard hF hZ hFM d hnd
  simp only [mateFloor, EVAL_ROUGHNESS] at hval
  simp only [EVAL_ROUGHNESS] at hspan
  have hvalv : MATE_LOWER + ((d : Int) - (n : Int)) * 15
      ≤ nullValueD2 G guard d (ch q) := by omega
  refine Or.inr ⟨hai, fun m hm hleg => ?_⟩
  have hmem' : m ∈ movesAbove G (val_lower (d + 1)) q :=
    mem_movesAbove_of_floor G hF (d := d + 1) (by omega) hm
  have hnearm := hnear m hmem'
  simp only [EVAL_ROUGHNESS] at hnearm
  have hbound : MATE_LOWER + (((d - n - 1 : Nat)) : Int) * EVAL_ROUGHNESS
      ≤ nullValueD2 G guard d m := by
    simp only [EVAL_ROUGHNESS]
    have hc : (((d - n - 1 : Nat)) : Int) = (d : Int) - (n : Int) - 1 := by omega
    rw [hc]
    omega
  obtain ⟨n', _hn'1, hn'2, hn'3⟩ :=
    forcedMate_of_value_dist G guard hF hQ hNM d (d - n - 1) m hleg hbound
  obtain ⟨n'', hn''le, hn''odd, hn''FM⟩ := forcedMate_odd_le G hn'3
  exact forcedMate_mono G hn''FM n (by omega)

/-! ### The strong recursion -/

/-- **The defender half's global induction.**  From a position no mate
can reach in fewer than `N` plies, the engine's own defence survives
`N` plies against EVERY attack.

The carrier is the negative spec statement `∀ i, ForcedlyMated G i q →
N ≤ i + 1` -- "no attacker budget below this one suffices" -- and not a
least distance, deliberately: it also covers the node the attacker has
already spoiled, where no forced mate exists at all and the quantifier
is vacuous.  The engine defends a position it has escaped as readily as
one it is still losing, and the induction should not need a case split
for that.

The recursion is `fuelMono`-shaped, descending two plies at a time.  At
`q` with budget `n + 2`: `hsafe` rules out the mate that has already
landed, a moveless `q` is the `draw` leaf, and otherwise every legal
attacker reply `m` to `ch q` inherits budget `n`.  What the child needs
is that no mate reaches IT in fewer than `n` plies, and that is two
plies of bookkeeping on the local step:

* `Checkmated m` makes `ch q` a mate in one (`ForcedMate.mate`);
* otherwise `ForcedlyMated G i m` makes it a mate in `i + 2`
  (`ForcedMate.step`, the constructor doing exactly what the play did);
* `forcedMate_odd_le` then `defence_resistance_step` turn either into
  `ForcedlyMated G j q` with `j` odd and `j ≤ i + 2`, and the carrier
  at `q` gives `N ≤ j + 1 ≤ i + 3` -- that is `n ≤ i + 1`, the child's
  carrier, with nothing to spare.

The `by_cases` on `n ≤ i + 1` is not a case analysis on the chess: for
a budget `i` already larger than the child needs there is nothing to
prove, and it is what keeps `i` small enough for the local step to fit
inside the horizon. -/
theorem defence_resists (G : QSGame) (guard : G.Pos → Bool)
    (ch : G.Pos → G.Pos) (d : Nat)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G) (hZ : NoZugzwang G guard)
    (hch : NearMaximalChoice G guard d ch)
    (hspan : (d : Int) * EVAL_ROUGHNESS ≤ 21366) :
    ∀ (N : Nat) (q : G.Pos),
      N ≤ d →
      hasKingCapture G.toNullGame.toGame q = false →
      ¬ Checkmated G q →
      (∀ i, ForcedlyMated G i q → N ≤ i + 1) →
      ResistsFor G ch N q := by
  intro N
  induction N using Nat.strongRecOn with
  | _ N ih =>
    intro q hNd hcapq hsafe hres
    rcases N with _ | _ | n
    · exact ResistsFor.zero
    · exact ResistsFor.safe hsafe
    · cases hai : allIllegalB G q with
      | true => exact ResistsFor.draw hai hsafe
      | false =>
        refine ResistsFor.step hsafe ?_
        intro m hm hleg
        -- the position the engine moved to is a real position with a king
        obtain ⟨hmem, _⟩ := hch q hai
        have hmq : ch q ∈ G.moves q := movesAbove_subset G _ q (ch q) hmem
        have hcap : ¬ (hasKingCapture G.toNullGame.toGame q = true) := by simp [hcapq]
        have hkgc : ¬ (G.eval (ch q) ≤ -MATE_LOWER) := fun hh =>
          hcap ((hasKingCapture_iff G.toNullGame.toGame q).mpr ⟨ch q, hmq, hh⟩)
        -- two plies of the play, read back as two plies of the spec
        have key : ∀ i, ForcedlyMated G i m → i + 3 ≤ d →
            ∃ j, j % 2 = 1 ∧ j ≤ i + 2 ∧ n + 2 ≤ j + 1 := by
          intro i hFL hid
          have hFMc : ForcedMate G (i + 2) (ch q) := by
            cases hFL with
            | inl hcm =>
              exact forcedMate_mono G
                (ForcedMate.mate (k := 0) hkgc hm hleg hcm) (i + 2) (by omega)
            | inr hrest => exact ForcedMate.step hkgc hm hleg hrest.1 hrest.2
          obtain ⟨j, hjle, hjodd, hjFM⟩ := forcedMate_odd_le G hFMc
          have hFLq := defence_resistance_step G guard ch d hF hQ hNM hZ hch hai
            hjodd (by omega) hspan hjFM
          exact ⟨j, hjodd, hjle, hres j hFLq⟩
        rcases n with _ | n'
        · exact ResistsFor.zero
        · refine ih (n' + 1) (by omega) m (by omega) hleg ?_ ?_
          · -- the child is not mated ALREADY: that would be a mate in two here
            intro hcm
            obtain ⟨j, hjodd, hjle, hjN⟩ := key 0 (Or.inl hcm) (by omega)
            omega
          · -- and no mate reaches it sooner than the budget it inherits
            intro i hFL
            by_cases hbig : n' + 1 ≤ i + 1
            · exact hbig
            · obtain ⟨j, hjodd, hjle, hjN⟩ := key i hFL (by omega)
              omega

/-- **The engine's defence is a LEGAL move**, wherever the claim needs
it to be.  `ResistsFor` does not demand this of `ch`, and neither does
`MatesWithin`; this is the fact they would demand, and it is a theorem
rather than a premise.

An illegal defence hands the attacker the king, and the model prices
that at the exact `MATE_UPPER` sentinel -- the largest value there is.
The engine MINIMISES the value it moves to, so an illegal `ch q` would
force every legal alternative to score within `EVAL_ROUGHNESS` of the
sentinel too.  That is the step where the terminal clamp becomes
quantitative: `hspan` caps the horizon at `d ≤ 1424` plies, so
`forcedMate_of_value_dist` at `t = 1423` leaves room for a forced mate
in ONE and nothing longer.  Every legal move at `q` would be mate in
one, `q` would be mated in two plies, and the carrier says it is not.

Three plies is exactly the range where this bites.  At a budget of two
the mate cannot land before ply two whatever the engine plays, so the
legality of the move is not part of the claim.  The recursion in
`defence_resists` hands each child these same hypotheses, so the same
theorem applies at every node of the play whose remaining budget is at
least three. -/
theorem defence_move_legal (G : QSGame) (guard : G.Pos → Bool)
    (ch : G.Pos → G.Pos) (d : Nat)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G)
    (hch : NearMaximalChoice G guard d ch)
    (hspan : (d : Int) * EVAL_ROUGHNESS ≤ 21366)
    {N : Nat} {q : G.Pos}
    (hai : allIllegalB G q = false)
    (hcapq : hasKingCapture G.toNullGame.toGame q = false)
    (hres : ∀ i, ForcedlyMated G i q → N ≤ i + 1)
    (hN : 3 ≤ N) (hNd : N ≤ d) :
    hasKingCapture G.toNullGame.toGame (ch q) = false := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  simp only [EVAL_ROUGHNESS] at hspan
  cases hc : hasKingCapture G.toNullGame.toGame (ch q) with
  | false => rfl
  | true =>
    exfalso
    obtain ⟨hmem, hnear⟩ := hch q hai
    have hmq : ch q ∈ G.moves q := movesAbove_subset G _ q (ch q) hmem
    have hcap : ¬ (hasKingCapture G.toNullGame.toGame q = true) := by simp [hcapq]
    have hkgc : ¬ (G.eval (ch q) ≤ -MATE_LOWER) := fun hh =>
      hcap ((hasKingCapture_iff G.toNullGame.toGame q).mpr ⟨ch q, hmq, hh⟩)
    have hvc : nullValueD2 G guard d (ch q) = MATE_UPPER :=
      nullValueD2_of_capture G guard d (ch q) hkgc hc
    have hFL : ForcedlyMated G 1 q := by
      refine Or.inr ⟨hai, fun m hm hleg => ?_⟩
      have hmem' : m ∈ movesAbove G (val_lower (d + 1)) q :=
        mem_movesAbove_of_floor G hF (d := d + 1) (by omega) hm
      have hnearm := hnear m hmem'
      simp only [EVAL_ROUGHNESS] at hnearm
      have hbound : MATE_LOWER + ((1423 : Nat) : Int) * EVAL_ROUGHNESS
          ≤ nullValueD2 G guard d m := by
        simp only [EVAL_ROUGHNESS]
        rw [hvc] at hnearm
        omega
      -- `hspan` caps the horizon at `d ≤ 1424`, so `n' ≤ 1`
      obtain ⟨n', _hn'1, hn'2, hn'3⟩ :=
        forcedMate_of_value_dist G guard hF hQ hNM d 1423 m hleg hbound
      exact forcedMate_mono G hn'3 1 (by omega)
    have := hres 1 hFL
    omega

/-- **DISTANCE-TO-MATE OPTIMALITY, defender half.**  From a root whose
EXACT distance to mate is `k + 1` plies -- `k` is the least attacker
budget the position permits, and `leastMated_odd_or_zero` makes `k + 1`
even -- the engine's own move choice, required only to be within the
driver's `EVAL_ROUGHNESS` stopping tolerance of the best, survives
`k + 1` plies against EVERY attack.

No defence whatsoever survives longer, so the engine attains the
game-theoretic optimum on the losing side too: "the losing side drags
the mate out as long as it can", proved rather than measured.

`k ≠ 0` excludes the position that is already checkmated, where there
is nothing left to defend and `ResistsFor G ch 0 q` is the whole
claim. -/
theorem leastMated_defence_resists (G : QSGame) (guard : G.Pos → Bool)
    (ch : G.Pos → G.Pos) (d : Nat)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G) (hZ : NoZugzwang G guard)
    (hch : NearMaximalChoice G guard d ch)
    {k : Nat} {q : G.Pos}
    (hLMd : LeastMated G k q) (hk : k ≠ 0) (hkd : k + 1 ≤ d)
    (hspan : (d : Int) * EVAL_ROUGHNESS ≤ 21366)
    (hcapq : hasKingCapture G.toNullGame.toGame q = false) :
    ResistsFor G ch (k + 1) q := by
  refine defence_resists G guard ch d hF hQ hNM hZ hch hspan (k + 1) q hkd hcapq ?_ ?_
  · exact fun hcm => hk (Nat.le_antisymm (hLMd.2 0 (Or.inl hcm)) (Nat.zero_le _))
  · exact fun i hFL => by have := hLMd.2 i hFL; omega

/-- **DISTANCE-TO-MATE OPTIMALITY, both directions.**  One engine, one
move rule, one tolerance: from a won root it mates in exactly the
number of plies the position permits, and from a lost root it survives
exactly the number of plies the position permits.

The two halves need the same premises and the same parity refund; they
differ only in which quantifier is the engine's.  The defender's depth
condition is one ply stricter (`k + 1 ≤ d` against `k + 1 ≤ d + 1`):
the attacker only has to FIND the mate it plays, the defender has to
outlast the faster mate that does not exist, and refuting it needs the
ply that would have shown it. -/
theorem dtm_optimal (G : QSGame) (guard : G.Pos → Bool)
    (ch : G.Pos → G.Pos) (d : Nat)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G) (hZ : NoZugzwang G guard)
    (hch : NearMaximalChoice G guard d ch)
    (hspan : (d : Int) * EVAL_ROUGHNESS ≤ 21366) :
    (∀ {k : Nat} {p : G.Pos}, LeastMate G k p → k + 1 ≤ d + 1 →
        hasKingCapture G.toNullGame.toGame p = false →
        MatesWithin G ch k p) ∧
      (∀ {k : Nat} {q : G.Pos}, LeastMated G k q → k ≠ 0 → k + 1 ≤ d →
        hasKingCapture G.toNullGame.toGame q = false →
        ResistsFor G ch (k + 1) q) :=
  ⟨fun hLM hkd hcapf =>
      leastMate_play_shortest G guard ch d hF hQ hNM hZ hch hLM hkd hspan hcapf,
   fun hLMd hk hkd hcapq =>
      leastMated_defence_resists G guard ch d hF hQ hNM hZ hch hLMd hk hkd hspan hcapq⟩

/-! ### The frontier premise is required HERE TOO, and not for the
reason one would guess

`Liveness.lean`'s `CexF` shows `NoMaskedMobility` is required for the
no-false-mates converse.  The natural hope is that the defender half
escapes it anyway: a phantom mate is invented at the QS frontier, where
almost no depth is left, so surely it can only claim the DEEPEST rungs
of the mate ladder -- and a real mate in `k` with `k + 1 <= d` claims a
higher one, so the phantom could never outrank the truth in the
engine's own ordering.

That hope is false, and the countermodel below says why in one number.
A masked node does not report a shallow mate.  Its filtered fold has no
admitted legal move left, so nothing ever displaces the initial
accumulator `LOSS = -MATE_UPPER`, and the node reports the exact
king-capture SENTINEL.  Negated at the parent that is `MATE_UPPER`,
which is not a rung at all: it is strictly above every value the mate
ladder can produce (`mateFloor_lt_MATE_UPPER`).  A phantom therefore
outranks EVERY real mate, at every distance and every depth, and the
rung machinery of this file cannot separate what is off the ladder.

`CexD` is that hope refuted at the exact hypotheses of
`leastMated_defence_resists`.  `Q` is genuinely lost in four plies; the
slow, correct defence `D` leads to a real mate in three, and the fast
loss `B` walks into mate in one.  One ply past the frontier of `D`'s
line sits `M`, a defender node whose only legal reply is filtered by
the depth-1 threshold while an illegal one survives it -- the position
class of the `#171` report, one ply deeper.  `D`'s subtree therefore
reports the sentinel instead of its true rung, and
`MATE_UPPER > MATE_LOWER + 3 * EVAL_ROUGHNESS` makes the correct
defence look WORSE than the mate in one.  Every near-maximal choice
must play `B`, and the engine is mated in two plies where the position
permitted four.

No acyclicity argument and no "eventually deep enough" argument can
retire this: `CexD` is a finite tree with no repetition anywhere in it,
and `d = 4` is as generous as the theorem allows.  The masking sits one
ply below the choice, and it stays one ply below the choice however
large `d` grows -- the frontier travels with the search.  What retires
the premise is the engine change: search the filtered tail before
declaring mate, and the sentinel never survives the fold. -/

inductive DPos where
  | Q | D | E | P | M | B | C | X | Z | S | W
  deriving DecidableEq

open DPos in
/-- `Q` (defender) → `{D, B}`; the slow defence `D → E → P` with `P`
mating on `C`, and the fast loss `B → C`.  `C` is checkmated: its only
move `X` is refuted by the recapture `Z`, and passing lets `W` take the
king.  `M` is the masked node -- `X` is illegal and admitted, `S` is
legal and filtered at the depth-1 threshold (`-150 < -100`), and `S` is
itself a stalemate. -/
def CexD : QSGame where
  Pos := DPos
  moves := fun p => match p with
    | Q => [D, B]
    | D => [E]
    | E => [P]
    | P => [M, C]
    | M => [X, S]
    | B => [C]
    | C => [X]
    | X => [Z]
    | W => [Z]
    | Z => []
    | S => []
  eval := fun p => match p with
    | Z => -MATE_UPPER
    | _ => 0
  pass := fun p => match p with
    | C => W
    | p => p
  val := fun p m => match p, m with
    | M, S => -150
    | _, _ => 0

instance : DecidableEq CexD.Pos := inferInstanceAs (DecidableEq DPos)

theorem cexD_floor : ValFloor CexD 192 := by
  intro p m _
  cases p <;> cases m <;> decide

theorem cexD_quiet : EvalQuiet CexD.toNullGame.toGame := by
  intro p
  cases p <;> decide

/-- The null option is never taken, so the fold's accumulator is the
bare `LOSS` sentinel -- which is the whole mechanism of the phantom. -/
theorem cexD_term (d : Nat) (p : CexD.Pos) :
    nullTermD2 CexD (fun _ => false) d p = LOSS := by
  simp only [nullTermD2]
  rw [if_neg (fun h => Bool.noConfusion h.1)]

theorem cexD_nozug : NoZugzwang CexD (fun _ => false) := by
  intro _ _ _ _ _ hg _
  exact Bool.noConfusion hg

/-! The declared values, bottom up. -/

theorem cexD_Z (d : Nat) :
    nullValueD2 CexD (fun _ => false) d DPos.Z = -MATE_UPPER :=
  nullValueD2_kingGone CexD _ d DPos.Z (by decide)

theorem cexD_X (d : Nat) :
    nullValueD2 CexD (fun _ => false) d DPos.X = MATE_UPPER :=
  nullValueD2_of_capture CexD _ d DPos.X (by decide) (by decide)

theorem cexD_S (d : Nat) :
    nullValueD2 CexD (fun _ => false) (d + 1) DPos.S = 0 := by
  rw [nullValueD2_of_allIllegal CexD _ d DPos.S (by decide) (by decide) (by decide)]
  simp only [terminalValue]
  rw [if_neg (by decide)]

theorem cexD_C (d : Nat) (hd : ((d : Int) + 1) * EVAL_ROUGHNESS ≤ 21366) :
    nullValueD2 CexD (fun _ => false) (d + 1) DPos.C
      = -MATE_LOWER - ((d : Int) + 1) * EVAL_ROUGHNESS := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  rw [nullValueD2_of_allIllegal CexD _ d DPos.C (by decide) (by decide) (by decide)]
  simp only [terminalValue, EVAL_ROUGHNESS] at hd ⊢
  rw [if_pos (by decide)]
  have hc : ((d + 1 : Nat) : Int) = (d : Int) + 1 := by omega
  rw [hc]
  omega

/-- **The phantom.**  `M`'s depth-1 fold admits only the illegal `X`
(the legal `S` is filtered at `val_lower 1 = -100`), so the accumulator
`LOSS` survives and the node reports the exact sentinel. -/
theorem cexD_M1 : nullValueD2 CexD (fun _ => false) 1 DPos.M = -MATE_UPPER := by
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hMU : MATE_UPPER = 69290 := rfl
  rw [nullValueD2_of_fold CexD _ 0 DPos.M (by decide) (by decide) (by decide)]
  have hma : movesAbove CexD (val_lower 1) DPos.M = [DPos.X] := by decide
  rw [hma, cexD_term]
  simp only [foldMax]
  rw [cexD_X]
  omega

/-- One ply up, the sentinel becomes a mate claim above every rung. -/
theorem cexD_P2 : nullValueD2 CexD (fun _ => false) 2 DPos.P = MATE_UPPER := by
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  rw [nullValueD2_of_fold CexD _ 1 DPos.P (by decide) (by decide) (by decide)]
  have hma : movesAbove CexD (val_lower 2) DPos.P = [DPos.M, DPos.C] := by decide
  rw [hma, cexD_term]
  simp only [foldMax]
  rw [cexD_M1, cexD_C 0 (by decide)]
  simp only [EVAL_ROUGHNESS]
  omega

theorem cexD_E3 : nullValueD2 CexD (fun _ => false) 3 DPos.E = -MATE_UPPER := by
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hMU : MATE_UPPER = 69290 := rfl
  rw [nullValueD2_of_fold CexD _ 2 DPos.E (by decide) (by decide) (by decide)]
  have hma : movesAbove CexD (val_lower 3) DPos.E = [DPos.P] := by decide
  rw [hma, cexD_term]
  simp only [foldMax]
  rw [cexD_P2]
  omega

/-- **The correct defence, mispriced.**  `D` is a real mate in three
and should be worth `MATE_LOWER + (4 - 3) * EVAL_ROUGHNESS = 47938`.
The phantom hands it `MATE_UPPER` instead. -/
theorem cexD_D4 : nullValueD2 CexD (fun _ => false) 4 DPos.D = MATE_UPPER := by
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hMU : MATE_UPPER = 69290 := rfl
  rw [nullValueD2_of_fold CexD _ 3 DPos.D (by decide) (by decide) (by decide)]
  have hma : movesAbove CexD (val_lower 4) DPos.D = [DPos.E] := by decide
  rw [hma, cexD_term]
  simp only [foldMax]
  rw [cexD_E3]
  omega

/-- The fast loss is priced honestly: a real mate in one, three rungs
above the floor -- and 21322 points BELOW the phantom. -/
theorem cexD_B4 : nullValueD2 CexD (fun _ => false) 4 DPos.B = 47968 := by
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  rw [nullValueD2_of_fold CexD _ 3 DPos.B (by decide) (by decide) (by decide)]
  have hma : movesAbove CexD (val_lower 4) DPos.B = [DPos.C] := by decide
  rw [hma, cexD_term]
  simp only [foldMax]
  rw [cexD_C 2 (by decide)]
  simp only [EVAL_ROUGHNESS]
  omega

/-- With four plies to spend the filter hides nothing and `M` is the
honest draw it always was.  The masking lives at the frontier only --
which is exactly why no amount of depth searches it away: the frontier
travels with the search. -/
theorem cexD_M4 : nullValueD2 CexD (fun _ => false) 4 DPos.M = 0 := by
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hMU : MATE_UPPER = 69290 := rfl
  rw [nullValueD2_of_fold CexD _ 3 DPos.M (by decide) (by decide) (by decide)]
  have hma : movesAbove CexD (val_lower 4) DPos.M = [DPos.X, DPos.S] := by decide
  rw [hma, cexD_term]
  simp only [foldMax]
  rw [cexD_X, cexD_S 2]
  omega

/-! The engine's move rule, and what it is forced to play. -/

open DPos in
/-- The value-minimising choice at every node with a legal move.  At
`Q` it is forced: `47968 < MATE_UPPER`, and near-maximality's tolerance
of one `EVAL_ROUGHNESS` cannot bridge 21322 points. -/
def chD : CexD.Pos → CexD.Pos := fun p => match p with
  | Q => B
  | D => E
  | E => P
  | P => C
  | M => S
  | B => C
  | C => X
  | X => Z
  | W => Z
  | Z => Z
  | S => S

/-- The choice is not merely near-maximal, it is the exact argmin. -/
theorem cexD_maximal : MaximalChoice CexD (fun _ => false) 4 chD := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  intro p hai
  cases p with
  | Q =>
    refine ⟨by decide, fun m hm => ?_⟩
    have hma : movesAbove CexD (val_lower 5) DPos.Q = [DPos.D, DPos.B] := by decide
    rw [hma] at hm
    show nullValueD2 CexD (fun _ => false) 4 DPos.B
      ≤ nullValueD2 CexD (fun _ => false) 4 m
    rw [cexD_B4]
    rcases List.mem_cons.mp hm with rfl | hm2
    · rw [cexD_D4]; omega
    · rw [List.mem_singleton.mp hm2, cexD_B4]; omega
  | P =>
    refine ⟨by decide, fun m hm => ?_⟩
    have hma : movesAbove CexD (val_lower 5) DPos.P = [DPos.M, DPos.C] := by decide
    rw [hma] at hm
    show nullValueD2 CexD (fun _ => false) 4 DPos.C
      ≤ nullValueD2 CexD (fun _ => false) 4 m
    rw [cexD_C 3 (by decide)]
    simp only [EVAL_ROUGHNESS]
    rcases List.mem_cons.mp hm with rfl | hm2
    · rw [cexD_M4]; omega
    · rw [List.mem_singleton.mp hm2, cexD_C 3 (by decide)]
      simp only [EVAL_ROUGHNESS]
      omega
  | M =>
    refine ⟨by decide, fun m hm => ?_⟩
    have hma : movesAbove CexD (val_lower 5) DPos.M = [DPos.X, DPos.S] := by decide
    rw [hma] at hm
    show nullValueD2 CexD (fun _ => false) 4 DPos.S
      ≤ nullValueD2 CexD (fun _ => false) 4 m
    rw [cexD_S 3]
    rcases List.mem_cons.mp hm with rfl | hm2
    · rw [cexD_X]; omega
    · rw [List.mem_singleton.mp hm2, cexD_S 3]; omega
  | D =>
    refine ⟨by decide, fun m hm => ?_⟩
    have hma : movesAbove CexD (val_lower 5) DPos.D = [DPos.E] := by decide
    rw [hma] at hm
    rw [List.mem_singleton.mp hm]
    show nullValueD2 CexD (fun _ => false) 4 DPos.E
      ≤ nullValueD2 CexD (fun _ => false) 4 DPos.E
    omega
  | E =>
    refine ⟨by decide, fun m hm => ?_⟩
    have hma : movesAbove CexD (val_lower 5) DPos.E = [DPos.P] := by decide
    rw [hma] at hm
    rw [List.mem_singleton.mp hm]
    show nullValueD2 CexD (fun _ => false) 4 DPos.P
      ≤ nullValueD2 CexD (fun _ => false) 4 DPos.P
    omega
  | B =>
    refine ⟨by decide, fun m hm => ?_⟩
    have hma : movesAbove CexD (val_lower 5) DPos.B = [DPos.C] := by decide
    rw [hma] at hm
    rw [List.mem_singleton.mp hm]
    show nullValueD2 CexD (fun _ => false) 4 DPos.C
      ≤ nullValueD2 CexD (fun _ => false) 4 DPos.C
    omega
  | X =>
    refine ⟨by decide, fun m hm => ?_⟩
    have hma : movesAbove CexD (val_lower 5) DPos.X = [DPos.Z] := by decide
    rw [hma] at hm
    rw [List.mem_singleton.mp hm]
    show nullValueD2 CexD (fun _ => false) 4 DPos.Z
      ≤ nullValueD2 CexD (fun _ => false) 4 DPos.Z
    omega
  | W =>
    refine ⟨by decide, fun m hm => ?_⟩
    have hma : movesAbove CexD (val_lower 5) DPos.W = [DPos.Z] := by decide
    rw [hma] at hm
    rw [List.mem_singleton.mp hm]
    show nullValueD2 CexD (fun _ => false) 4 DPos.Z
      ≤ nullValueD2 CexD (fun _ => false) 4 DPos.Z
    omega
  | C => exact absurd hai (by decide)
  | Z => exact absurd hai (by decide)
  | S => exact absurd hai (by decide)

/-- ... and therefore under the driver's real, tolerant rule as well.
The refutation does not lean on the `EVAL_ROUGHNESS` slack at all: it
already defeats the exactly-converged bisection that `MaximalChoice`
idealises. -/
theorem cexD_nearMax : NearMaximalChoice CexD (fun _ => false) 4 chD :=
  nearMaximalChoice_of_maximal CexD (fun _ => false) 4 chD cexD_maximal

/-! The chess facts of the countermodel, from the spec side. -/

theorem cexD_C_mated : Checkmated CexD DPos.C := ⟨by decide, by decide⟩

theorem cexD_B_mate : ForcedMate CexD 1 DPos.B :=
  ForcedMate.mate (k := 0) (by decide) (by decide) (by decide) cexD_C_mated

theorem cexD_P_mate : ForcedMate CexD 1 DPos.P :=
  ForcedMate.mate (k := 0) (by decide) (by decide) (by decide) cexD_C_mated

/-- The slow defence really is a mate in three: `D → E`, and `E`'s only
legal reply `P` mates in one. -/
theorem cexD_D_mate3 : ForcedMate CexD 3 DPos.D := by
  have hrep : ∀ m' ∈ CexD.moves DPos.E,
      hasKingCapture CexD.toNullGame.toGame m' = false → ForcedMate CexD 1 m' := by
    intro m' hm' _
    rw [List.mem_singleton.mp hm']
    exact cexD_P_mate
  exact ForcedMate.step (k := 1) (by decide)
    (show DPos.E ∈ CexD.moves DPos.D by decide) (by decide) (by decide) hrep

/-- ... and no faster: `E` is not checkmated, and a two-ply budget
would need its reply mated in zero. -/
theorem cexD_D_not_fast : ∀ j, j ≤ 2 → ¬ ForcedMate CexD j DPos.D := by
  intro j hj h
  cases h with
  | @mate k p m _ hm _ hmate =>
    rw [List.mem_singleton.mp hm] at hmate
    exact absurd hmate.1 (by decide)
  | @step k p m _ hm _ _ hrep =>
    rw [List.mem_singleton.mp hm] at hrep
    obtain rfl : k = 0 := by omega
    exact not_forcedMate_zero CexD DPos.P (hrep DPos.P (by decide) (by decide))

/-- `Q`'s exact distance: mated with attacker budget three, so four
plies of resistance are what the position permits. -/
theorem cexD_leastMated : LeastMated CexD 3 DPos.Q := by
  constructor
  · refine Or.inr ⟨by decide, fun m hm _ => ?_⟩
    rcases List.mem_cons.mp hm with rfl | hm2
    · exact cexD_D_mate3
    · rw [List.mem_singleton.mp hm2]
      exact forcedMate_mono CexD cexD_B_mate 3 (by omega)
  · intro j hj
    cases hj with
    | inl hcm => exact absurd hcm.1 (by decide)
    | inr hrest =>
      by_cases hle : 3 ≤ j
      · exact hle
      · exact absurd (hrest.2 DPos.D (by decide) (by decide))
          (cexD_D_not_fast j (by omega))

/-- **The engine is mated in two where the position permitted four.**
Every near-maximal choice plays `B`, whose only legal reply is the
checkmated `C`. -/
theorem cexD_not_resists : ¬ ResistsFor CexD chD 4 DPos.Q := by
  intro h
  cases h with
  | draw hterm _ => exact absurd hterm (by decide)
  | step _ hrep =>
    exact not_resistsFor_of_checkmated CexD chD cexD_C_mated (by omega)
      (hrep DPos.C (by decide) (by decide))

theorem cexD_masked_mobility : ¬ NoMaskedMobility CexD := by
  intro h
  exact absurd (h DPos.M (by decide) DPos.S (by decide)) (by decide)

/-- **The finding, bundled: `NoMaskedMobility` cannot be dropped from
the defender half.**  Every other premise of
`leastMated_defence_resists` holds -- the fidelity premises, the
zugzwang premise, the engine's move rule at `d = 4` in its IDEALISED
exact-argmax form (so the tolerance is not what breaks it), the exact
distance `LeastMated CexD 3 Q` with `3 + 1 ≤ 4`, root legality, and the
span condition -- and the conclusion `ResistsFor CexD chD 4 Q` is
FALSE.  The one premise that fails is `NoMaskedMobility`, at `M`.

The theorem without it is therefore not merely unproven but false, and
false for a reason no rung argument can repair, because the phantom's
value is not on the ladder.  The attacker half goes the same way under
the mirror construction: a phantom sibling valued `MATE_UPPER`
outranks a real mate in one at every depth. -/
theorem cexD_defence_needs_frontier :
    ValFloor CexD 192 ∧ EvalQuiet CexD.toNullGame.toGame ∧
      NoZugzwang CexD (fun _ => false) ∧
      MaximalChoice CexD (fun _ => false) 4 chD ∧
      NearMaximalChoice CexD (fun _ => false) 4 chD ∧
      LeastMated CexD 3 DPos.Q ∧
      hasKingCapture CexD.toNullGame.toGame DPos.Q = false ∧
      ¬ NoMaskedMobility CexD ∧
      ¬ ResistsFor CexD chD 4 DPos.Q :=
  ⟨cexD_floor, cexD_quiet, cexD_nozug, cexD_maximal, cexD_nearMax,
   cexD_leastMated, by decide, cexD_masked_mobility, cexD_not_resists⟩

end Sunfish
