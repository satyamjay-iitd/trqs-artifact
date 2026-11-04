import Intervalset.Interval
import Intervalset.SplitIntervalSet
import Intervalset.Then

/-!
# The `THEN[gmin, gmax]` (sequential concatenation with a gap) operation

This generalises `THEN` (see `Then.lean`) by allowing a *gap* of `g` frames
between the two intervals, where `gmin ≤ g ≤ gmax`:

  `(a₁, b₁) THEN[gmin,gmax] (a₂, b₂) = (a₁, b₂)`
      if  `b₁ + 1 + gmin ≤ a₂ ≤ b₁ + 1 + gmax`, else `EMPTY`.

(The gap-free `THEN` is the special case `gmin = gmax = 0`.)  The result spans
both intervals *and* the gap, so its size is `size₁ + size₂ + g`; the smallest
possible gap `gmin` therefore enters the output size bound.

We prove the compact representation operation agrees, interval-by-interval,
with the enumeration semantics.  As in `Then.lean` we assume `1 ≤ a.c`,
`1 ≤ b.c` (harmless: every interval has size `≥ 1`), and additionally
`gmin ≤ gmax`.

## Derivation of the compact formulas

Membership of `[a, b]` in `A THEN[gmin,gmax] B` asks for a first stop `m` and a
gap `g` with

* `[a, m] ∈ A`,
* `[m+1+g, b] ∈ B`,
* `gmin ≤ g ≤ gmax`.

`m` ranges over `[max(a.r, a + a.c − 1), a.e]` and `g` over `[gmin, gmax]`
independently, and every second-interval constraint depends only on the sum
`t = m + g`.  Hence such `m, g` exist iff the `m`-range is non-empty *and* the
achievable range of `t`, `[max(a.r, a+a.c−1) + gmin, a.e + gmax]`, meets the
required range `[b.s − 1, min(b.l − 1, b − b.c)]`.  Expanding those conditions
and projecting onto `(a, b)` yields:

* `s' = a.s`,                                       `e' = b.e`
* `l' = min(a.l, a.e + 1 − a.c, b.l − a.c − gmin)`
* `r' = max(b.r, a.r + b.c + gmin, b.s + b.c − 1)`
* `c' = a.c + b.c + gmin`

with global feasibility `a.r ≤ a.e`, `max(a.r + gmin, b.s − 1) ≤ min(a.e + gmax, b.l − 1)`.

Note that the gap does **not** loosen the start bound `a.e + 1 − a.c`: the first
stop `m` is capped by `a.e` regardless of how large the gap is.  The minimum
gap `gmin` enters the size bound (`c' = a.c + b.c + gmin`) and the `b`-side
bounds, which is precisely the `g_min` term of the original formula.
-/

/-- `THEN[gmin, gmax]` on two single intervals: defined iff the second starts
within `[gmin, gmax]` frames after the first ends. -/
def Interval.thenGapOp (gmin gmax : Nat) (i1 i2 : Interval) : Option Interval :=
  if h : i1.stop + 1 + gmin ≤ i2.start ∧ i2.start ≤ i1.stop + 1 + gmax then
    some { start := i1.start, stop := i2.stop,
           valid := by have := i1.valid; have := i2.valid; omega }
  else
    none

/-- `THEN[gmin, gmax]` on the compact `SplitIntervalSet` representation.

The leading conjunction is the global feasibility guard: the first stop must be
realisable (`a.r ≤ a.e`) and the split-sum ranges must meet
(`max(a.r + gmin, b.s − 1) ≤ min(a.e + gmax, b.l − 1)`), written without
truncated subtraction.  The final `a.c + gmin ≤ b.l` keeps the
`b.l - a.c - gmin` term from being silently floored to `0` in `ℕ`. -/
def SplitIntervalSet.thenGapOp (gmin gmax : Nat) (a b : SplitIntervalSet) :
    Option SplitIntervalSet :=
  if a.r ≤ a.e ∧ a.r + gmin + 1 ≤ b.l ∧
     b.s ≤ a.e + gmax + 1 ∧ b.s ≤ b.l ∧ a.c + gmin ≤ b.l then
    let s' := a.s
    let l' := min a.l (min (a.e + 1 - a.c) (b.l - a.c - gmin))
    let r' := max b.r (max (a.r + b.c + gmin) (b.s + b.c - 1))
    let e' := b.e
    let c' := a.c + b.c + gmin
    if hv : s' ≤ e' then
      if hb : c' ≤ e' - s' + 1 then
        some ⟨s', l', r', e', c', hv, hb⟩
      else none
    else none
  else none

set_option maxHeartbeats 400000 in
theorem thenGapOp_spec (gmin gmax : Nat) (hgg : gmin ≤ gmax)
    (a b : SplitIntervalSet) (ha : 1 ≤ a.c) (hb : 1 ≤ b.c) (i : Interval) :
    (∃ sis, a.thenGapOp gmin gmax b = some sis ∧ sis.contains i) ↔
    (∃ i1 i2, a.contains i1 ∧ b.contains i2 ∧ i1.thenGapOp gmin gmax i2 = some i) := by
  constructor
  -- Compact ⟶ enumeration: recover a split point `m` and a gap `g`.
  · rintro ⟨sis, hsis, hc⟩
    simp only [SplitIntervalSet.thenGapOp] at hsis
    by_cases hgd : a.r ≤ a.e ∧ a.r + gmin + 1 ≤ b.l ∧
        b.s ≤ a.e + gmax + 1 ∧ b.s ≤ b.l ∧ a.c + gmin ≤ b.l
    · rw [if_pos hgd] at hsis
      by_cases hv : a.s ≤ b.e
      · rw [dif_pos hv] at hsis
        by_cases hb' : a.c + b.c + gmin ≤ b.e - a.s + 1
        · rw [dif_pos hb'] at hsis
          simp only [Option.some.injEq] at hsis
          rw [← hsis] at hc
          simp only [SplitIntervalSet.contains, Interval.size] at hc
          obtain ⟨hc1, hc2, hc3, hc4, hc5⟩ := hc
          obtain ⟨hgd1, hgd2, hgd3, hgd4, hgd5⟩ := hgd
          have hiv := i.valid
          have hav := a.valid; have hab := a.size_bound
          have hbv := b.valid; have hbb := b.size_bound
          -- `p` = least valid first stop; `t = m + g`; `q` = recovered first stop.
          -- Resolve the nested `max`es once into linear facts, then discard them so
          -- the remaining `omega` calls need not re-split on them.
          generalize hp : max a.r (i.start + a.c - 1) = p
          generalize ht : max (p + gmin) (b.s - 1) = t
          generalize hq : max p (t - gmax) = q
          have key :
              (i.start ≤ q ∧ a.r ≤ q ∧ q ≤ a.e ∧ i.start + a.c - 1 ≤ q) ∧
              (b.s ≤ t + 1 ∧ t + 1 ≤ b.l ∧ t + b.c ≤ i.stop ∧ t + 1 ≤ i.stop) ∧
              (q + gmin ≤ t ∧ t ≤ q + gmax) := by omega
          clear hp ht hq
          obtain ⟨⟨hk1, hk2, hk3, hk4⟩, ⟨hk5, hk6, hk7, hk8⟩, ⟨hk9, hk10⟩⟩ := key
          refine ⟨{ start := i.start, stop := q, valid := by omega },
                  { start := t + 1, stop := i.stop, valid := by omega }, ?_, ?_, ?_⟩
          · simp only [SplitIntervalSet.contains, Interval.size]; omega
          · simp only [SplitIntervalSet.contains, Interval.size]; omega
          · unfold Interval.thenGapOp
            rw [dif_pos (show q + 1 + gmin ≤ t + 1 ∧ t + 1 ≤ q + 1 + gmax by omega)]
        · rw [dif_neg hb'] at hsis; exact absurd hsis (by simp)
      · rw [dif_neg hv] at hsis; exact absurd hsis (by simp)
    · rw [if_neg hgd] at hsis; exact absurd hsis (by simp)
  -- Enumeration ⟶ compact: the two halves and the gap force every constraint.
  · rintro ⟨i1, i2, hA, hB, hthen⟩
    simp only [SplitIntervalSet.contains, Interval.size] at hA hB
    obtain ⟨hA1, hA2, hA3, hA4, hA5⟩ := hA
    obtain ⟨hB1, hB2, hB3, hB4, hB5⟩ := hB
    unfold Interval.thenGapOp at hthen
    by_cases hadj : i1.stop + 1 + gmin ≤ i2.start ∧ i2.start ≤ i1.stop + 1 + gmax
    · rw [dif_pos hadj] at hthen
      simp only [Option.some.injEq] at hthen
      obtain ⟨hadj1, hadj2⟩ := hadj
      have hi1v := i1.valid
      have hi2v := i2.valid
      have hstart : i.start = i1.start := by rw [← hthen]
      have hstop  : i.stop  = i2.stop  := by rw [← hthen]
      have hgd : a.r ≤ a.e ∧ a.r + gmin + 1 ≤ b.l ∧
          b.s ≤ a.e + gmax + 1 ∧ b.s ≤ b.l ∧ a.c + gmin ≤ b.l :=
        ⟨by omega, by omega, by omega, by omega, by omega⟩
      have hv : a.s ≤ b.e := by omega
      have hb' : a.c + b.c + gmin ≤ b.e - a.s + 1 := by omega
      refine ⟨⟨a.s, min a.l (min (a.e + 1 - a.c) (b.l - a.c - gmin)),
                max b.r (max (a.r + b.c + gmin) (b.s + b.c - 1)), b.e,
                a.c + b.c + gmin, hv, hb'⟩, ?_, ?_⟩
      · simp only [SplitIntervalSet.thenGapOp]
        rw [if_pos hgd, dif_pos hv, dif_pos hb']
      · simp only [SplitIntervalSet.contains, Interval.size]; omega
    · rw [dif_neg hadj] at hthen; exact absurd hthen (by simp)

/-! ## Specialisation: `THEN[0, 0]` is the gap-free `THEN`

With `gmin = gmax = 0` every `+ 0` / `- 0` reduces away, so the gap operations
coincide definitionally (split-interval-set level) or up to the trivial guard
rephrasing `0 ≤ g ≤ 0 ↔ g = 0` (interval level) with the gap-free `THEN` of
`Then.lean`. -/

/-- `THEN[0,0]` on single intervals coincides with the gap-free `THEN`. -/
theorem Interval.thenGapOp_zero_zero (i1 i2 : Interval) :
    i1.thenGapOp 0 0 i2 = i1.thenOp i2 := by
  unfold Interval.thenGapOp Interval.thenOp
  by_cases h : i2.start = i1.stop + 1
  · rw [dif_pos h,
        dif_pos (show i1.stop + 1 + 0 ≤ i2.start ∧ i2.start ≤ i1.stop + 1 + 0 by omega)]
  · rw [dif_neg h,
        dif_neg (show ¬(i1.stop + 1 + 0 ≤ i2.start ∧ i2.start ≤ i1.stop + 1 + 0) by omega)]

/-- `THEN[0,0]` on the compact representation coincides with the gap-free `THEN`. -/
theorem SplitIntervalSet.thenGapOp_zero_zero (a b : SplitIntervalSet) :
    a.thenGapOp 0 0 b = a.thenOp b := by
  rfl
