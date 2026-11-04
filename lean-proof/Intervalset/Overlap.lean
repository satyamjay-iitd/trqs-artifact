import Intervalset.Interval
import Intervalset.SplitIntervalSet

/-!
# The `OVERLAP` operation (Allen's *overlaps*)

`OVERLAP` merges two properly overlapping intervals: the second starts strictly
after the first starts, overlaps it, and ends strictly after it.  The result
spans from the first interval's start to the second interval's stop:

  `[a₁, b₁] OVERLAP [a₂, b₂] = [a₁, b₂]`  if `a₁ < a₂ ≤ b₁ < b₂`, else `EMPTY`.

This is exactly Allen's *overlaps* relation.  The enumeration semantics on
interval-sets is all non-empty pairwise results:

  `A OVERLAP B = { I₁ OVERLAP I₂ | I₁ ∈ A, I₂ ∈ B, ≠ EMPTY }`.

## Derivation of the compact formulas

Membership of `[a, b]` asks for a first stop `b₁` and a second start `a₂` with
`[a, b₁] ∈ A`, `[a₂, b] ∈ B`, and `a < a₂ ≤ b₁ < b`.  Here

* `b₁ ∈ [max(a.r, a + a.c − 1), min(a.e, b − 1)]`,
* `a₂ ∈ [max(b.s, a + 1), min(b.l, b − b.c + 1)]`,

linked by `a₂ ≤ b₁`.  Such a pair exists iff both ranges are non-empty and the
smallest `a₂` does not exceed the largest `b₁`.  Expanding and projecting onto
`(a, b)` (assuming `1 ≤ a.c`, `1 ≤ b.c`) yields:

* `s' = a.s`,                                       `e' = b.e`
* `l' = min(a.l, a.e + 1 − a.c, a.e − 1, b.l − 1)`
* `r' = max(b.r, a.r + 1, b.s + b.c − 1, b.s + 1)`
* `c' = max(a.c + 1, b.c + 1, 3)`

with global feasibility `a.r ≤ a.e`, `b.s ≤ a.e`, `b.s ≤ b.l`, `a.s + 1 ≤ a.e`,
`a.s + 1 ≤ b.l`.  The minimum size `3` reflects that the smallest proper overlap
`a < a+1 ≤ a+1 < a+2` already spans three frames; the last two guards also keep
`a.e − 1` and `b.l − 1` from underflowing in `ℕ`.
-/

/-- `OVERLAP` on two single intervals (Allen *overlaps*): when the second starts
within the first but strictly later, and ends strictly later, return the merged
span `[i1.start, i2.stop]`. -/
def Interval.overlapOp (i1 i2 : Interval) : Option Interval :=
  if h : i1.start < i2.start ∧ i2.start ≤ i1.stop ∧ i1.stop < i2.stop then
    some { start := i1.start, stop := i2.stop,
           valid := by obtain ⟨_, _, _⟩ := h; have := i1.valid; have := i2.valid; omega }
  else
    none

/-- `OVERLAP` on the compact `SplitIntervalSet` representation. -/
def SplitIntervalSet.overlapOp (a b : SplitIntervalSet) : Option SplitIntervalSet :=
  if a.r ≤ a.e ∧ b.s ≤ a.e ∧ b.s ≤ b.l ∧ a.s + 1 ≤ a.e ∧ a.s + 1 ≤ b.l then
    let s' := a.s
    let l' := min a.l (min (a.e + 1 - a.c) (min (a.e - 1) (b.l - 1)))
    let r' := max b.r (max (a.r + 1) (max (b.s + b.c - 1) (b.s + 1)))
    let e' := b.e
    let c' := max (a.c + 1) (max (b.c + 1) 3)
    if hv : s' ≤ e' then
      if hb : c' ≤ e' - s' + 1 then
        some ⟨s', l', r', e', c', hv, hb⟩
      else none
    else none
  else none

set_option maxHeartbeats 400000 in
theorem overlapOp_spec (a b : SplitIntervalSet) (ha : 1 ≤ a.c) (hb : 1 ≤ b.c)
    (i : Interval) :
    (∃ sis, a.overlapOp b = some sis ∧ sis.contains i) ↔
    (∃ i1 i2, a.contains i1 ∧ b.contains i2 ∧ i1.overlapOp i2 = some i) := by
  constructor
  -- Compact ⟶ enumeration: recover a first stop `b₁` and a second start `a₂`.
  · rintro ⟨sis, hsis, hc⟩
    simp only [SplitIntervalSet.overlapOp] at hsis
    by_cases hg : a.r ≤ a.e ∧ b.s ≤ a.e ∧ b.s ≤ b.l ∧ a.s + 1 ≤ a.e ∧ a.s + 1 ≤ b.l
    · rw [if_pos hg] at hsis
      by_cases hv : a.s ≤ b.e
      · rw [dif_pos hv] at hsis
        by_cases hb' : max (a.c + 1) (max (b.c + 1) 3) ≤ b.e - a.s + 1
        · rw [dif_pos hb'] at hsis
          simp only [Option.some.injEq] at hsis
          rw [← hsis] at hc
          simp only [SplitIntervalSet.contains, Interval.size] at hc
          obtain ⟨hc1, hc2, hc3, hc4, hc5⟩ := hc
          obtain ⟨hg1, hg2, hg3, hg4, hg5⟩ := hg
          have hiv := i.valid
          have hav := a.valid; have hab := a.size_bound
          have hbv := b.valid; have hbb := b.size_bound
          -- flatten the nested `min`/`max` of the output box into linear facts,
          -- then drop them so the witness goals split only on `a2`, `b1`
          have hl1 : i.start ≤ a.l := by omega
          have hl2 : i.start + a.c - 1 ≤ a.e := by omega
          have hl3 : i.start + 1 ≤ a.e := by omega
          have hl4 : i.start + 1 ≤ b.l := by omega
          have hr1 : b.r ≤ i.stop := by omega
          have hr2 : a.r + 1 ≤ i.stop := by omega
          have hr3 : b.s + b.c - 1 ≤ i.stop := by omega
          have hr4 : b.s + 1 ≤ i.stop := by omega
          have hz1 : i.start + a.c ≤ i.stop := by omega
          have hz2 : i.start + b.c ≤ i.stop := by omega
          have hz3 : i.start + 2 ≤ i.stop := by omega
          clear hc2 hc3 hc5
          generalize ha2 : max b.s (i.start + 1) = a2
          generalize hb1 : max a2 (max a.r (i.start + a.c - 1)) = b1
          refine ⟨{ start := i.start, stop := b1, valid := by omega },
                  { start := a2, stop := i.stop, valid := by omega }, ?_, ?_, ?_⟩
          · simp only [SplitIntervalSet.contains, Interval.size]; omega
          · simp only [SplitIntervalSet.contains, Interval.size]; omega
          · unfold Interval.overlapOp
            rw [dif_pos (show i.start < a2 ∧ a2 ≤ b1 ∧ b1 < i.stop by omega)]
        · rw [dif_neg hb'] at hsis; exact absurd hsis (by simp)
      · rw [dif_neg hv] at hsis; exact absurd hsis (by simp)
    · rw [if_neg hg] at hsis; exact absurd hsis (by simp)
  -- Enumeration ⟶ compact: the proper overlap pins every output bound.
  · rintro ⟨i1, i2, hA, hB, hov⟩
    simp only [SplitIntervalSet.contains, Interval.size] at hA hB
    obtain ⟨hA1, hA2, hA3, hA4, hA5⟩ := hA
    obtain ⟨hB1, hB2, hB3, hB4, hB5⟩ := hB
    unfold Interval.overlapOp at hov
    by_cases hov' : i1.start < i2.start ∧ i2.start ≤ i1.stop ∧ i1.stop < i2.stop
    · rw [dif_pos hov'] at hov
      simp only [Option.some.injEq] at hov
      obtain ⟨hov1, hov2, hov3⟩ := hov'
      have hi1v := i1.valid
      have hi2v := i2.valid
      have hstart : i.start = i1.start := by rw [← hov]
      have hstop  : i.stop  = i2.stop  := by rw [← hov]
      have hg : a.r ≤ a.e ∧ b.s ≤ a.e ∧ b.s ≤ b.l ∧ a.s + 1 ≤ a.e ∧ a.s + 1 ≤ b.l :=
        ⟨by omega, by omega, by omega, by omega, by omega⟩
      have hv : a.s ≤ b.e := by omega
      have hb' : max (a.c + 1) (max (b.c + 1) 3) ≤ b.e - a.s + 1 := by omega
      refine ⟨⟨a.s, min a.l (min (a.e + 1 - a.c) (min (a.e - 1) (b.l - 1))),
                max b.r (max (a.r + 1) (max (b.s + b.c - 1) (b.s + 1))), b.e,
                max (a.c + 1) (max (b.c + 1) 3), hv, hb'⟩, ?_, ?_⟩
      · simp only [SplitIntervalSet.overlapOp]; rw [if_pos hg, dif_pos hv, dif_pos hb']
      · simp only [SplitIntervalSet.contains, Interval.size]; omega
    · rw [dif_neg hov'] at hov; exact absurd hov (by simp)
