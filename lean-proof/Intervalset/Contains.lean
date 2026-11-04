import Intervalset.Interval
import Intervalset.SplitIntervalSet

/-!
# The `CONTAINS` operation (Allen's *contains*)

`CONTAINS` keeps the *first* interval when the second lies strictly inside it
(starts strictly later and ends strictly earlier):

  `[a₁, b₁] CONTAINS [a₂, b₂] = [a₁, b₁]`  if `a₁ < a₂ ∧ b₂ < b₁`, else `EMPTY`.

This is exactly Allen's *contains* (`di`) relation.  The enumeration semantics on
interval-sets is all non-empty pairwise results:

  `A CONTAINS B = { I₁ CONTAINS I₂ | I₁ ∈ A, I₂ ∈ B, ≠ EMPTY }`.

## Derivation of the compact formulas

An interval `i = [a, b]` is in `A CONTAINS B` iff `i ∈ A` and some interval
`[a₂, b₂] ∈ B` sits strictly inside, `a < a₂ ≤ b₂ < b`.  Here

* `a₂ ∈ [max(b.s, a + 1), b.l]`,
* `b₂ ∈ [b.r, min(b.e, b − 1)]`,

linked by `b₂ ≥ a₂ + b.c − 1`.  Such a pair exists iff both ranges are non-empty
and the smallest `a₂` admits a fitting `b₂`.  Expanding and intersecting with
`i ∈ A` (assuming `1 ≤ b.c`) yields:

* `s' = a.s`,                                       `e' = a.e`
* `l' = min(a.l, b.l − 1, b.e − b.c)`
* `r' = max(a.r, b.r + 1, b.s + b.c)`
* `c' = max(a.c, b.c + 2)`

with global feasibility `b.s ≤ b.l`, `b.r ≤ b.e`, `a.s + 1 ≤ b.l`, `b.c ≤ b.e`.
The `b.c + 2` size floor reflects that `i` must strictly enclose an interval of
size `≥ b.c`; the last two guards also keep `b.l − 1` and `b.e − b.c` from
underflowing in `ℕ`.
-/

/-- `CONTAINS` on two single intervals (Allen *contains*): keep the first iff the
second lies strictly inside it.  The result is the first interval. -/
def Interval.containsOp (i1 i2 : Interval) : Option Interval :=
  if i1.start < i2.start ∧ i2.stop < i1.stop then some i1 else none

/-- `CONTAINS` on the compact `SplitIntervalSet` representation. -/
def SplitIntervalSet.containsOp (a b : SplitIntervalSet) : Option SplitIntervalSet :=
  if b.s ≤ b.l ∧ b.r ≤ b.e ∧ a.s + 1 ≤ b.l ∧ b.c ≤ b.e then
    let s' := a.s
    let l' := min a.l (min (b.l - 1) (b.e - b.c))
    let r' := max a.r (max (b.r + 1) (b.s + b.c))
    let e' := a.e
    let c' := max a.c (b.c + 2)
    if hv : s' ≤ e' then
      if hb : c' ≤ e' - s' + 1 then
        some ⟨s', l', r', e', c', hv, hb⟩
      else none
    else none
  else none

theorem containsOp_spec (a b : SplitIntervalSet) (hb : 1 ≤ b.c) (i : Interval) :
    (∃ sis, a.containsOp b = some sis ∧ sis.contains i) ↔
    (∃ i1 i2, a.contains i1 ∧ b.contains i2 ∧ i1.containsOp i2 = some i) := by
  constructor
  -- Compact ⟶ enumeration: place a strictly-inside interval `[a₂, b₂] ∈ B`.
  · rintro ⟨sis, hsis, hc⟩
    simp only [SplitIntervalSet.containsOp] at hsis
    by_cases hg : b.s ≤ b.l ∧ b.r ≤ b.e ∧ a.s + 1 ≤ b.l ∧ b.c ≤ b.e
    · rw [if_pos hg] at hsis
      by_cases hv : a.s ≤ a.e
      · rw [dif_pos hv] at hsis
        by_cases hb' : max a.c (b.c + 2) ≤ a.e - a.s + 1
        · rw [dif_pos hb'] at hsis
          simp only [Option.some.injEq] at hsis
          rw [← hsis] at hc
          simp only [SplitIntervalSet.contains, Interval.size] at hc
          obtain ⟨hc1, hc2, hc3, hc4, hc5⟩ := hc
          obtain ⟨hg1, hg2, hg3, hg4⟩ := hg
          have hiv := i.valid
          have hbv := b.valid; have hbb := b.size_bound
          -- flatten the box bounds, then drop them so `omega` splits only on `a2`, `b2`
          have hl1 : i.start ≤ a.l := by omega
          have hl2 : i.start + 1 ≤ b.l := by omega
          have hl3 : i.start + b.c ≤ b.e := by omega
          have hr1 : a.r ≤ i.stop := by omega
          have hr2 : b.r + 1 ≤ i.stop := by omega
          have hr3 : b.s + b.c ≤ i.stop := by omega
          have hz1 : a.c ≤ i.stop - i.start + 1 := by omega
          have hz2 : i.start + b.c + 1 ≤ i.stop := by omega
          clear hc2 hc3 hc5
          generalize ha2 : max b.s (i.start + 1) = a2
          generalize hb2 : max b.r (a2 + b.c - 1) = b2
          refine ⟨i, { start := a2, stop := b2, valid := by omega }, ?_, ?_, ?_⟩
          · simp only [SplitIntervalSet.contains, Interval.size]; omega
          · simp only [SplitIntervalSet.contains, Interval.size]; omega
          · simp only [Interval.containsOp]
            rw [if_pos (show i.start < a2 ∧ b2 < i.stop by omega)]
        · rw [dif_neg hb'] at hsis; exact absurd hsis (by simp)
      · rw [dif_neg hv] at hsis; exact absurd hsis (by simp)
    · rw [if_neg hg] at hsis; exact absurd hsis (by simp)
  -- Enumeration ⟶ compact: the strict inclusion pins the start cap and stop floor.
  · rintro ⟨i1, i2, hA, hB, hov⟩
    simp only [SplitIntervalSet.contains, Interval.size] at hA hB
    obtain ⟨hA1, hA2, hA3, hA4, hA5⟩ := hA
    obtain ⟨hB1, hB2, hB3, hB4, hB5⟩ := hB
    simp only [Interval.containsOp] at hov
    by_cases hov' : i1.start < i2.start ∧ i2.stop < i1.stop
    · rw [if_pos hov'] at hov
      simp only [Option.some.injEq] at hov
      subst i1
      obtain ⟨hov1, hov2⟩ := hov'
      have hi2v := i2.valid
      have hav := a.valid; have hab := a.size_bound
      have hg : b.s ≤ b.l ∧ b.r ≤ b.e ∧ a.s + 1 ≤ b.l ∧ b.c ≤ b.e :=
        ⟨by omega, by omega, by omega, by omega⟩
      have hv : a.s ≤ a.e := a.valid
      have hb' : max a.c (b.c + 2) ≤ a.e - a.s + 1 := by omega
      refine ⟨⟨a.s, min a.l (min (b.l - 1) (b.e - b.c)),
                max a.r (max (b.r + 1) (b.s + b.c)), a.e,
                max a.c (b.c + 2), hv, hb'⟩, ?_, ?_⟩
      · simp only [SplitIntervalSet.containsOp]; rw [if_pos hg, dif_pos hv, dif_pos hb']
      · simp only [SplitIntervalSet.contains, Interval.size]; omega
    · rw [if_neg hov'] at hov; exact absurd hov (by simp)
