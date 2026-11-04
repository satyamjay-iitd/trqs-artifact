import Intervalset.Interval
import Intervalset.SplitIntervalSet

/-!
# The `STARTS` operation

`STARTS` keeps the *first* interval when both begin together and the second
ends no later than the first:

  `[a₁, b₁] STARTS [a₂, b₂] = [a₁, b₁]`  if `a₁ = a₂ ∧ b₂ ≤ b₁`, else `EMPTY`.

The enumeration semantics on interval-sets is all non-empty pairwise results:

  `A STARTS B = { I₁ STARTS I₂ | I₁ ∈ A, I₂ ∈ B, ≠ EMPTY }`.

## Derivation of the compact formulas

An interval `i = [a, b]` is in `A STARTS B` iff `i ∈ A` and some `[a, b₂] ∈ B`
has `b₂ ≤ b`.  So `a ∈ [b.s, b.l]` and

* `b₂ ∈ [max(b.r, a + b.c − 1, a), min(b.e, b)]`.

This range is non-empty iff `b.r ≤ b.e`, `a + b.c − 1 ≤ b.e`, `a ≤ b.e`,
`b.r ≤ b`, and `b.c ≤ b − a + 1` (`a ≤ b` holds by validity).  Intersecting
with `i ∈ A` yields:

* `s' = max(a.s, b.s)`,                             `e' = a.e`
* `l' = min(a.l, b.l, b.e, b.e + 1 − b.c)`
* `r' = max(a.r, b.r)`
* `c' = max(a.c, b.c)`

with global feasibility `b.r ≤ b.e`.  Unlike `EQUAL` (the field-wise
intersection), the second operand only bounds the *start* of the result: its
stop may extend past `b.e`, so `e' = a.e`, while `b.e` caps the start instead.
`b.e + 1 − b.c` cannot underflow since `b.c ≤ b.e − b.s + 1`.
-/

/-- `STARTS` on two single intervals: keep the first iff both start together and
the second ends no later.  The result is the first interval. -/
def Interval.startOp (i1 i2 : Interval) : Option Interval :=
  if i1.start = i2.start ∧ i2.stop ≤ i1.stop then some i1 else none

/-- `STARTS` on the compact `SplitIntervalSet` representation. -/
def SplitIntervalSet.startOp (a b : SplitIntervalSet) : Option SplitIntervalSet :=
  if b.r ≤ b.e then
    let s' := max a.s b.s
    let l' := min a.l (min b.l (min b.e (b.e + 1 - b.c)))
    let r' := max a.r b.r
    let e' := a.e
    let c' := max a.c b.c
    if hv : s' ≤ e' then
      if hb : c' ≤ e' - s' + 1 then
        some ⟨s', l', r', e', c', hv, hb⟩
      else none
    else none
  else none

theorem startOp_spec (a b : SplitIntervalSet) (i : Interval) :
    (∃ sis, a.startOp b = some sis ∧ sis.contains i) ↔
    (∃ i1 i2, a.contains i1 ∧ b.contains i2 ∧ i1.startOp i2 = some i) := by
  constructor
  -- Compact ⟶ enumeration: pick the shortest `[a, b₂] ∈ B` sharing `i`'s start.
  · rintro ⟨sis, hsis, hc⟩
    simp only [SplitIntervalSet.startOp] at hsis
    by_cases hg : b.r ≤ b.e
    · rw [if_pos hg] at hsis
      by_cases hv : max a.s b.s ≤ a.e
      · rw [dif_pos hv] at hsis
        by_cases hb' : max a.c b.c ≤ a.e - max a.s b.s + 1
        · rw [dif_pos hb'] at hsis
          simp only [Option.some.injEq] at hsis
          rw [← hsis] at hc
          simp only [SplitIntervalSet.contains, Interval.size] at hc
          obtain ⟨hc1, hc2, hc3, hc4, hc5⟩ := hc
          have hiv := i.valid
          have hbb := b.size_bound
          generalize hb2 : max b.r (max (i.start + b.c - 1) i.start) = b2
          refine ⟨i, { start := i.start, stop := b2, valid := by omega }, ?_, ?_, ?_⟩
          · simp only [SplitIntervalSet.contains, Interval.size]; omega
          · simp only [SplitIntervalSet.contains, Interval.size]; omega
          · rw [Interval.startOp, if_pos (show i.start = i.start ∧ b2 ≤ i.stop by omega)]
        · rw [dif_neg hb'] at hsis; exact absurd hsis (by simp)
      · rw [dif_neg hv] at hsis; exact absurd hsis (by simp)
    · rw [if_neg hg] at hsis; exact absurd hsis (by simp)
  -- Enumeration ⟶ compact: the shared start lies in both start ranges.
  · rintro ⟨i1, i2, hA, hB, hov⟩
    simp only [SplitIntervalSet.contains, Interval.size] at hA hB
    obtain ⟨hA1, hA2, hA3, hA4, hA5⟩ := hA
    obtain ⟨hB1, hB2, hB3, hB4, hB5⟩ := hB
    simp only [Interval.startOp] at hov
    by_cases hov' : i1.start = i2.start ∧ i2.stop ≤ i1.stop
    · rw [if_pos hov'] at hov
      simp only [Option.some.injEq] at hov
      subst i1
      obtain ⟨hov1, hov2⟩ := hov'
      have hiv := i.valid
      have hi2v := i2.valid
      have hg : b.r ≤ b.e := by omega
      have hv : max a.s b.s ≤ a.e := by omega
      have hb' : max a.c b.c ≤ a.e - max a.s b.s + 1 := by omega
      refine ⟨⟨max a.s b.s, min a.l (min b.l (min b.e (b.e + 1 - b.c))),
                max a.r b.r, a.e, max a.c b.c, hv, hb'⟩, ?_, ?_⟩
      · simp only [SplitIntervalSet.startOp]; rw [if_pos hg, dif_pos hv, dif_pos hb']
      · simp only [SplitIntervalSet.contains, Interval.size]; omega
    · rw [if_neg hov'] at hov; exact absurd hov (by simp)
