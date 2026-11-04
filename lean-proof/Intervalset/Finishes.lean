import Intervalset.Interval
import Intervalset.SplitIntervalSet

/-!
# The `FINISHES` operation

`FINISHES` keeps the *first* interval when both end together and the second
starts no earlier than the first:

  `[a₁, b₁] FINISHES [a₂, b₂] = [a₁, b₁]`  if `b₁ = b₂ ∧ a₁ ≤ a₂`, else `EMPTY`.

The enumeration semantics on interval-sets is all non-empty pairwise results:

  `A FINISHES B = { I₁ FINISHES I₂ | I₁ ∈ A, I₂ ∈ B, ≠ EMPTY }`.

This is the mirror of `STARTS` under reversing the time axis: there the operands
share a start and the second ends no later, here they share a stop and the
second begins no earlier.

## Derivation of the compact formulas

An interval `i = [a, b]` is in `A FINISHES B` iff `i ∈ A` and some `[a₂, b] ∈ B`
has `a ≤ a₂`.  So `b ∈ [b.r, b.e]` and

* `a₂ ∈ [max(b.s, a), min(b.l, b − b.c + 1, b)]`.

This range is non-empty iff `b.s ≤ b.l`, `b.s + b.c − 1 ≤ b`, `b.s ≤ b`,
`a ≤ b.l`, and `b.c ≤ b − a + 1` (`a ≤ b` holds by validity).  Intersecting
with `i ∈ A` yields:

* `s' = a.s`,                                       `e' = min(a.e, b.e)`
* `l' = min(a.l, b.l)`
* `r' = max(a.r, b.r, b.s, b.s + b.c − 1)`
* `c' = max(a.c, b.c)`

with global feasibility `b.s ≤ b.l`.  Unlike `EQUALS` (the field-wise
intersection), the second operand only bounds the *stop* of the result: its
start may reach back before `b.s`, so `s' = a.s`, while `b.s` caps the stop
instead.  Both `b.s` and `b.s + b.c − 1` are needed in `r'`: the latter is the
weaker bound exactly when `b.c = 0`.
-/

/-- `FINISHES` on two single intervals: keep the first iff both end together and
the second starts no earlier.  The result is the first interval. -/
def Interval.finishOp (i1 i2 : Interval) : Option Interval :=
  if i1.stop = i2.stop ∧ i1.start ≤ i2.start then some i1 else none

/-- `FINISHES` on the compact `SplitIntervalSet` representation. -/
def SplitIntervalSet.finishOp (a b : SplitIntervalSet) : Option SplitIntervalSet :=
  if b.s ≤ b.l then
    let s' := a.s
    let l' := min a.l b.l
    let r' := max a.r (max b.r (max b.s (b.s + b.c - 1)))
    let e' := min a.e b.e
    let c' := max a.c b.c
    if hv : s' ≤ e' then
      if hb : c' ≤ e' - s' + 1 then
        some ⟨s', l', r', e', c', hv, hb⟩
      else none
    else none
  else none

theorem finishOp_spec (a b : SplitIntervalSet) (i : Interval) :
    (∃ sis, a.finishOp b = some sis ∧ sis.contains i) ↔
    (∃ i1 i2, a.contains i1 ∧ b.contains i2 ∧ i1.finishOp i2 = some i) := by
  constructor
  -- Compact ⟶ enumeration: pick the latest-starting `[a₂, b] ∈ B` sharing `i`'s stop.
  · rintro ⟨sis, hsis, hc⟩
    simp only [SplitIntervalSet.finishOp] at hsis
    by_cases hg : b.s ≤ b.l
    · rw [if_pos hg] at hsis
      by_cases hv : a.s ≤ min a.e b.e
      · rw [dif_pos hv] at hsis
        by_cases hb' : max a.c b.c ≤ min a.e b.e - a.s + 1
        · rw [dif_pos hb'] at hsis
          simp only [Option.some.injEq] at hsis
          rw [← hsis] at hc
          simp only [SplitIntervalSet.contains, Interval.size] at hc
          obtain ⟨hc1, hc2, hc3, hc4, hc5⟩ := hc
          have hiv := i.valid
          have hbb := b.size_bound
          generalize ha2 : max b.s i.start = a2
          refine ⟨i, { start := a2, stop := i.stop, valid := by omega }, ?_, ?_, ?_⟩
          · simp only [SplitIntervalSet.contains, Interval.size]; omega
          · simp only [SplitIntervalSet.contains, Interval.size]; omega
          · rw [Interval.finishOp, if_pos (show i.stop = i.stop ∧ i.start ≤ a2 by omega)]
        · rw [dif_neg hb'] at hsis; exact absurd hsis (by simp)
      · rw [dif_neg hv] at hsis; exact absurd hsis (by simp)
    · rw [if_neg hg] at hsis; exact absurd hsis (by simp)
  -- Enumeration ⟶ compact: the shared stop lies in both stop ranges.
  · rintro ⟨i1, i2, hA, hB, hov⟩
    simp only [SplitIntervalSet.contains, Interval.size] at hA hB
    obtain ⟨hA1, hA2, hA3, hA4, hA5⟩ := hA
    obtain ⟨hB1, hB2, hB3, hB4, hB5⟩ := hB
    simp only [Interval.finishOp] at hov
    by_cases hov' : i1.stop = i2.stop ∧ i1.start ≤ i2.start
    · rw [if_pos hov'] at hov
      simp only [Option.some.injEq] at hov
      subst i1
      obtain ⟨hov1, hov2⟩ := hov'
      have hiv := i.valid
      have hi2v := i2.valid
      have hg : b.s ≤ b.l := by omega
      have hv : a.s ≤ min a.e b.e := by omega
      have hb' : max a.c b.c ≤ min a.e b.e - a.s + 1 := by omega
      refine ⟨⟨a.s, min a.l b.l,
                max a.r (max b.r (max b.s (b.s + b.c - 1))), min a.e b.e,
                max a.c b.c, hv, hb'⟩, ?_, ?_⟩
      · simp only [SplitIntervalSet.finishOp]; rw [if_pos hg, dif_pos hv, dif_pos hb']
      · simp only [SplitIntervalSet.contains, Interval.size]; omega
    · rw [if_neg hov'] at hov; exact absurd hov (by simp)
