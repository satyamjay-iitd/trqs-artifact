import Intervalset.Interval
import Intervalset.SplitIntervalSet

/-!
# The `EQUAL` operation

`EQUAL` keeps an interval only when it occurs identically in both operands:

  `[a₁, b₁] EQUAL [a₂, b₂] = [a₁, b₁]`  if `a₁ = a₂ ∧ b₁ = b₂`, else `EMPTY`.

The enumeration semantics on interval-sets is all non-empty pairwise results:

  `A EQUAL B = { I₁ EQUAL I₂ | I₁ ∈ A, I₂ ∈ B, ≠ EMPTY }`.

Since `I₁ EQUAL I₂` is non-empty exactly when `I₁ = I₂`, this is the set of
intervals common to both, i.e. `⟦A⟧ ∩ ⟦B⟧`.  On the compact representation that
intersection is computed field-wise:

  `A EQUAL B = ⟨max(a.s,b.s), min(a.l,b.l), max(a.r,b.r), min(a.e,b.e), max(a.c,b.c)⟩`

(guarded by the usual well-formedness conditions).  This operation coincides
with set intersection (`And`/`inter`). -/

/-- `EQUAL` on two single intervals: keep the first iff the two are identical.
The result is that common interval. -/
def Interval.equalsOp (i1 i2 : Interval) : Option Interval :=
  if i2.start = i1.start ∧ i2.stop = i1.stop then some i1 else none

/-- `EQUAL` on the compact `SplitIntervalSet` representation: the field-wise
intersection. -/
def SplitIntervalSet.equalsOp (a b : SplitIntervalSet) : Option SplitIntervalSet :=
  let s' := max a.s b.s
  let l' := min a.l b.l
  let r' := max a.r b.r
  let e' := min a.e b.e
  let c' := max a.c b.c
  if hv : s' ≤ e' then
    if hb : c' ≤ e' - s' + 1 then
      some ⟨s', l', r', e', c', hv, hb⟩
    else none
  else none

theorem equalsOp_spec (a b : SplitIntervalSet) (i : Interval) :
    (∃ sis, a.equalsOp b = some sis ∧ sis.contains i) ↔
    (∃ i1 i2, a.contains i1 ∧ b.contains i2 ∧ i1.equalsOp i2 = some i) := by
  constructor
  -- Compact ⟶ enumeration: the same interval `i` witnesses both operands.
  · rintro ⟨sis, hsis, hc⟩
    simp only [SplitIntervalSet.equalsOp] at hsis
    by_cases hv : max a.s b.s ≤ min a.e b.e
    · rw [dif_pos hv] at hsis
      by_cases hb : max a.c b.c ≤ min a.e b.e - max a.s b.s + 1
      · rw [dif_pos hb] at hsis
        simp only [Option.some.injEq] at hsis
        rw [← hsis] at hc
        simp only [SplitIntervalSet.contains, Interval.size] at hc
        obtain ⟨h1, h2, h3, h4, h5⟩ := hc
        refine ⟨i, i, ?_, ?_, ?_⟩
        · simp only [SplitIntervalSet.contains, Interval.size]; omega
        · simp only [SplitIntervalSet.contains, Interval.size]; omega
        · simp [Interval.equalsOp]
      · rw [dif_neg hb] at hsis; exact absurd hsis (by simp)
    · rw [dif_neg hv] at hsis; exact absurd hsis (by simp)
  -- Enumeration ⟶ compact: equality forces `i ∈ A` and `i ∈ B`.
  · rintro ⟨i1, i2, hA, hB, hov⟩
    simp only [SplitIntervalSet.contains, Interval.size] at hA hB
    obtain ⟨hA1, hA2, hA3, hA4, hA5⟩ := hA
    obtain ⟨hB1, hB2, hB3, hB4, hB5⟩ := hB
    simp only [Interval.equalsOp] at hov
    by_cases hcond : i2.start = i1.start ∧ i2.stop = i1.stop
    · rw [if_pos hcond] at hov
      simp only [Option.some.injEq] at hov
      subst i1
      obtain ⟨hcs, hce⟩ := hcond
      have hiv := i.valid
      have hv : max a.s b.s ≤ min a.e b.e := by omega
      have hbound : max a.c b.c ≤ min a.e b.e - max a.s b.s + 1 := by omega
      refine ⟨⟨max a.s b.s, min a.l b.l, max a.r b.r, min a.e b.e, max a.c b.c, hv, hbound⟩,
              ?_, ?_⟩
      · simp only [SplitIntervalSet.equalsOp]; rw [dif_pos hv, dif_pos hbound]
      · simp only [SplitIntervalSet.contains, Interval.size]; omega
    · rw [if_neg hcond] at hov; exact absurd hov (by simp)
