import Intervalset.Interval
import Intervalset.SplitIntervalSet

/-!
# The `THEN` (sequential concatenation) operation

`THEN` on two single intervals concatenates them when the second starts
immediately after the first ends:

  `(a₁, b₁) THEN (a₂, b₂) = (a₁, b₂)`  if `a₂ = b₁ + 1`, else `EMPTY`.

The *enumeration* semantics of `THEN` on two interval-sets `A`, `B` is the
set of all non-empty `i₁ THEN i₂` with `i₁ ∈ A`, `i₂ ∈ B` (all pairs).

The *compact* representation `(s, l, r, e, c)` stands for every interval
that starts in `[s, l]`, ends in `[r, e]`, and has size `≥ c`
(`SplitIntervalSet.contains`).

This file defines `THEN` directly on the compact representation and proves it
agrees, interval-by-interval, with the enumeration semantics.

We work in the "simple" (gap-free) setting: the two intervals must be exactly
adjacent.  We assume `1 ≤ a.c` and `1 ≤ b.c`; this is no loss of generality
because every interval has size `≥ 1`, so the constraints `c = 0` and `c = 1`
describe the same set.
-/

/-- `THEN` on two single intervals: defined iff the second is adjacent to the
first, in which case the result spans both. -/
def Interval.thenOp (i1 i2 : Interval) : Option Interval :=
  if h : i2.start = i1.stop + 1 then
    some { start := i1.start, stop := i2.stop,
           valid := by have := i1.valid; have := i2.valid; omega }
  else
    none

/-- `THEN` on the compact `SplitIntervalSet` representation.

The output start/stop boxes and the size bound are:

* `s' = a.s`,                                     `e' = b.e`
* `l' = min(a.l, a.e + 1 - a.c, b.l - a.c)`       (tightest start upper bound)
* `r' = max(b.r, a.r + b.c, b.s + b.c - 1)`       (tightest stop lower bound)
* `c' = a.c + b.c`

The leading conjunction is the *global feasibility* guard: there must exist a
valid split point `m` at all, i.e. `max(a.r, b.s − 1) ≤ min(a.e, b.l − 1)`,
written here without truncated subtraction.  The extra `a.c ≤ b.l` keeps the
`b.l - a.c` term from being silently floored to `0` in `ℕ`. -/
def SplitIntervalSet.thenOp (a b : SplitIntervalSet) : Option SplitIntervalSet :=
  if a.r ≤ a.e ∧ a.r + 1 ≤ b.l ∧ b.s ≤ a.e + 1 ∧ b.s ≤ b.l ∧ a.c ≤ b.l then
    let s' := a.s
    let l' := min a.l (min (a.e + 1 - a.c) (b.l - a.c))
    let r' := max b.r (max (a.r + b.c) (b.s + b.c - 1))
    let e' := b.e
    let c' := a.c + b.c
    if hv : s' ≤ e' then
      if hb : c' ≤ e' - s' + 1 then
        some ⟨s', l', r', e', c', hv, hb⟩
      else none
    else none
  else none

theorem thenOp_spec (a b : SplitIntervalSet) (ha : 1 ≤ a.c) (hb : 1 ≤ b.c)
    (i : Interval) :
    (∃ sis, a.thenOp b = some sis ∧ sis.contains i) ↔
    (∃ i1 i2, a.contains i1 ∧ b.contains i2 ∧ i1.thenOp i2 = some i) := by
  constructor
  -- Compact ⟶ enumeration
  · rintro ⟨sis, hsis, hc⟩
    simp only [SplitIntervalSet.thenOp] at hsis
    by_cases hg : a.r ≤ a.e ∧ a.r + 1 ≤ b.l ∧ b.s ≤ a.e + 1 ∧ b.s ≤ b.l ∧ a.c ≤ b.l
    · rw [if_pos hg] at hsis
      by_cases hv : a.s ≤ b.e
      · rw [dif_pos hv] at hsis
        by_cases hb' : a.c + b.c ≤ b.e - a.s + 1
        · rw [dif_pos hb'] at hsis
          simp only [Option.some.injEq] at hsis
          rw [← hsis] at hc
          simp only [SplitIntervalSet.contains, Interval.size] at hc
          obtain ⟨hc1, hc2, hc3, hc4, hc5⟩ := hc
          obtain ⟨hg1, hg2, hg3, hg4, hg5⟩ := hg
          have hiv := i.valid
          have hav := a.valid; have hab := a.size_bound
          have hbv := b.valid; have hbb := b.size_bound
          -- split point: the smallest valid stop of the first interval
          refine ⟨{ start := i.start, stop := max a.r (max (i.start + a.c - 1) (b.s - 1)),
                    valid := by omega },
                  { start := max a.r (max (i.start + a.c - 1) (b.s - 1)) + 1, stop := i.stop,
                    valid := by omega }, ?_, ?_, ?_⟩
          · simp only [SplitIntervalSet.contains, Interval.size]; omega
          · simp only [SplitIntervalSet.contains, Interval.size]; omega
          · unfold Interval.thenOp
            rw [dif_pos rfl]
        · rw [dif_neg hb'] at hsis; exact absurd hsis (by simp)
      · rw [dif_neg hv] at hsis; exact absurd hsis (by simp)
    · rw [if_neg hg] at hsis; exact absurd hsis (by simp)
  -- Enumeration ⟶ compact
  · rintro ⟨i1, i2, hA, hB, hthen⟩
    simp only [SplitIntervalSet.contains, Interval.size] at hA hB
    obtain ⟨hA1, hA2, hA3, hA4, hA5⟩ := hA
    obtain ⟨hB1, hB2, hB3, hB4, hB5⟩ := hB
    unfold Interval.thenOp at hthen
    by_cases hadj : i2.start = i1.stop + 1
    · rw [dif_pos hadj] at hthen
      simp only [Option.some.injEq] at hthen
      have hi1v := i1.valid
      have hi2v := i2.valid
      -- read off `i.start = i1.start`, `i.stop = i2.stop`
      have hstart : i.start = i1.start := by rw [← hthen]
      have hstop  : i.stop  = i2.stop  := by rw [← hthen]
      have hg : a.r ≤ a.e ∧ a.r + 1 ≤ b.l ∧ b.s ≤ a.e + 1 ∧ b.s ≤ b.l ∧ a.c ≤ b.l :=
        ⟨by omega, by omega, by omega, by omega, by omega⟩
      have hv : a.s ≤ b.e := by omega
      have hb' : a.c + b.c ≤ b.e - a.s + 1 := by omega
      refine ⟨⟨a.s, min a.l (min (a.e + 1 - a.c) (b.l - a.c)),
                max b.r (max (a.r + b.c) (b.s + b.c - 1)), b.e, a.c + b.c, hv, hb'⟩, ?_, ?_⟩
      · simp only [SplitIntervalSet.thenOp]
        rw [if_pos hg, dif_pos hv, dif_pos hb']
      · simp only [SplitIntervalSet.contains, Interval.size]; omega
    · rw [dif_neg hadj] at hthen; exact absurd hthen (by simp)
