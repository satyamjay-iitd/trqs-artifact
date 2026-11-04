import Intervalset.Interval

@[ext]
structure SplitIntervalSet where
  s          : Nat
  l          : Nat
  r          : Nat
  e          : Nat
  c          : Nat
  valid      : s ≤ e
  size_bound : c ≤ e - s + 1

def SplitIntervalSet.contains (sis : SplitIntervalSet) (i : Interval) : Prop :=
  sis.s ≤ i.start ∧ i.start ≤ sis.l ∧
  sis.r ≤ i.stop  ∧ i.stop  ≤ sis.e ∧
  sis.c ≤ i.size

