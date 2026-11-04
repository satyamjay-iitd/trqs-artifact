@[ext]
structure Interval where
  start : Nat
  stop  : Nat
  valid : start ≤ stop


def Interval.equals(i1 i2: Interval) : Prop :=
  i1.start = i2.start ∧ i1.stop = i2.stop

def Interval.size (i : Interval) : Nat :=
  i.stop - i.start + 1


def Interval.includes (i1 i2 : Interval) : Prop :=
  i1.start ≤ i2.start ∧ i2.stop ≤ i1.stop
