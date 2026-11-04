# Lean proofs for the TrQS interval-set algebra

Machine-checked equivalence between the compact interval-set formulas and
their pairwise specifications. Checking these proofs requires no datasets,
Rust build, Python environment, or GPU.

## Checking proofs

You need Git and **Elan**, the Lean toolchain manager, with its `lean` and `lake`
commands on your PATH. Lake is Lean's package manager and build tool. If you do
not have Elan, follow the installation instructions in the
[Elan repository](https://github.com/leanprover/elan#installation).

From the root of the repo, run:

```sh
cd lean-proof
lean --version
lake exe cache get
lake build Intervalset Intervalset.Contains Intervalset.Equal \
  Intervalset.Finishes Intervalset.Starts Intervalset.Overlap \
  Intervalset.Then Intervalset.ThenWithGap
```

Elan selects **Lean 4.30.0-rc1** from `lean-toolchain`; `lake-manifest.json`
pins mathlib and its dependencies. The first run downloads the toolchain,
dependencies, and compiled cache. Keep both pin files unchanged.

**Use the full build command above:** `lake build` alone checks only the basic
definitions. Success is exit status zero. To check one operator, use e.g.
`lake build Intervalset.Contains`.

## Operators and source files

In the table, the first input is `[a₁, b₁]` and the second is `[a₂, b₂]`.
A pair that fails its condition contributes no result.

| Operation | Condition on a pair | Returned interval | Source and main theorem |
|---|---|---|---|
| `EQUAL` | `a₁ = a₂` and `b₁ = b₂` | First interval | [Equal.lean](Intervalset/Equal.lean), `equalsOp_spec` |
| `CONTAINS` | `a₁ < a₂` and `b₂ < b₁` | First interval | [Contains.lean](Intervalset/Contains.lean), `containsOp_spec` |
| `STARTS` | `a₁ = a₂` and `b₂ ≤ b₁` | First interval | [Starts.lean](Intervalset/Starts.lean), `startOp_spec` |
| `FINISHES` | `b₁ = b₂` and `a₁ ≤ a₂` | First interval | [Finishes.lean](Intervalset/Finishes.lean), `finishOp_spec` |
| `OVERLAP` | `a₁ < a₂ ≤ b₁ < b₂` | `[a₁, b₂]` | [Overlap.lean](Intervalset/Overlap.lean), `overlapOp_spec` |
| `THEN` | `a₂ = b₁ + 1` | `[a₁, b₂]` | [Then.lean](Intervalset/Then.lean), `thenOp_spec` |
| `THEN[gmin, gmax]` | `b₁ + 1 + gmin ≤ a₂ ≤ b₁ + 1 + gmax` | `[a₁, b₂]` | [ThenWithGap.lean](Intervalset/ThenWithGap.lean), `thenGapOp_spec` |

The gap counts frames strictly between the two intervals: a gap of zero means
adjacency. `ThenWithGap.lean` also proves that setting both bounds to zero
recovers `THEN`, for both individual intervals and the compact representation.
The table states the precise operand conventions used by these proofs,
particularly for `STARTS` and `FINISHES`.

## Optional mathematical notes

[operators.tex](operators.tex) describes the operators;
[then.tex](then.tex) covers adjacency and bounded gaps. These optional LaTeX
notes are independent of the proof build. The `.lean` theorem statements specify
what is checked.
