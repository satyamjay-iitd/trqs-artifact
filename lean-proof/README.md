# Lean proofs for the TrQS interval-set algebra

This directory contains machine-checked proofs for the compact interval-set
operations used in the TrQS paper. These operations let the query algebra
combine families of temporal intervals without enumerating every interval pair.
The proofs establish that the compact formulas return exactly the intervals
specified by the corresponding pairwise operations.

You can check the proofs independently of the experiments. No datasets, Rust
build, Python environment, detector, or GPU are needed.

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

Elan selects **Lean 4.30.0-rc1** from `lean-toolchain`, downloading that version
if necessary. `lake-manifest.json` pins mathlib and its dependencies to specific
Git revisions. The first run needs network access to obtain the toolchain,
dependencies, and mathlib's compiled cache. `lake exe cache get` downloads that
cache to avoid compiling mathlib from source. Keep the supplied toolchain and
manifest when reproducing the proof build.

The final command checks every operator proof and must finish with exit status
zero. Lake reports a successful build; there is no benchmark output or dataset
result to inspect. Later builds reuse the local cache under `.lake/`, which is
excluded from version control.

**Use the full command above.** The default target (`lake build`, or
`lake build Intervalset`) imports only the basic definitions and does not check
all the operator proofs.

To check one operator after editing it, for example:

```sh
lake build Intervalset.Contains
```

If `lake` is not found, ensure Elan's executable directory (normally
`~/.elan/bin`) is on your PATH and open a new shell. If dependencies or compiled
mathlib imports are missing, run `lake exe cache get` from this directory before
repeating the build.

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

[operators.tex](operators.tex) collects operator definitions and mathematical
explanations. [then.tex](then.tex) develops adjacency and bounded-gap
concatenation in more detail. They are standalone LaTeX documents for human
readers; neither is an input to Lake, the proofs, or the experiment pipeline.
A LaTeX installation is therefore optional. The `.lean` theorem statements are
the source to consult for exactly what the machine checks.
