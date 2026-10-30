# large_R — homomer arrays with many more receptors than dimensions

Companion of `analytic_baseline`, which tests the closed form for `R <= D`. Here
`R >> D`, at small `D` (2 and 3) so that the regime is reachable. Same setting: one
Gaussian ligand family, gaussian kernel, one ligand per sniff, homomers, shared
energies with `Emax` fixed (`fixed_max_energy`) and `Ebase` learned, near-hard
readout. MI = mutual information, in bits.

## What is predicted

**Exact reformulation.** With equal norms `U`, receptor `r` fires iff
`u_r . omega > c`, with `omega = l/||l||` the ligand direction and `c = tau/||l||`
the shared threshold over the ligand norm. The pattern depends on `(omega, c)` only:
it is the set of receptor directions inside a cap of the sphere, centred on `omega`,
of angular radius `arccos c`. `|c| >= 1` gives all on or all off.

**Counting bounds.** Each receptor is a hyperplane in `(l, tau)`, a space of
dimension `D+1`.

| array | hyperplanes | bound | large R |
|---|---|---|---|
| equal norms (shell) | all through one point | `log2(2 sum_{i<=D} C(R-1,i))` | `~ D log2 R` |
| unequal norms (ladder) | general position | `log2(sum_{i<=D+1} C(R,i))` | `~ (D+1) log2 R` |

So at large R a shell wastes one dimension and the optimal array should spread its
norms.

**Expansion** (`analytic_baseline/_lib/theory.mi_large_R`). High-resolution
quantisation of `(omega, c)` by a shell with uniformly spread directions:

```
H(A) ~ log2|S^{D-1}| + h(c) + D log2(R f_D) + (D-1)/2 E log2(1-c^2) - kappa_D
```

`f_D(c) = (1-c^2)^((D-3)/2) / B(1/2,(D-1)/2)` is the density of `u . omega`,
`h(c)` the differential entropy of `c`, `kappa_D` the mean log2-volume of the cell
containing the stimulus in the unit-density local arrangement (`theory.cell_kappa`,
Monte Carlo, about +-0.2 bit). Atoms `|c| >= 1` are weighted separately. For a
small, centred `c` with `sd(c) = nu/sqrt(D)`, the `c`-dependent part is
`log2 nu - 0.72 D nu^2`, maximal at `nu = 1/sqrt(D)`, that is `sd(c) = 1/D`.

Validity: `R >> D^2` (many cells per dimension) and `R sd(c) >> 1` (the `c` axis is
resolved). These are conditions, not checked results. The expansion has not been
validated against an exact computation yet.

## The sweep

`scripts/sweep_large_R.py`. Zipped axes, cold start (a warm-start chain would share
one environment across R, see AGENTS.md).

| axis | default |
|---|---|
| `latent_dim` D | 2, 3 |
| R = `n_genes` | 2, 4, 8, 16, 32, 64 |
| `conc_std` (log10) | 0.1086, 0.4343, 1.7372 |
| replicates | 3 |
| epochs | `600 R + 2000` (rule from the analytic_baseline convergence) |

Fixed: `family_spread = 1`, `lambda = sqrt(2D)` so that `rho = 0.707` at every D,
`Emax = 10`, `temperature = 0.05`, `n_ligands = 50000`, `final_test_batch_size = 2^20`
(counting ceiling 20 bits). The dry run prints the theory table, both bounds and the
budget (about 1.6 M epochs).

```
python3 tasks/receptors/large_R/scripts/sweep_large_R.py --dry-run
./run_remote.sh receptors/large_R sweep_large_R.py 0
tasks/receptors/large_R/sync.sh
```

## Analysis

`analysis/large_R.py`. One cell per check, figures shown, nothing saved.

| figure | content |
|---|---|
| MI vs R | data, expansion at `U*` (dashed), expansion at the measured `U`, `Ebase` (crosses), small-R theory for `R <= D+1`, both bounds, `R = D^2` marked |
| slope | `dMI/dlog2R` against `D` and `D+1` |
| geometry | `std(U)/<U>` (ladder) and zero-sum residual vs R |
| operating point | `D sd(c)` at the measured geometry against 1, and the weight of the atoms |

Runs with Good-Turing missing mass above 1e-3, or within 2 bits of the counting
ceiling, are dropped and listed.

## Layout

```
_lib/large_r.py              bounds, c law, lambda per D; imports theory and runs
                             from analytic_baseline/_lib
scripts/sweep_large_R.py     the sweep
analysis/large_R.py          the analysis
sync.sh                      pull data/large_R from the cluster
```

## Status

Written, dry-run and config expansion checked, analysis smoke-tested on the
analytic_baseline data. No large_R simulation has been run.
