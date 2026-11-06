# two_families — does an array facing two distant families split into two arrays?

Companion of `analytic_baseline` (one family, `R <= D`) and `large_R` (one family,
`R >> D`). Same setting: gaussian kernel, one ligand per sniff, homomers, shared
energies with `Emax` fixed and `Ebase` learned, near-hard readout. MI = mutual
information, in bits.

## What is predicted

Two families `F = 1, 2`, presented with probabilities `pi` and `1 - pi`, far enough
apart that a receptor tuned to one never fires for a ligand of the other. Each
receptor then serves one family. With `R_f` receptors on family `f` and `G(n)` the
MI of an optimal single-family array of `n` receptors, the exact chain rule gives

```
MI_2 = H(F) + pi H(A|F=1) + (1-pi) H(A|F=2) - H(F|A)
     ~ H(F) + max_{R_1} [ pi G(R_1) + (1-pi) G(R-R_1) ] - H(F|A)
```

The second line is the prediction. It assumes `H(A|F=f) = G(R_f)`: the group on a
family behaves as an optimal single-family array and ignores the other family.
`H(F|A)` is the ambiguity of the patterns both families can produce: all off, and all
on when the concentration exceeds the shared ceiling `Ebase + Emax`.

**Allocation.** The next receptor goes to the family with the larger `pi_f dG(R_f)`.
While `pi dG(R_1) > (1 - pi) G(1)`, every receptor goes to the more probable family.
After that the weighted marginal gains equalise. In the log regime `G ~ s log2 n` this
gives `R_f` proportional to `pi_f`.

**`G` is not simulated here.** It is the MI of the `large_R` sweep at `D = 2`,
`conc_std = 1.7372` (3 replicates, R = 2 to 64), whose parameters this task copies.
`G(1) = 1` (one balanced receptor), `G(0) = 0`, and `G` is interpolated linearly in
`log2 n` between measured R (`_lib/two_fam.interp_G`). At `pi = 0.8` the predicted split
puts every receptor on family 1 up to `R = 12`, then `R_1* = 14, 21, 27, 39, 52` at
`R = 16, 24, 32, 48, 64`.

## Setting

| parameter | value | note |
|---|---|---|
| `D` | 2 | as large_R |
| `lambda` | 2 | `sqrt(2D)`, `rho = 0.707` |
| `conc_std` | 1.7372 (log10) | as large_R; closed-form `nu = 1.68` |
| `Emax` | 10, fixed | |
| `family_weights` | (0.8, 0.2) | new `RunConfig` field, see below |
| geometry | `asymmetric`, average centre distance 8 | centres random in a disc, so the realised distance changes from run to run |
| R | 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64 | |
| replicates | 1 | |
| epochs | `600 R + 2000` | |

Fixed as in large_R: `family_spread = 1`, `temperature = 0.05`, `n_ligands = 10000`,
`final_test_batch_size = 2^20` (counting ceiling 20 bits). 12 runs, 0.16 M epochs,
about 3 h at 17 epochs/s.

`family_weights` (`src/environment.py`): each pool ligand is assigned to family `f` with
probability `family_weights[f]`, instead of uniformly. With one ligand drawn uniformly
per sniff, this is the probability of each family. Default `None` keeps the old
behaviour.

```
python3 tasks/receptors/two_families/scripts/sweep_two_families.py --dry-run
./tasks/run_remote.sh receptors/two_families sweep_two_families.py 0
tasks/receptors/two_families/sync.sh
```

`--pi`, `--family-dist`, `--R`, `--replicates` override the defaults. The dry run prints
`G` from large_R and the predicted split.

## Known departures from the prediction's assumptions

1. **Initialisation far from the families.** `unit_latent` starts as `randn` around
   the origin, and the centres can sit up to about 8.8 from it. At that distance the
   kernel is about `exp(-d^2/lambda^2) ~ 1e-6` and the positions get almost no
   gradient. If the optimiser does not reach the families, a Monte Carlo move that
   reassigns a receptor to the other family is the planned fix.
2. **Shared saturation.** The concentration exceeds the shared ceiling with
   probability about 0.2 (closed form). Then every receptor fires, whatever the
   family. This adds about `0.2 h2(pi)` to `H(F|A)`, and is the floor of cross-family
   firing.
3. **Random separation.** Two centres drawn in a disc can land close. The analysis
   reports the realised distance per run.

## Analysis

`analysis/two_families.py`. Each learned array is re-evaluated by Monte Carlo
(`two_fam.decompose`, `2^20` sniffs, hard threshold, the run's own ligand pool), giving
`H(A)`, `H(F)`, `H(A|F=f)`, `H(F|A)`, the allocation (receptors nearest each centre),
`P(s > ceiling)` and `P(fire | other family)`. On analytic_baseline runs this `H(A)`
equals the simulator's `codeword_entropy_mm` to 0.001 bit.

| figure | content |
|---|---|
| MI vs R | two-family runs, `G(R)` from large_R, the prediction with and without `H(F|A)`, and at the measured split |
| core claim | `H(A|F=f)` of each group against `G(R_f)`, and `R_1/R` against the predicted split |
| side terms | `H(F|A)`, saturation, cross-family firing, realised centre geometry |

Runs with Good-Turing missing mass above 1e-3, or within 2 bits of the counting
ceiling, are drawn hollow.

## Layout

```
_lib/two_fam.py                  G interpolation, best split, Monte Carlo decomposition
scripts/sweep_two_families.py    the sweep
analysis/two_families.py         the analysis
sync.sh                          pull data/two_families from the cluster
```

## Status

Written. Dry run, config expansion and the `family_weights` round trip (config JSON,
environment, `clone_with_extra_units`) checked. `decompose` checked on analytic_baseline
runs and on a synthetic two-family array, where the chain rule closes exactly. No
two_families simulation has been run.
