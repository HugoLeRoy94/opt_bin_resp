# analytic_baseline — validating the closed-form homomer MI against simulation

Reference: `analytics/homomer_array_mi.tex`. Equation numbers below are that
document's, as built on 2026-10-19. MI = mutual information, GT = Good-Turing,
MM = Miller-Madow.

## What is being tested

For a single Gaussian ligand family, the **gaussian (saturating) kernel**, one
ligand per sniff, a deterministic threshold readout and `R <= D` freely placed
homomers, the analytics predicts that the whole parameter dependence collapses
onto one scalar:

```
match      K   = exp(-||l-u||^2 / lambda^2)            in (0,1]
demand     Z   = (Ebase + Emax - s) / Emax             Gaussian, Eq. 9
fires iff  K > Z                                        exact, no expansion
nu^2(w)    = [ rho^2/2 + Q e^{2w} ] / w                 Eq. 32
             rho = sigma sqrt(D)/lambda,  Gamma = Emax/sigma_c,  w = U^2/lambda^2
             Q   = D e^{2 rho^2} / (4 rho^2 Gamma^2)
optimum    Lambda = W( P/(Qe) ),  w* = (1+Lambda)/2,  nu_min^2 = rho^2/Lambda   Eq. 39
MI         = R eta(nu) + I(A;q)                         Eq. 48
eta(nu)    = E_z[ h2( Phibar(nu z) ) ],  z ~ N(0,1)
```

`sigma` is the per-coordinate spread of the ligand cloud (`family_spread`),
`sigma_c` and `m` the standard deviation and mean of `ln c`, `lambda` the
affinity length scale, `h2` the binary entropy in bits, `Phibar` the standard
normal upper tail and `W` the Lambert function.

**Energy units.** The analytics is in natural log. The simulator stores
`conc_mean` and `conc_std` in log10, so `sigma_c = conc_std * ln10`. Use
`_lib/theory.from_log10`. Getting this wrong rescales `nu` by 2.3.

## Three facts about the parameters, before designing any sweep

These are not obvious and two of them cost a wasted sweep if ignored.

**1. `nu` does not depend on R.** Varying the number of receptors moves the
array between the regimes of Section 13 (`R <= D` exchangeable, `R > D`
geometric saturation) and changes the crossover `nu_star ~ 2.08 R/log2 R`, but
it leaves `nu` itself untouched. An R sweep alone will not show you different
`nu` regimes.

**2. `lambda` has a floor and cannot reach small `nu`.** `nu_min` is
*non-monotone* in `lambda`. Large `lambda` reduces `rho`, which helps, but it
also flattens the match across the cloud, which kills the private signal. At
`D=8, sigma=1, Emax=10, sigma_c=1` the floor is `nu ~ 0.98` near `lambda ~ 5`.
`lambda` is the geometry knob, not the `nu` knob.

**3. The clean knob for `nu` is `conc_std`.** At fixed `lambda = 4`, `D = 8`,
`sigma_c = 4, 2, 1, 0.5` gives `Gamma = 2.5, 5, 10, 20` and
`nu = 3.15, 1.68, 0.99, 0.67`, spanning strong degradation to near-ideal with
the self-consistency group of Eq. 41 staying usable. That is the sweep the
script implements.

## Does the analytics describe the simulation? Now yes, with one flag

It did not before. `base_energy_u` and `max_energy_u_raw` are per-gene learnable
tensors (`src/environment.py`), while the closed form assumes a single shared
`E_base` and `E_max` (assumption A6). With per-gene energies the simulator
optimises a strictly larger family, and the closed form describes only the
special case in which the optimiser happens to make them equal.

`RunConfig.shared_unit_energies = True` collapses both to one learnable scalar
each, still optimised, positions still per gene. **`sweep_R.py` sets it**, and any
run compared against the tex must. Implementation notes in
`doc/theory/07_optimization_pipeline.md`; the short version is that all readers go
through `env.base_energies()` / `env.max_energies_raw()`, which return `(n_genes,)`
in both modes, and the gradient reaching the scalar is the sum over genes, which is
the true shared-parameter gradient. Not available with `use_interface_model`.

What the flag does NOT fix, and what therefore remains a difference between the
analytics and the simulation:

| | analytics | simulation | consequence |
|---|---|---|---|
| ligand draw | fresh continuous Gaussian each sniff | fixed pool of `n_ligands` | caps the alphabet at `log2(L(R+1))`; keep `L` large |
| readout | hard step | sigmoid at temperature `T` | need `T << 0.1 * Delta_priv`; the dry run prints the bound |
| objective | exact `H(A)`, global maximum | annealed blocked-Shannon / collision estimator, gradient descent | cannot be removed. Compare against the MEASURED codeword entropy, never the training loss, and treat a shortfall as possible non-convergence |

The third is the one with no remedy. A disagreement in check 3 is therefore not
automatically a refutation of the analytics: it can equally be the optimiser failing
to find the optimum the analytics describes. Checks 1 and 2 exist precisely to tell
those apart, which is why they come first.

**`Emax` is fixed.** With `Emax` learnable the closed form has no finite optimum:
raising `Emax` together with `U` lowers `nu` without limit. In the 2026-10-19 sweep
the optimiser did not follow that direction to the end. `Emax` went from 10 to
between 6 and 23, depending on `conc_std` and on R (R and the number of epochs were
confounded). The learned value was therefore set by the training dynamics, not by
an optimum. `RunConfig.fixed_max_energy` fixes `Emax` to one value for every gene and
removes it from the optimised parameters. `sweep_R.py` sets it through `--emax`
(default 10). Gaussian kernel and classic model only.

## Layout

```
_lib/theory.py     closed form, both routes. No src/ dependency.
_lib/runs.py       loads runs, pulls geometry out of best_model.pt, attaches
                   the matching prediction at the measured Emax.
scripts/selftest_theory.py   tier 0, CPU, seconds. Run first, always.
scripts/sweep_R.py           the simulation sweep. --dry-run prints the budget.
analysis/geometry_norms.py          check 1
analysis/geometry_orthogonality.py  check 2
analysis/mi_regimes.py              check 3
```

`theory.py` offers two routes and they must agree where both are valid:

| route | what it does | when to use |
|---|---|---|
| delta | expands `ln Z` about its mean, Lambert-W optimum in closed form | `selfcons` (Eq. 41) above ~2 |
| exact | product quadrature over the two shared scalars, saturation atoms kept | always valid, slower |

`selfcons = sqrt(Lambda D / (2 rho^4))` is `Gamma*zbar` at the optimum and is
independent of `Gamma`: the operating point is set by geometry alone. Measured
agreement, from `selftest_theory.py`: at `selfcons = 3.4` the two routes differ
by 1.6% in `U*` and 0.0% in `eta`; at `selfcons = 0.44` they differ by 14% and
21%. The criterion works, so trust it rather than eyeballing `rho`.

---

## Tier 0 — self-consistency of the closed form

`scripts/selftest_theory.py`, no GPU, no `src/`. Currently **all checks pass**.
It asserts normalisation of `P(K)`, agreement of the two MI routes (Eq. 45 vs
Eq. 46), `eta(1) = 1/(2 ln 2)` to 1e-9, the small- and large-`nu` laws, the
limits `MI -> R` and `MI -> 1 bit`, the constant
`A = 2.6060764897` recomputed to 1e-6 (it matches the tex), and the
degeneracy of the quadratic kernel.

One implementation note that cost an hour: Gauss-Hermite weights **overflow to
NaN** above a few hundred nodes, silently. `q_nodes_delta` uses a normalised
uniform grid in `z`, refined so the kernel width `1/nu` is always resolved.
Separately, `q = 0` and `q = 1` occur exactly, at the grid edges and at the
saturation atoms, so the count distribution must avoid `0 * log 0`.

---

## The sweep

`scripts/sweep_R.py`. Single family, gaussian kernel, exactly one ligand per
sniff (`n_presence_blocks = mu_sources = mu_ligands_per_source = 1`), homomers
(`receptor_indices` left `None`, so `R = n_genes`), `observation_noise_sigma = 0`.

| axis | default | role |
|---|---|---|
| `n_genes` = R | 2,3,4,5,6,8,10,12,14,16 | crosses `D = 8`, six runs below and four above |
| `conc_std` | 1.7372, 0.8686, 0.4343, 0.2171 (log10) | the `nu` axis: 3.15, 1.68, 0.99, 0.67 |
| replicates | 3 | independent environments, never a shared seed (AGENTS.md) |
| epochs | `600 R + 2000` | see below |

Fixed: `D = 8`, `family_spread = 1`, `lambda = 4` so `rho = 0.707`, `Emax = 10`,
`temperature = 0.05`, `n_ligands = 10000`, `final_test_batch_size = 2^18`.

**Epochs.** The 2026-10-19 sweep used `170 R + 500`. The receptor temperature
anneals over the first 80% of the epochs. Over the last 20%, at fixed temperature,
the measured entropy was flat only for `R <= 3`. For `R >= 5` it was still rising
by 0.3 to 1.5 bits per 1000 epochs. Exponential fits to the tail of the learning
curves put the number of epochs needed to come within 0.05 bit of the plateau
between about 2000 (`R = 5`, low `conc_std`) and 8000 (`R = 8` to 12,
`conc_std = 0.87`), and about 10000 at `R = 8`, `conc_std = 1.74`. `600 R + 2000`
covers all of these except the last. The fits are rough. Check that the tail of each
new learning curve is flat before using a run.

Three budget constraints, all printed by `--dry-run`:

1. **Hard threshold (A3).** The sigmoid width must be small against the private
   signal in energy units, `Delta_priv = 2 U sigma Emax zbar / lambda^2`, which
   is 1.08 to 1.30 across the grid. `T = 0.05` is about 1/22 of it.
2. **Counting ceiling.** Any counting MI is capped by `log2 B = 18`. The worst
   node (`R=16`, `nu=0.67`) predicts 14.6 bits, a margin of 3.4. Larger `R`
   needs a larger batch, not a larger pool.
3. **Fixed ligand pool.** With `L` ligands and one per sniff the reachable
   alphabet is `L(R+1)`, here `log2 = 17.4` bits at `R = 16`. Because one ligand per sniff
   makes `s_upper = 1`, a large `L` is nearly free in memory, so keep it large.

---

## Check 1 — where the receptors sit

`analysis/geometry_norms.py`. This is first because it is the most exact part of
the calculation: it tests the geometry before any entropy estimator is involved.

| sub-check | prediction | source |
|---|---|---|
| 1a | all receptors at the same distance `U` from the family centre | Lemma 1 |
| 1b | that distance equals `U* = lambda sqrt(w*)` | Eq. 39 |
| 1c | operating point `zbar = exp(-(rho^2 + w*))`, read off `Ebase` | Eq. 30 |

1a needs no closed form, only isotropy, so it fails only if the reduction of
Sections 1 to 5 is wrong. 1b additionally tests the lever-arm argument. Distances
are measured from `env.family_latent[0]`, **not** from the origin: with one family
the centre is a random point in a ball, not the origin.

The prediction **inverts above `nu = 1`**. Section 12 says the optimal array
stops being a balanced shell and becomes a ladder of thresholds, so a rising
`std(U)/<U>` at `nu > 1` confirms the theory rather than contradicting it. The
script plots that spread against `nu` with the predicted transition marked.

Expect a small positive bias in `U/U*` even where everything works: the exact
route puts `U*` 1.6 to 2.5% below the Lambert value at these parameters.

---

## Check 2 — how the directions are arranged

`analysis/geometry_orthogonality.py`. The directions are not orthogonal: assumption
A5 was wrong. Orthogonal directions make the projections independent, but the
outputs still share the threshold. A negative pairwise cosine cancels part of that
shared fluctuation. Full cancellation needs `cos = -nu^2`, and geometry imposes
`<cos> >= -1/(R-1)`, with equality iff the directions sum to zero. On this grid
`-nu^2` is always below that floor, so the optimum is the zero-sum configuration, a
regular simplex for `R <= D+1`.

| quantity | prediction |
|---|---|
| mean cos | on `-1/(R-1)` at every R, including above `R = D` |
| zero-sum residual `||sum_r u_r|| / R` | zero |
| std of the cosines | zero for `R <= D+1`, positive above (no simplex fits) |

Measured in the 2026-10-19 sweep: mean cos on the floor to about 0.01 everywhere.
Spread about 0.02 for `R <= D` at the two lowest `conc_std`, 0.2 to 0.28 for
`R > D`, and up to 0.8 at small R for the two highest `conc_std`.

---

## Check 3 — the information, and the two regimes in R

`analysis/mi_regimes.py`. Run it last. The measured quantity is
`codeword_entropy`, which binarises the activity at 0.5 and so is exactly the
`H(A)` of assumption A3, hence exactly the MI. Miller-Madow is the less biased of
its two estimates, but neither is a bound.

| prediction | where it lives |
|---|---|
| `MI = R eta(nu) + I(A;q)` for `R <= D` | Eq. 48 |
| MI falls below that extrapolation past `R = D`, bending toward `O(D log2 R)` | Section 13 |
| `(MI - I(A;q))/R` collapses onto `eta(nu)` for every `(R, D)` | the core claim |
| `eta(1) = 0.7213` exactly | Eq. 52 |

The collapse is the real claim, more than any single value: it asserts that `nu`
is the **only** control parameter. If each `(R, conc_std)` curve is smooth but
they do not collapse, the theory is wrong in a way no single-point agreement
would reveal.

Two estimator guards are applied before any point is plotted: GT missing mass
below 1e-3, and MI at least 2 bits below `log2 B`. Points failing either are
dropped and listed, because both failure modes mimic the real sublinearity that
check 3 is looking for.

Figures are built and shown. Nothing is written to `figures/` unless asked.

---

## Diagnostic table

| symptom | most likely cause |
|---|---|
| receptor norms drift outward without converging | the quadratic kernel was used, Eq. 34 degeneracy |
| 1a passes, 1b fails | the reduction is fine, the lever-arm argument of Section 7 is not |
| 1b fails only where `selfcons < 2` | the delta method, not the theory. Use `theory.exact_optimum` |
| `|cos|` near `sqrt(2/pi D)` | optimiser never spread the array; check convergence before anything else |
| MI below theory, worsening with R | the `log2 B` ceiling or the ligand pool, not physics. Check the diagnostics panel |
| MI above theory anywhere | estimator bias; check GT missing mass first |
| collapse fails but each curve is smooth | `nu` is not the only control parameter, which falsifies the core claim |
| everything agrees at `nu ~ 1` and fails at `nu ~ 3` | `selfcons = 0.90` there; the delta route is not quantitative |

## Status

`shared_unit_energies` and `fixed_max_energy` make the simulated model the one that
was computed, up to the three residual differences tabulated above.

The first sweep (2026-10-19, learnable `Emax`, `170 R + 500` epochs) has been run
and audited by Monte Carlo on the learned arrays (2026-10-28):

- the simulator's codeword entropy equals the exact MI of its arrays to 0.02 bit for
  `R <= 10`;
- assumption A4 and the Gaussian threshold each underestimate the MI, by at most
  about 0.25 bit for `R <= 8`, and are negligible at `R = 2`;
- the closed form at the measured `nu` and mean cosine is within 0.02 bit of the
  exact MI for `R <= D` at the lowest `conc_std`, and fails for `R > D` and at the
  highest `conc_std`, where the norms are unequal (`std(U)/<U>` = 0.5 to 0.8);
- runs with `R >= 5` had not converged.

Next: rerun with `Emax` fixed and the longer epoch schedule.
