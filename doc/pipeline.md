# Pipeline

End-to-end map of one run: what each stage computes and which function does it.
Physics derivations live in `../doc/theory/` and are cited, not repeated.
Cell mode is in `cells.md`, memory and batch sizes in `batching.md`.

Acronyms. MI = mutual information, in bits. KT = Kolchinsky-Tracey pairwise-distance
bound on mixture entropy (Kolchinsky and Tracey, Entropy 2017). ZTP = zero-truncated
Poisson. MM = Miller-Madow bias correction. GT = Good-Turing. EC50 = concentration at
half activation. RNG = random number generator.

Symbols.

| symbol | meaning | code |
|---|---|---|
| $B$ | sniffs (samples) in a batch | `batch_size` |
| $L$ | ligands in the fixed pool | `n_ligands` |
| $S$ | sparse ligand width of a batch, $S \le L$ | `env.s_upper` |
| $D$ | latent dimension | `latent_dim` |
| $F$ | ligand families | `n_families` |
| $G$ | genes (units) | `n_genes` |
| $k$ | subunits, hence pockets, per receptor | `k_sub` |
| $R$ | receptors in the array (in cell mode the pool $R_{\rm pool}$) | `receptor_indices` |
| $P$ | distinct pockets of the array, $P \le \min(G^2, Rk)$ | `env.pocket_pairs` |
| $N$ | binary outputs the estimator sees: $R$, or cells $C$ in cell mode | |
| $K_s$ | presence blocks (sources) | `n_presence_blocks` |
| $T$ | sigmoid temperature | `physics.temperature` |
| $\lambda$ | affinity kernel width | `kernel_params[0]` |
| $E_{\max}$ | fixed selectivity ceiling, natural-log units | `max_energy` |

Flow:

```
Config ─ train/config.py::Config.__post_init__        resolve the array (and cells)
  └ train/sweep.py::Sweep.execute                     one Runner per Config
      └ train/runner.py::Runner.run
          ├ Runner.build                              environment, physics, objective, Adam
          ├ Runner._train                             anneal T, Adam steps, optional family moves
          └ train/io.py::RunLogger.save_checkpoint    config.json, stats.csv, final_model.pt
test script ─ tasks/_lib/measure.py::measure_goal
  └ measure_run → Runner.evaluate → Runner._eval_stats → test_<label>.json
```

---

## 1. Environment

`model/environment.py::LigandEnvironment`. Theory: `03_latent_environment.md` §3.1 to §3.3.

**1.1 Family centres.** `LigandEnvironment._generate_family_centers`. $F$ centres
uniform in a $D$-ball of radius $\rho = d_{\rm fam}/\bar d_1$, with
$d_{\rm fam}$ = `average_family_distance` and $\bar d_1$ the mean distance between two
uniform points of the unit ball, estimated from 20000 pairs. Uniform ball sampling is
`UniformNBall.rsample`: direction $\times\, \rho\, u^{1/D}$, $u \sim U(0,1)$.
Buffer `family_latent` $(F, D)$.

**1.2 Ligand pool.** `LigandEnvironment.__init__`, `LigandEnvironment.sample_near_centers`.
Each of the $L$ ligands gets a family, uniformly or with probabilities
`family_weights`, then a fixed coordinate $\mathbf v_\ell$ drawn around its centre
$\mathbf v_f$ by `distribution_type`:
`gaussian` $\mathcal N(\mathbf v_f, \sigma_{\rm sh}^2 I_D)$, `uniform` uniform in the ball of
radius $\sigma_{\rm sh}$, `shell` uniform direction with radius uniform in $[0, \sigma_{\rm sh}]$.
$\sigma_{\rm sh}$ = `family_spread`. Buffers `ligand_family_assignments`, `ligand_latent` $(L, D)$.

**1.3 Presence sampler.** `LigandEnvironment._sample_masks`. Ligands are split into
$K_s$ blocks by a permutation seeded with $131071\,L + K_s$, so the partition depends
on $(L, K_s)$ only. Block sizes are $\lfloor L/K_s \rfloor$, the first $L \bmod K_s$
blocks one larger. Per sniff:

1. $n_{\rm src} \sim {\rm ZTP}(\mu_{\rm src}, n_{\max} = K_s)$, then $n_{\rm src}$ distinct blocks uniformly (Gumbel top-$k$).
2. For each active block of size $m$: $n_{\rm lig} \sim {\rm ZTP}(\mu_{\rm lig}, n_{\max} = m)$, then $n_{\rm lig}$ distinct ligands of the block uniformly.

${\rm ZTP}(\mu, n_{\max})$: $P(n) = \frac{\mu^n/n!}{\sum_{j=1}^{n_{\max}}\mu^j/j!}$, $n = 1..n_{\max}$.
$\mu_{\rm src}$ = `mu_sources`, $\mu_{\rm lig}$ = `mu_ligands_per_source`. Every sniff has at
least one ligand. `n_presence_blocks=1` with `mu_ligands_per_source=1e-6` gives one
ligand per sniff with probability $1 - O(10^{-6})$.

The selected ligands are returned as a padded index table $(B, S)$, padding index $L$.
$S$ = `s_upper` is a high-water mark: it starts at 1 and grows to the largest mixture
ever drawn, never shrinks, so no selected ligand is dropped. While $S < L$ this costs
one GPU-to-CPU synchronisation per batch. Monotone growth keeps the output shape stable.

**1.4 Concentrations.** `LogNormalConcentration.sample`.
$\log_{10} c_\ell \sim \mathcal N(\mu_{c,\ell}, \sigma_{c,\ell}^2)$, fresh per sniff and ligand,
gathered onto the $(B, S)$ table, padding set to $c = 0$. $\mu_{c,\ell}$ = `conc_mean`,
$\sigma_{c,\ell}$ = `conc_std`, scalar or one per ligand. With `block_shared_conc_mean` and
$K_s > 1$, $\mu_{c,\ell}$ is replaced once by its block average (`LigandEnvironment.__init__`).

**1.5 Observation noise.** `LigandEnvironment._sample_noisy_ligands`.
$\mathbf v_{{\rm obs},\ell} = \mathbf v_\ell + \sigma_{\rm obs}\,\boldsymbol\epsilon$, $\boldsymbol\epsilon \sim \mathcal N(0, I_D)$, fresh per
sniff and ligand slot, shared by all pockets. $\sigma_{\rm obs}$ = `observation_noise_sigma`.
Padding coordinates are set to 0.

**1.6 Pockets and energies.** Each gene $u$ has faces $\pm$ with learnable positions
$\mathbf v^\pm_u$ (`unit_latent_plus`, `unit_latent_minus`, $(G, D)$) and base energies
$E^\pm_u$ (`base_energy_u_plus`, `base_energy_u_minus`), or one scalar `base_energy_shared`
for every face when `shared_unit_energies`. The pocket between the $+$ face of $i$ and
the $-$ face of $j$ (`LigandEnvironment.pocket`):

$$\mathbf v_{ij} = \tfrac12(\mathbf v^+_i + \mathbf v^-_j), \qquad E^{\rm base}_{ij} = \tfrac12(E^+_i + E^-_j).$$

Receptor $(u_1..u_k)$ on a ring has pockets $(u_m, u_{m+1 \bmod k})$.
`LigandEnvironment.bind_receptors` lists the $P$ distinct pairs of the array
(`pocket_pairs`, $(P, 2)$) and the composition matrix $n_{pr}$ = number of times pocket
$p$ occurs in receptor $r$ (`composition`, $(P, R)$, columns sum to $k$). A pocket's
energy depends only on its pair, so each is computed once. `LigandEnvironment._compute_energies`:

$$E_{b\ell p} = E^{\rm base}_p + E_{\max}\Big(1 - e^{-\|\mathbf v_p - \mathbf v_{{\rm obs},b\ell}\|^2/\lambda^2}\Big),$$

with $\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2\langle a, b\rangle$, clamped at 0. $E_{\max}$ is the
buffer `max_energy`, never optimised, and must be positive.
`LigandEnvironment.sample_batch` returns $E$ $(B, S, P)$, $c$ $(B, S)$ and the dense
presence mask $(B, L)$. It requires `bind_receptors` first.

**1.7 Initialisation.** $\mathbf v^\pm_u \sim \mathcal N(0, I_D)$.
$E^\pm_u = \overline{\ln c} = \ln 10 \cdot \frac1L\sum_\ell \mu_{c,\ell}$, so EC50 matches a
typical concentration at step 0. With `family_moves > 0` both faces of every gene are
then placed at one draw from a uniformly chosen family (`train/moves.py::place_near_families`).

---

## 2. Receptor physics and temperature

`model/receptors.py::BinaryReceptor.forward`. Theory: `02_biophysics_mwc.md` §2.4 to §2.7.

$$\ln {\rm EC50}_{b\ell r} = \frac1k \sum_p E_{b\ell p}\, n_{pr}, \qquad
z_{br} = \operatorname{LSE}_\ell\big(\ln(c_{b\ell} + 10^{-12}) - \ln {\rm EC50}_{b\ell r}\big),$$
$$p_{br} = \operatorname{clamp}_{[0,1]}\, \sigma(z_{br}/T).$$

LSE = log-sum-exp over the $S$ slots, $\sigma$ the logistic function. The first line is
one matmul `E @ composition / k_sub`. Padding slots contribute $10^{-12}/{\rm EC50}$.

**Temperature schedule.** `train/runner.py::Runner._train`. With
$t_a = \max(1, \lfloor 0.8\, n_{\rm epochs}\rfloor)$:

$$T(t) = T_{\rm end} + (T_0 - T_{\rm end})\big(1 - \min(1, t/t_a)\big) \ \text{ if } T_{\rm end} < T_0, \qquad T(t) = T_{\rm end} \text{ otherwise}.$$

$T_{\rm end}$ = `temperature`. $T_0$ = `initial_temperature`, a float, or `"auto"`:
`model/receptors.py::compute_initial_temperature` returns
$\max(\operatorname{std}_{b,r} z_{br}, 10^{-3})$ over 2048 sniffs at initialisation, so the
sigmoid starts neither saturated (vanishing gradient) nor flat at 0.5. The per-epoch
measurement, the checkpoint and every later measurement use $T_{\rm end}$.

---

## 3. Cell mode

Entered when any of `cell_gene_sets`, `cell_receptors`, `n_cells` is set. The physics
runs over the pool $R_{\rm pool}$ and `model/cells.py::cell_activity` maps $(B, R_{\rm pool})$
to $(B, C)$ with $A_{bc} = \sum_r w_{cr}\, p_{br}$. Everything downstream sees $(B, N)$
and does not know which. Switch point: `Runner._activity`. Details in `cells.md`.

---

## 4. Objectives

`train/runner.py::build_loss`, classes in `estimate/losses.py`, kernels in
`estimate/entropy.py`. Each returns $-I$ and is minimised. There is no entropy-only
objective: $\max H(Y)$ is attained by fair coins that carry no information.

| `mi` \ `grouped` | `False` | `True` (cell mode only) |
|---|---|---|
| `shannon` | `ExactMutualInformationLoss` | `GroupedCellMutualInformationLoss` |
| `kt` | `KTMutualInformationLoss` | `GroupedKTMutualInformationLoss` |

Notation. $A_{bn} = P(Y_n = 1 \mid x_b)$, the output probabilities $(B, N)$. Outputs are
independent given the input $X$. The response distribution is the empirical mixture
$p(y) = \frac1B\sum_b \prod_n A_{bn}^{y_n}(1-A_{bn})^{1-y_n}$. $h_2(a) = -a\log_2 a - (1-a)\log_2(1-a)$.
Unless stated, $A$ is clamped to $[\epsilon, 1-\epsilon]$, $\epsilon = 10^{-6}$ (`KT_EPS`).
Theory: `04_discrete_information.md` §4.

**Conditional entropy**, shared by all.
`estimate/entropy.py::compute_response_conditional_entropy`:
$H(Y|X) = \frac1B\sum_b\sum_n h_2(A_{bn})$.

**Exact MI** (`shannon`, ungrouped). `ExactMutualInformationLoss.forward`, $H(Y)$ from
`compute_shannon_joint_entropy` by building the $(B, 2^N)$ joint table:
$I = -\sum_y p(y)\log_2\max(p(y), 10^{-12}) - H(Y|X)$. Cost $O(B\,2^N)$, so small $N$ only.
This $H(Y)$ uses unclamped $A$.

**KT lower bound** (`kt`). `compute_kt_entropy`, with Bhattacharyya coefficient
$\ln {\rm BC}_{ij} = \sum_n \ln\big(\sqrt{A_{in}A_{jn}} + \sqrt{(1-A_{in})(1-A_{jn})}\big)$:

$$I_{\rm lo} = -\frac1B\sum_i \log_2\Big(\frac1B\sum_j {\rm BC}_{ij}\Big), \qquad H_{\rm lo}(Y) = H(Y|X) + I_{\rm lo}.$$

All pairs, diagonal kept, normalised by the full $B$. ${\rm BC}_{ii} = 1$, so
$I_{\rm lo} \le \log_2 B$. `return_mi=True` returns the separation term $I_{\rm lo}$ directly,
which avoids subtracting two large entropies. The loss is $-I_{\rm lo}$. Cost $O(B^2 N)$,
tiled in $(m, m)$ blocks, $m = \min(B, 4096)$ in training (`KT_TRAIN_TILE`).
`recompute_backward` gradient-checkpoints each row block, `compile_kt` compiles the tile
kernel (`_kt_logbc_tile`). Neither changes the value.

**KT upper bound** (measurement only). `compute_kt_upper_entropy`, same structure with
${\rm BC}$ replaced by $e^{-{\rm KL}_{ij}}$,
${\rm KL}_{ij} = \sum_n \big[A_{in}\ln\frac{A_{in}}{A_{jn}} + (1-A_{in})\ln\frac{1-A_{in}}{1-A_{jn}}\big]$:

$$I_{\rm up} = \min\Big(-\frac1B\sum_i \log_2\Big(\frac1B\sum_j e^{-{\rm KL}_{ij}}\Big),\ N - H(Y|X)\Big), \qquad
H_{\rm up}(Y) = \min\big(H(Y|X) + \text{sep},\ N\big).$$

The cap holds because $H(Y) \le N$ bits for $N$ binary outputs, and the minimum of two
upper bounds is an upper bound.

**Grouped versions.** Cells with identical abundance rows form group $j$ of size $n_j$,
$J$ groups, group probability $p_{bj}$ = mean of its cells' $A$ (`estimate/grouping.py::CellGrouping.probabilities`).
See `cells.md` for why this is exact.

- `GroupedKTMutualInformationLoss`: the KT kernels above on $J$ columns with every
  per-column term ($\ln{\rm BC}$, KL, $h_2$) multiplied by $n_j$, and $N = \sum_j n_j = C$
  in the upper cap. Identical to ungrouped KT on the $C$ cells, at cost $O(B^2 J)$.
- `GroupedCellMutualInformationLoss`: exact over count vectors $\mathbf k$,
  $P(\mathbf k|x_b) = \prod_j \binom{n_j}{k_j} p_{bj}^{k_j}(1-p_{bj})^{n_j - k_j}$, enumerated over the
  $Q = \prod_j(n_j+1)$ states (`count_states`), renormalised per row:
  $I = H(\mathbf K) - H(\mathbf K|X)$ with $H(\mathbf K)$ the entropy of $\frac1B\sum_b P(\cdot|x_b)$.
  Refused when $Q >$ `cell_grouped_max_states` (default 65536). Labeled entropies add
  $D = \mathbb E_{\mathbf K}\sum_j\log_2\binom{n_j}{K_j}$ to both terms (`metrics_from_statistics`).

---

## 5. Training loop

`train/runner.py::Runner._train`. Adam (`torch.optim.Adam`, learning rate `lr`, no
scheduler) over `env.parameters()`: `unit_latent_plus`, `unit_latent_minus`, and the base
energies. `BinaryReceptor` has no parameters. Per epoch $t$:

1. set $T(t)$ (§2),
2. `env.sample_batch(batch_size)`, activity via `Runner._activity`, loss, backward, Adam step,
3. one family move if due (below),
4. every $\max(1, \lfloor n_{\rm epochs}/100 \rfloor)$ epochs, append one row of statistics.

**stats.csv** (`train/io.py::RunLogger.save_stats`), one row per logged epoch:

| column | when | meaning |
|---|---|---|
| `train_mutual_information` | always | $-$loss on the training batch at the current $T(t)$ |
| `lr` | always | Adam learning rate |
| `loss` | `per_epoch_measure=False` | training loss |
| each key of `measurement_fns`, `response_evaluation_samples` | `per_epoch_measure=True` (default False; needs `measurement_fns`, default empty) | `Runner._eval_stats` on `test_batch_size` sniffs at $T_{\rm end}$ |
| `moves_accepted` | `family_moves > 0` | cumulative accepted moves |
| `epoch` | always | index of the logged row |

**Checkpoint.** `RunLogger.save_checkpoint` writes `final_model.pt` once, after the last
epoch: `epoch`, `env_state`, `physics_state`, `receptor_indices`, and `readout_state`
(the $W$ buffer) in cell mode. It is the final state. Runs written before the rename
carry `best_model.pt`, still read (`train/io.py::checkpoint_path`).

**Family moves.** `train/moves.py::FamilyMoves`, homomers only (`Config` refuses
otherwise), off when `family_moves = 0`. Receptor $r$ is gene $r$, its pocket is at
$\frac12(\mathbf v^+_r + \mathbf v^-_r)$.

- Proposal (`FamilyMoves._jump`): a uniform gene, both faces moved to one draw from a
  uniformly chosen family. $E^{\rm base}$ unchanged.
- Score (`FamilyMoves._mi`): $\Delta I = I_{\rm after} - I_{\rm before}$, both $= -$loss on the
  same batch of `batch_size` sniffs, fixed by an RNG seed under `torch.random.fork_rng`.
- Relax (`FamilyMoves._refine`, when `family_move_relax`): before scoring, the moved
  gene alone takes `every` Adam steps with a fresh optimiser, all other rows frozen. A
  rejection then costs compute only.
- Acceptance (`FamilyMoves.step`): probability $\min(1, e^{\Delta I / T_{\rm mv}})$,
  $T_{\rm mv}(t) = T_{\rm mv,0}(1 - t/t_a)$. $T_{\rm mv,0} = \max(\operatorname{std}\Delta I, 10^{-3})$ over $G$
  unrelaxed trial moves, each undone, at construction. Rejected moves are undone.
- Accepted moves zero the Adam first moment (`exp_avg`) of the moved row in both face
  tensors. The second moment is kept, its scale does not depend on the site.
- Schedule (`FamilyMoves.due`): every `every` $= \max(1, \lfloor t_a / (\texttt{family\_moves}\cdot G)\rfloor)$
  epochs for $0 < t < t_a$, so about `family_moves` proposals per gene.

---

## 6. Measurement

A separate step. Counting needs $B$ far above the effective alphabet, KT costs
$O(B^2 N)$ and wants a modest $B$, so one budget cannot serve both, and re-measuring
must not require retraining.

**Driver.** `tasks/_lib/measure.py::measure_goal` → `measure_sweep` → `measure_run`:
`train/io.py::load_config`, `Runner.build`, weights from
`analysis/load.py::load_model`, $T = T_{\rm end}$, `bind_receptors`, then
`Runner.evaluate(measurements, batch_size, eval_chunk_size)`. Skips runs without
`final_model.pt`, and runs that already have `test_<label>.json` unless `overwrite`.

**Streaming.** `Runner._eval_stats` draws sniffs in chunks of
$\min(\texttt{eval\_chunk\_size} \text{ or } \texttt{batch\_size}, B_{\rm eval})$ until $B_{\rm eval}$ are
seen, under `no_grad`, float32. It accumulates $H(Y|X)$, grouped count sums, and the
activities (for KT), hard codes (for `codeword_entropy`). Output draws use their
own generator seeded from `torch.initial_seed() + 104729`, so they do not move the
environment RNG. Every measurement uses the full $B_{\rm eval}$. An unknown or empty
request raises.

**Names.** `estimate/measurements.py::MEASUREMENTS`. Keys written are in brackets.

| name | records |
|---|---|
| `entropy_kt` | $H_{\rm lo}(Y)$ [`full_array_entropy_kt`] |
| `entropy_kt_upper` | $H_{\rm up}(Y)$ [`full_array_entropy_kt_upper`] |
| `conditional_entropy_response` | $H(Y\mid X)$ [`conditional_entropy_response`] |
| `mutual_information_kt` | $I_{\rm lo}$, grouped kernel if the loss is grouped KT [`mutual_information_kt`] |
| `mutual_information_kt_upper` | $I_{\rm up}$ [`mutual_information_kt_upper`] |
| `codeword_entropy` | counting entropy of hard codewords $\mathbb 1[A > 0.5]$ [`codeword_entropy_{plugin,mm,log2B,K_hat,K_frac,missing_mass}`] |
| `grouped_information` | exact count enumeration of §4 [`grouped_count_entropy`, `mutual_information_grouped`, ..., see `cells.md`] |
| `grouped_counting` | binomial draws $K_j \sim {\rm Bin}(n_j, p_j)$, counting entropy of $\mathbf K$ minus analytic $H(\mathbf K\mid X)$ [`*_counting_{plugin,mm}`, see `cells.md`] |

KT in evaluation runs on the concatenated batch with tile 2048 (`EVAL_TILE`).

**Counting estimators.** `estimate/counting.py::entropy_from_counts`, from the
frequencies $f_s$ of the $U$ distinct symbols among $n$ observations:

- plug-in $\hat H = -\sum_s \frac{f_s}{n}\log_2\frac{f_s}{n}$, at most $\log_2 n$,
- MM $\hat H_{\rm MM} = \hat H + \frac{U-1}{2n\ln 2}$,
- GT missing mass $f_1/n$, $f_1$ = number of symbols seen exactly once: the estimated
  probability of never-sampled symbols. A plug-in value where it is large measures $n$,
  not the distribution.

Neither correction is a bound. Negative MI estimates are kept, not clipped.
`codeword_entropy` also reports $K_{\rm frac} = U/2^N$. It is the entropy of the
deterministic thresholded map, so subtracting $H(Y\mid X)$ from it would mix channels.
Binary codes are counted by `torch.unique(dim=0)` in `miller_madow_entropy`. Grouped
counts use `estimate/counting.py::SymbolCounter` (see `batching.md`).

**Exact hard-codeword estimators.** `estimate/exact.py`, driven by
`tasks/_lib/measure.py::measure_exact_run` [`entropy_exact`, with `_sem`, `_samples`] and
`measure_channels_run` [`entropy_given_ligand`, `entropy_given_conc`]. No finite ligand
set and no counting, so the error does not grow with $H$. Model: ligand
$\mathbf l \sim \mathcal N(\mathbf m, \sigma^2 I_D)$, $s = \ln c \sim \mathcal N(\mu_c, \sigma_c^2)$, receptor
mixing weights $w_{rq} = n_{qr}/k$ over pocket (or gene) positions $\mathbf x_q$:

$$a_r(\mathbf l) = -\ln\sum_q w_{rq}e^{-\|\mathbf l - \mathbf x_q\|^2/\lambda^2}, \qquad
\theta_r = E^{\rm base} + E_{\max}(1 - e^{-a_r}), \qquad Y_r = \mathbb 1[s > \theta_r].$$

Homomers use $a_r = \|\mathbf l - \mathbf x_r\|^2/\lambda^2$. The codeword $S$ is the set of the
$|S|$ receptors with smallest $a_r$, so
$p(S) = P_g(V_S)\,\mathbb E_g[\Phi(\tfrac{\theta_{\rm hi} - \mu_c}{\sigma_c}) - \Phi(\tfrac{\theta_{\rm lo} - \mu_c}{\sigma_c}) \mid V_S]$,
$V_S$ the ligand region whose best-matched set is $S$, $\theta_{\rm lo}$ the largest
threshold in $S$, $\theta_{\rm hi}$ the smallest outside, $\Phi$ the standard normal
cumulative distribution. $P_g(V_S)$ is estimated by subset simulation with elliptical
slice sampling (`_subset`, `_ess`), and $H = \mathbb E[-\log_2 p(S)]$ over drawn stimuli.

- `exact_codeword_entropy`: $H(Y)$. Relative error on $p(S)$ is about
  $\sqrt{\ell(1-\rho)/(\rho\, n_{\rm chains})}$, $\ell \approx \log_2(1/P_g(V))$ levels at $\rho = 1/2$.
  It averages out of $H$ and leaves an upward bias of order its square. A disconnected
  $V_S$ (possible for heteromers) adds a bias this formula does not cover.
- `exact_entropy_given_ligand`: $H(Y|\mathbf l)$ in closed form per $\mathbf l$ from the sorted thresholds.
- `exact_entropy_given_concentration`: $H(Y|s)$ by the same subset simulation.
  $I(Y;\mathbf l) = H - H(Y|\mathbf l)$, $I(Y;s) = H - H(Y|s)$.

Validity: one ligand per sniff, no observation noise, hard threshold ($T \to 0$, so
$H(Y|X) = 0$ and $H = I$), one shared $E^{\rm base}$, one Gaussian family whose continuous
cloud replaces the finite pool, and concentration parameters common to all ligands.
`_exact_inputs` asserts `n_families == 1`, `distribution_type == "gaussian"`,
`observation_noise_sigma == 0` and `shared_unit_energies`. It reads $\mu_c$, $\sigma_c$ from
ligand 0 and does not check the other conditions.

---

## 7. Sweeps and IO

**Config.** `train/config.py::Config` is flat and is exactly `config.json`.
`Config.__post_init__` resolves the receptor array (`model/arrays.py::build_heteromer_array`
from `n_receptors`, else one homomer per gene) or, in cell mode, the gene sets and pool,
and stores the result. A run is then reproducible from its own directory without
replaying a seed. It also rejects `grouped` or grouped measurements outside cell mode,
and `family_moves` without one homomer per gene.

**Sweep.** `train/config.py::sweep(base, **axes)` returns one `Config` per step with
axes zipped, never crossed. All axes must share one length. `base` is a dict, so a
resolved array is never inherited by a step that changed `n_genes`.
`train/sweep.py::Sweep.execute` trains the runs in order. Runs are independent:
nothing is carried between them, so the spread across a sweep reads as independent
replicates. Batch sizes are explicit integers (`batching.md`).

**Layout.** `train/io.py::SweepLogger`, `RunLogger`, `_run_rel_path`:

```
<data_root>/<name>_<YYYYmmdd_HHMMSS>[_n]/
    sweep_config.json            every run's Config
    .state                       running | complete | failed | interrupted
    <field>_<value>/.../run_<YYYYmmdd_HHMMSS_ffffff>/
        config.json              resolved Config
        stats.csv                training history (§5)
        final_model.pt           checkpoint (§5)
        test_<label>.json        one per measurement pass
```

One directory level per field that varies across the sweep, sorted by name. Sequence
fields (`_UNNAMEABLE`) are skipped, except `cell_receptors`, named
`receptors_per_cell_<counts>`. `_n` suffixes a second sweep started within the same
second. The tree is the only index.

A run is finished when `final_model.pt` exists (`is_finished`). A test pass writes
`test_<label>.json` (`RunLogger.save_results`): each measured key as a one-element list,
`<label>_samples` (sniffs actually used) and `<label>_batch_size`.
`load_run` merges all `test_*.json`. A key reported by two passes with different values
is kept as `<key> (<label>)` for each, and the plain key is dropped.

**Legacy runs.** `train/io.py::load_config` calls `_upgrade_legacy_config`: it renames
`fixed_max_energy` to `max_energy`, raises `LegacyRunError` when `max_energy` is missing
(learnable $E_{\max}$) or when a classic-model (`use_interface_model: false`) run has any
heteromer. `upgrade_env_state`, applied by `analysis/load.py::load_model`, maps a classic
homomer checkpoint exactly: both faces at the old `unit_latent`, $E^{\rm base}$ copied to
both faces, `max_energy` from the config. A homomer pocket then sits at the unit's old
position with the old energies.
