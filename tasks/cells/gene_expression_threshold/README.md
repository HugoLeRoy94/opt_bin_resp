# Genes expressed per cell, with a thresholded cell

The counterpart of `cells/gene_expression_mean`. One variable changes, the cell
readout. Everything else — environment, physics, expression design, objective,
estimator — is imported from
`tasks/cells/gene_expression_mean/_experiments.py`, so a difference in the
curves is the readout and nothing else.

| | cell activity | parameters |
|---|---|---|
| `mean` | `A = S` | none |
| `threshold` | `A = sigmoid((S - theta) / T)` | `theta`, `T` |

- `S` (drive) is the abundance-weighted fraction of a cell's receptors that are
  open in a given sniff.
- `theta` is ONE scalar shared by every cell. It is **pinned** to the median of
  the drive and re-pinned every `cell_recalibrate_every = 25` epochs, so it is
  determined by the data and is not a fitted parameter. `--cell_threshold_learnable`
  fits it instead, which collapses to the fair-coin degeneracy while cells are
  still soft; that flag is for the ablation only.
- `T` (cell sharpness) is not learned. It anneals during phase 2 of training down
  to `cell_temperature = 0.01` times the live spread of the drive, which leaves
  roughly 0.8% of (sniff, cell) pairs inside the sigmoid's transition band, i.e.
  cells end up effectively deterministic.
- `cell_n_molecules = 1e4` is the receptor copy number per cell. `1/N` is the
  resolution of the drive and floors `theta`, which is what stops a rank
  statistic with no notion of scale from running off toward zero on a sparse
  code. It is an order-of-magnitude placeholder.

## Design

G = 5 genes, C = 30 cells, complete coverage (every gene is expressed by at least
one cell), genes per cell g = 1..5, latent dimension 6, 100 ligands with
`mu_ligands_per_source = 1e-6` so every sniff contains exactly one ligand.
Training uses the exact grouped objective; the final evaluation enumerates joint
count states exactly. These are the same numbers as the G=5, C=30
`replicates_*_complete` sweeps of the mean task, and with `--seed 0` the gene
sets are byte-identical to replicate 0 of those sweeps.

The **environments are not matched**: the world seed is hashed from the
experiment name, so each sweep draws its own ligand cloud. A small readout gap is
therefore not yet resolved; raise `--replicates` to turn it into an estimate.

The largest joint count alphabet is 460,800 states at g=2, hence the
`--max_states 1048576` default. Training holds one fp32 `(batch, states)` table of
about 7 GiB before gradients; `cell_pool_chunk=128` and `recompute_backward=True`
bound the rest.

## Running

```bash
# inspect the design without creating or training anything
python3 tasks/cells/gene_expression_threshold/scripts/gene_expression.py --dry_run

# baseline, no straight-through
bash tasks/run_remote.sh cells/gene_expression_threshold gene_expression.py 0 -- --condition heteromers
bash tasks/run_remote.sh cells/gene_expression_threshold gene_expression.py 0 -- --condition homomers

# cell only / receptor only / both
for F in "--cell_straight_through" "--receptor_straight_through" \
         "--cell_straight_through --receptor_straight_through"; do
  bash tasks/run_remote.sh cells/gene_expression_threshold gene_expression.py 0 -- --condition heteromers $F
  bash tasks/run_remote.sh cells/gene_expression_threshold gene_expression.py 0 -- --condition homomers   $F
done

bash tasks/cells/gene_expression_threshold/sync.sh
```

Output lands in
`data/gene_expression_threshold/theta<theta_mode>_<grad_mode>_<condition>_<coverage>_<timestamp>/`,
where `grad_mode` is `plain`, `stc`, `str` or `stcr`. The variant is in the folder name
because these are different experiments and a timestamp is not a description.

**g=1 is the control.** Both conditions build bit-identical arrays there (same gene sets,
same five homomers, same W), the receptors are saturated (<5% of (sniff, receptor) pairs
unsaturated), so any non-pathological theta gives the same cell and homomers and
heteromers must agree. They differ by less than 0.5% under the `mean` readout on two
independent environments, so a disagreement under the threshold readout is an
optimization failure, not environmental variability.

## Analysis

`analysis/gene_expression.py`, run cell by cell. It selects sweeps **explicitly**:
the first cell prints every sweep on disk in both goals and then stops until the
two `threshold_*` folder names are pasted into `SWEEPS`. The mean baselines are
already filled in.

It produces, in order: a per-run table, a per-point table, the printed curves, the
MI difference `threshold - mean` at matched g, a four-panel figure (MI, MI
retained against each curve's own 1-gene baseline, `H(response | sniff)`,
`H(response)`), the unaveraged per-optimization scatter, where `theta` actually
landed relative to the `k/g` drive atoms, one latent UMAP per run, and one
per-cell response UMAP per run.

Runs whose replicate seed has no counterpart in the other readout are dropped, so
the comparison stays paired on repertoires.

## Gradient flow (the reason this task underperforms the mean one)

Both nonlinearities are sigmoids whose derivative is a bump of width equal to their
own temperature:

```
cell      A = sigmoid((S - theta) / T_cell)      dA/dS = sigmoid'(.) / T_cell
receptor  p = sigmoid(ln_sum   / T_receptor)     dp/dE = sigmoid'(.) / T_receptor
```

Outside a few temperatures the derivative is not small, it underflows to zero. The two
sit IN SERIES in the chain rule:

```
dLoss/dparams = dLoss/dA . [dA/dS] . dS/dp . [dp/dE] . dE/dparams
```

As training anneals both temperatures down, both windows close, and a receptor that has
saturated (p -> 0 or 1) is far outside them and receives NO gradient — permanently,
because nothing can bring it back. It is a ratchet: receptors drift into saturation and
are absorbed there.

Measured at g=1, where the array should reach 5 bits (five binary receptors, deterministic
cells): the threshold model ends with three of five receptors frozen at 0 or 1 and 0.60
bits, having been at 4.98 bits at 90% of training. The `mean` readout, whose `dA/dS = 1`
has no window at all, keeps all five near 0.45 and reaches 5.4 bits.

`--cell_straight_through` and `--receptor_straight_through` keep the forward pass
EXACTLY as it is (the cell is still the hard sigmoid at `T_cell`, so reported MI stays
honest) and run the backward pass at a wide temperature instead:

- cell: one full drive spread, the phase-1 value, refreshed at every recalibration
- receptor: `physics.compute_initial_temperature`, i.e. the T at which the sigmoid
  argument has unit spread over the LIVE energy distribution, refreshed on the same
  cadence (the environment learns to spread its ligands, so a window measured at epoch 0
  goes narrow)

### What has been measured so far

One environment, G=5, C=15, g=1, 3000 epochs, local. NOT replicated — treat the ranking
as provisional and the magnitudes as meaningless:

| variant | MI | dead receptors |
|---|---|---|
| two-phase, no straight-through (default) | 1.52 | 1/5 |
| `--cell_phase_split 0` (anneal together) | 1.15 | 4/5 |
| `--cell_straight_through` | 2.97 | 0/5 |
| `--cell_phase_split 0 --cell_straight_through` | 0.00 | 5/5 |
| `cell_readout="mean"` reference | 5.11 | 0/5 |

So: keep the phase split (removing it is worse, and the rationale in
`doc/theory/07` 3b.3b still holds), and straight-through on the cell helps but does not
close the gap — which is what motivates doing the receptor as well.
`--receptor_straight_through` has NOT been tested; that is what this sweep is for.

## Known caveat

`cells.median_threshold` breaks a tie by stepping UP off the atom the median
lands on. With uniform homomer abundances the drive puts most of its mass on the
atoms `k/g`, and at g=2 the central atom carries about 39% of it, so the two
sides of the tie differ: stepping up gives `P(fire) = 0.29` and `H(Y)/cell = 0.863`,
stepping down gives `0.61` and `0.964`. Section 6 of the analysis prints which
gap `theta` ended up in, so this is visible rather than silent.
