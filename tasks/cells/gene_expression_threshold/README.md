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

bash tasks/run_remote.sh cells/gene_expression_threshold gene_expression.py 0 -- --condition heteromers
bash tasks/run_remote.sh cells/gene_expression_threshold gene_expression.py 0 -- --condition homomers
bash tasks/cells/gene_expression_threshold/sync.sh
```

Output lands in `data/gene_expression_threshold/threshold_<condition>_complete_<timestamp>/`.

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

## Known caveat

`cells.median_threshold` breaks a tie by stepping UP off the atom the median
lands on. With uniform homomer abundances the drive puts most of its mass on the
atoms `k/g`, and at g=2 the central atom carries about 39% of it, so the two
sides of the tie differ: stepping up gives `P(fire) = 0.29` and `H(Y)/cell = 0.863`,
stepping down gives `0.61` and `0.964`. Section 6 of the analysis prints which
gap `theta` ended up in, so this is visible rather than silent.
