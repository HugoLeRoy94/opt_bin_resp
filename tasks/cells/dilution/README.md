# dilution — heteromerisation with dilution held fixed

`fig2` sweeps genes per cell, which moves **four** things at once:

1. receptor-pool diversity
2. how finely grained each cell's weighted drive is
3. how many receptors each cell averages over (dilution)
4. how many distinct cell types exist, and how many copies of each

A peak in that sweep cannot tell you which of the four produced it. This task pins
three and moves one.

## The design

Every run holds the **pool size** at 8 receptors and the **cell subsets** fixed:
`shared_subsets` returns pool *indices*, and all three scripts use the same
`CELL_SEED`, so cell `c` holds the same slot of the pool in every arm. The arms are
matched cell-type for cell-type — identical counts of distinct repertoires at every
point (8, 23, 31, 40, 21, 1 across the sweep) and identical multiplicities.

| script | genes per receptor | what it is |
|---|---|---|
| `scripts/homomers.py` | 1 | `(u,u,u,u,u)` — one pocket, five copies |
| `scripts/dimers.py` | 2 | two genes mixed around the ring |
| `scripts/full_heteromers.py` | 5 | one gene per subunit, five distinct pockets |

Sweep axis: **receptors per cell** ∈ (1, 2, 3, 4, 6, 8). With explicit
`cell_receptors` the weights are uniform, so a cell's drive is the plain average of
its receptors' open probabilities and this axis is dilution alone — no multinomial
weighting riding along with it.

So: a **vertical gap** between curves is heteromerisation per se, and the **slope
along x** is dilution per se.

## Why genes-per-receptor is the heteromer knob

Under the interface model a binding pocket forms between adjacent subunits. A
homomer presents `k_sub` copies of a single pocket; a fully mixed word presents
`k_sub` different ones. That is the thing heteromerisation actually buys, and it is
independent of how many receptors a cell owns.

G = 8 genes, because the homomer arm needs 8 distinct homomers to fill the pool.
`R_pool` is 8 in every run, so these are cheap: 6 runs per script, 18 in total.

## Environment

Identical to `tasks/cells/fig2`: 100 ligands, `MU_LIGANDS = 3.0`,
`CONC_SPAN = 1.0`, `CONC_STD = 1.0`, latent dimension 6, one family. Each arm has
its own `WORLD_SEED` and `CONC_SEED`, so the three curves do not share a world.
Objective `kt_mi`, measured with `grouped_counting`.

## Running

```bash
bash tasks/run_remote.sh cells/dilution homomers.py 0
bash tasks/run_remote.sh cells/dilution dimers.py 1
bash tasks/run_remote.sh cells/dilution full_heteromers.py 2
bash tasks/cells/dilution/sync.sh
```

`analysis/dilution.py` draws MI, H(Y\|X) per cell, and H(Y) against receptors per
cell, one curve per arm.

## What each outcome would mean

- **Curves separate vertically, slopes identical** → heteromerisation is a
  per-receptor gain and dilution is a property of averaging, independent of each
  other. The `fig2` peak is their product.
- **Curves lie on top of each other** → heteromerisation buys nothing once pool
  size and cell diversity are controlled, and the `fig2` heteromer advantage was
  pool diversity in disguise.
- **Slopes differ between arms** → the two are not separable; dilution costs more
  for one kind of receptor than another, and the `fig2` peak needs a joint account.
