# fig3 — can the MI maximum be moved?

In `fig2` the maximum sits at 2 genes per cell for every G and every C >= 20, and it
will not move. The reason is structural: under the mean readout the receptor **pool**
and the averaging **width** are the same function of g. A cell expressing g genes
assembles `(g^k + ...)/k` receptors, and that same number is what its drive averages
over. Benefit and cost rise together.

`k_sub` changes how fast both rise. At G=5, C=50:

```
       pool per genes/cell          receptors averaged per cell
k=2     5  15  15  15  15            1   3   6  10  15
k=3     5  25  45  45  45            1   4  11  24  45
k=5     5  65 365 605 629            1   8  51 208 629
```

Ratio of pool gain to dilution gain, step by step:

```
k_sub 1->2  2->3  3->4  4->5
    2  1.00  0.50  0.60  0.67
    3  1.25  0.65  0.46  0.53
    5  1.62  0.88  0.41  0.34
```

A step pays off only above 1.

## The prediction

**The peak moves LEFT as k_sub falls**: g=1 at k=2, g=2 at k=3 and k=5.

This corrects an earlier claim that lower `k_sub` would move it right because
dilution grows more slowly. It does — but the pool grows more slowly by at least as
much, and the ratio is what matters.

At k=2 the pool is exhausted at g=2, which makes that arm a **pure-dilution
control**: everything past g=2 is dilution with zero pool gain behind it. If MI
still falls there, dilution alone is sufficient to produce the decline.

## Running

```bash
bash tasks/run_remote.sh cells/fig3 k2.py 0
bash tasks/run_remote.sh cells/fig3 k3.py 1
bash tasks/run_remote.sh cells/fig3 k5.py 2
bash tasks/cells/fig3/sync.sh
```

15 runs per arm (5 expression levels x 3 replicate worlds), 45 total, about 1 h.

G=5, C=50, complete coverage, heteromers only — the peak does not exist in the
homomer arm of fig2, so there is nothing there to move. Environment identical to
fig2. `CELL_SEED` is shared across the three arms, so the gene sets are the same and
only the receptors those genes assemble differ.

`analysis/fig3.py` draws MI against genes per cell with the peak marked, beside the
receptor pool on the same axis.

## If the goal is to move the peak RIGHT

`k_sub` is the wrong lever, for the reason above. The right one is to break the
coupling: let a cell express g genes but assemble only a capped random subset of its
repertoire, via explicit `cell_receptors`. Dilution is then pinned while the pool
keeps growing. That is the design in `tasks/cells/dilution` run the other way round,
and it is not built yet.
