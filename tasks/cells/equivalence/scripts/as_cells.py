#!/usr/bin/env python3
"""Equivalence check, CELL side: C cells, each holding exactly ONE receptor.

Run this and as_receptors.py, then compare with analysis/equivalence.py. A cell holding
one receptor IS that receptor, so cell mode should reproduce the receptor model.

`cell_receptors` states the repertoire outright, one receptor per cell. Gene sets cannot
express this for a heteromer — expressing the genes of [0,0,0,1,1] also produces every
other combination of 0 and 1 — so only the explicit path can pin an arbitrary receptor
into a cell. With RECEPTORS sorted, W is exactly the identity, and the drive is p itself.

The readout is `mean`: activity = W @ p. Since W is the identity here, the cell activity
is exactly the receptor opening probability p. There is no cell threshold, calibration,
or second sharpening phase. Both scripts start from the same seed and use the same
receptor annealing schedule, so any discrepancy diagnoses the cell plumbing rather than
a difference between response models.

  ../../run_remote.sh cells/equivalence as_cells.py 0
"""
import time
import sys
sys.path.append('/app')
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent))

from src.config import RunConfig
from src.run import SweepRunner
from _shared import COMMON, RECEPTORS, seed_everything


def main():
    seed_everything()
    config = RunConfig(
        # one cell per receptor: cell j contains exactly RECEPTORS[j]
        cell_receptors = tuple((r,) for r in RECEPTORS),
        cell_readout   = "mean",
        sweep_name     = "as_cells",
        **COMMON,
    )

    print(config)
    print(f"{len(RECEPTORS)} cells, one receptor each -> W is the identity, drive = p")

    t0 = time.time()
    SweepRunner(config).execute()
    print(f"\nas_cells complete!  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
