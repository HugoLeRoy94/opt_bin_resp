#!/usr/bin/env python3
"""Equivalence check, CELL side: C cells, each holding exactly ONE receptor.

Run this and as_receptors.py, then compare with analysis/equivalence.py. A cell holding
one receptor IS that receptor, so cell mode should reproduce the receptor model.

`cell_receptors` states the repertoire outright, one receptor per cell. Gene sets cannot
express this for a heteromer — expressing the genes of [0,0,0,1,1] also produces every
other combination of 0 and 1 — so only the explicit path can pin an arbitrary receptor
into a cell. With RECEPTORS sorted, W is exactly the identity, and the drive is p itself.

## What "equivalent" means with the THRESHOLD readout

The readout is the real one, `threshold`: activity = sigmoid((p - theta) / T_cell). So
this tests the MODEL, not just the plumbing — which is the interesting question, but it
means the agreement is approximate rather than bit-exact. Three things follow.

  * It only works because theta is floored at 1/N (doc/theory/09 §9.8.1). Unfloored, the
    median runs to ~1e-39 on a sparse code and splits a cloud of drives that are below
    one open channel; floored, it sits between OFF and ON. Measured on this config:
    99.9-100% of hard codes agree with the receptor code.

  * The cell run has an extra hardening phase. `threshold` sets `is_cell` True, so the
    two-phase schedule runs: phase 1 (80% here, matching the receptor run's annealing
    window) anneals the receptor with the cell held soft; phase 2 hardens the cell. The
    receptor run has no phase 2 because a receptor array is already binarised by its own
    temperature. The trajectories are therefore close but not identical, and the test is
    "do they converge to the same answer", not "are they the same run".

  * Compare the HARD codeword metrics, not the KT entropies. The receptor run reports
    soft p, whose KT value includes a conditional-entropy term; the cell run reports a
    near-binary activity, which does not. Expect receptor KT >= cell KT by roughly that
    term. `codeword_entropy_*` binarises both at 0.5 and is the like-for-like comparison.

For the bit-exact plumbing check instead, set cell_readout="mean": with W = I the
readout is a pass-through (activity = W @ p = p) and `is_cell` stays False, so the
schedules match too and the two runs agree to float noise.

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
        cell_readout   = "threshold",   # the real readout; see the module docstring
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
