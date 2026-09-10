#!/usr/bin/env python3
"""Equivalence check, RECEPTOR side: the same receptors, as a plain receptor array.

The control for as_cells.py. Identical environment, identical receptors, identical
schedule, identical seed — the ONLY difference is that the array is declared as
receptors rather than as one-receptor cells. Compare with analysis/equivalence.py.

  ../../run_remote.sh cells/equivalence as_receptors.py 0
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
        # the same receptors, declared directly. RECEPTORS is sorted, and CellArray's
        # pool is the sorted unique union, so both runs order their channels the same.
        receptor_indices = RECEPTORS,      # tuple: a list here would read as a sweep axis
        sweep_name       = "as_receptors",
        **COMMON,
    )

    print(config)
    print(f"{len(RECEPTORS)} receptors, declared directly")

    t0 = time.time()
    SweepRunner(config).execute()
    print(f"\nas_receptors complete!  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
