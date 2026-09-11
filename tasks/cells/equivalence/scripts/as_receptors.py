#!/usr/bin/env python3
"""Equivalence check, RECEPTOR side: the same receptors, as a plain receptor array.

The control for as_cells.py: the same initial environment and receptors, and comparable
schedules. Cell calibration consumes additional samples, and the threshold changes
the channel. This is a qualitative endpoint comparison, not identical trajectories.
Compare with analysis/equivalence.py.

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
