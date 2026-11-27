#!/usr/bin/env python3
"""fig2 arm: ONE LIGAND PER SNIFF, complete-coverage gene sets, HOMOMERS.

Same as train_complete_homomers, but each sniff holds a single ligand instead of a
mixture, and only the headline array (G = 5 genes, C = 30 cells) is run. Gene
sets are identical to the complete arms at that (G, C): same seeds.

Each cell assembles single-gene receptors only (homomers), stated outright as cell_receptors.

Writes to <data_root>/fig2_single_ligand_homomers/, so the sweep directory names this file.

    python -m tasks.fig2.train_single_ligand_homomers --dry_run
    python -m tasks.fig2.train_single_ligand_homomers
"""
import argparse

from tasks.fig2._common import launch

N_GENES_LIST = (5,)
N_CELLS_LIST = (30,)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="data/fig2")
    p.add_argument("--dry_run", action="store_true",
                   help="print the design and the budget projection, run nothing")
    a = p.parse_args()
    launch("complete", "homomers", a.data_root, a.dry_run,
           name="fig2_single_ligand_homomers", single_ligand=True,
           n_genes_list=N_GENES_LIST, n_cells_list=N_CELLS_LIST)
