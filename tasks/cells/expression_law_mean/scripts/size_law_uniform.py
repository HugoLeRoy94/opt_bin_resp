#!/usr/bin/env python3
"""Arm A of the expression-law comparison: genes uniform, NUMBER of genes uniform.

A cell draws how many genes it expresses from a flat law over {1..M}, with M set
so the mean hits the swept target, then draws which genes uniformly without
replacement. Pair with size_law_exponential.py, which keeps the gene law uniform
and makes only the SIZE law exponential, so the two curves differ in one thing.

Both arms run the same (G, mean) grid: --match_size_support drops the points the
flat law cannot reach, since a flat law over {1..M} only reaches means (M+1)/2.
Without it the exponential arm would cover means the uniform arm does not and the
curves would have different support.

The two arms do NOT share a world. world_seed_for() hashes the size family into
the seed, so the latent space and the per-ligand concentration means differ
between them: a gap between the curves has to survive a change of environment.

    python3 tasks/cells/expression_law_mean/scripts/size_law_uniform.py --dry_run
    python3 tasks/cells/expression_law_mean/scripts/size_law_uniform.py --condition homomers
Anything passed on the command line overrides the defaults below.
"""
import sys
from pathlib import Path

sys.path.append('/app')
sys.path.insert(0, str(Path(__file__).resolve().parent))

from expression_law import main

DEFAULTS = [
    "--design", "size_law",
    "--size_family", "uniform",
    "--gene_family", "uniform",
    "--match_size_support",
    "--n_genes", "3", "5",
    "--means", "1.0", "1.5", "2.0", "2.5", "3.0",
    "--n_cells", "30",
    "--replicates", "3",
    "--max_pool", "8192",
    # Environment, identical to size_law_exponential.py and to
    # tasks/cells/gene_expression_mean: a mixture of ~3 ligands per sniff over a
    # concentration channel spanning 4 decades of per-ligand mean.
    "--mu_ligands", "3.0",
    "--conc_log10_span", "2.0",
]

if __name__ == "__main__":
    main(DEFAULTS + sys.argv[1:])
