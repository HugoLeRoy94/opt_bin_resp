#!/usr/bin/env python3
"""Arm B of the expression-law comparison: genes uniform, NUMBER of genes exponential.

A cell draws how many genes it expresses from p_i proportional to exp(lambda*i)
over {1..G}, with lambda solved for the swept mean, then draws which genes
uniformly without replacement. Only the SIZE law changes against
size_law_uniform.py; the gene law stays uniform, so the two curves differ in one
thing.

The exponential law has support on the WHOLE range {1..G} at every mean above 1,
so even at mean 1.5 a few cells express all G genes. Two consequences:
  - the receptor pool saturates at the full G-gene pool almost immediately
    (629 receptors at G=5 for every mean > 1, against 53 -> 629 for the flat law),
    so this arm costs more per point at matched mean;
  - matched MEAN is not matched COST or matched pool. That is the law differing,
    not a confound, but it has to be said out loud when reading the curves.

--match_size_support keeps this arm on the same (G, mean) grid as the flat arm,
which can only reach means (M+1)/2.

The two arms do NOT share a world: world_seed_for() hashes the size family, so
latents and per-ligand concentration means differ between them.

    python3 tasks/cells/expression_law_mean/scripts/size_law_exponential.py --dry_run
    python3 tasks/cells/expression_law_mean/scripts/size_law_exponential.py --condition homomers
Anything passed on the command line overrides the defaults below.
"""
import sys
from pathlib import Path

sys.path.append('/app')
sys.path.insert(0, str(Path(__file__).resolve().parent))

from expression_law import main

DEFAULTS = [
    "--design", "size_law",
    "--size_family", "exponential",
    "--gene_family", "uniform",
    "--match_size_support",
    "--n_genes", "3", "5",
    "--means", "1.0", "1.5", "2.0", "2.5", "3.0",
    "--n_cells", "30",
    "--replicates", "3",
    "--max_pool", "8192",
    # Environment, identical to size_law_uniform.py and to
    # tasks/cells/gene_expression_mean: a mixture of ~3 ligands per sniff over a
    # concentration channel spanning 4 decades of per-ligand mean.
    "--mu_ligands", "3.0",
    "--conc_log10_span", "2.0",
]

if __name__ == "__main__":
    main(DEFAULTS + sys.argv[1:])
