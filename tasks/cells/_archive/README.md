# _archive — scripts built on the old `_experiments.py` machinery

Moved here, unchanged, when `tasks/cells/` was flattened into self-contained
scripts. They still import `_archive/_experiments.py`, which carries its own
argparse, manifest and design-expansion layer. Nothing in the live tasks imports
any of it.

Three are sweeps that would need rewriting as one-sheet scripts to come back:
`environment.py`, `replicates.py`, `scaling.py`.

Three operate on ALREADY SAVED checkpoints rather than launching a sweep, so they
are a different shape and were never going to be one-sheet scripts:

- `evaluation_budget.py` — re-measure saved models at increasing input budgets
- `estimator_crosscheck.py` — re-measure saved models with both estimators, to split
  the exact-vs-KT gap into estimator bias and optimization loss
- `training_batch.py` — retrain at several training batch sizes, judged by the exact
  estimator

`equivalence_shared.py` is the old `equivalence/scripts/_shared.py`; its contents are
now written out in `as_cells.py` and `as_receptors.py`.

Import paths inside these files were repointed at `tasks.cells._archive.*` when they
moved, so they still import and the unit tests still cover them. What they do NOT
have is a live task to run against: `environment.py`, `replicates.py` and
`scaling.py` write into `/app/data/gene_expression`, the same goal the flattened
scripts use, under their own sweep names.

## analysis/

The analysis notebooks that go with the archived scripts, moved here unchanged:
`environment.py`, `replicates.py`, `scaling.py`, `estimator_comparison.py`,
`estimator_crosscheck.py`, `evaluation_budget.py`, plus the `_shared.py` /
`_method_comparison.py` loader layer they import. Their `ROOT`/`HERE` paths and
their imports were repointed at `tasks.cells._archive.analysis.*`. Their `ROOT` /
`HERE` path constants still name `tasks/cells/gene_expression_mean/analysis`, so fix
those before running one as a notebook.
