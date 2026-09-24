# %%
"""Side-by-side training/evaluation methods on the same replicates design."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tasks.cells.gene_expression.analysis._method_comparison import load_comparison

DATA = ROOT / 'data' / 'gene_expression'
# None finds the latest matching design with multiple methods and selects the
# latest sweep per method/condition. Set a list of explicit sweep folders to
# select an older comparison; unrelated designs/budgets are rejected.
SWEEPS = None
SAVE_FIGURES = True
OUT = Path(__file__).resolve().parent
RUNS, SUMMARY = load_comparison(DATA, SWEEPS)
COLORS = {'heteromers': 'tab:blue', 'homomers': 'tab:orange'}

if not SUMMARY.empty:
    METHODS = list(SUMMARY.method.drop_duplicates())
    columns = ['method', 'condition', 'n_genes', 'n_cells', 'genes_per_cell',
               'n', 'mi_mean', 'mi_sem', 'counting_unique_fraction_mean']
    print('\n' + SUMMARY[columns].to_string(index=False))
    print('\nEach optimization contributes one mean over its test repeats. '
          'Error bars are SEM across optimizations; n counts completed runs.')
    print('Training objective AND evaluation estimator differ between panels. '
          'This does not isolate KT optimization error from counting bias. '
          'Re-evaluate both sets of checkpoints with a common estimator to do that.')


def setup_axis(ax, method, ylabel):
    part = SUMMARY[SUMMARY.method == method]
    arrays = ', '.join(f'G={g}, C={c}' for g, c in
                       part[['n_genes', 'n_cells']].drop_duplicates().itertuples(index=False, name=None))
    ax.set(title=f'{method}\n{arrays}', xlabel='genes expressed per cell', ylabel=ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y', alpha=.2)
    missing = set(COLORS) - set(part.condition)
    if missing:
        ax.text(.02, .97, 'No saved ' + ', '.join(sorted(missing)) + ' runs',
                transform=ax.transAxes, va='top', fontsize=9, color='dimgray')


def plot_metric(ax, method, column, ylabel, show_runs=False):
    part = SUMMARY[SUMMARY.method == method]
    for (condition, genes, cells, profile, coverage), points in part.groupby(
            ['condition', 'n_genes', 'n_cells', 'profile', 'coverage'], sort=False):
        points = points.sort_values('genes_per_cell')
        n_min, n_max = int(points.n.min()), int(points.n.max())
        n_label = str(n_min) if n_min == n_max else f'{n_min}–{n_max}'
        label = f'{condition} (n={n_label})'
        if len(part[['n_genes', 'n_cells', 'profile', 'coverage']].drop_duplicates()) > 1:
            label += f', G={genes}, C={cells}, {profile}, {coverage}'
        color = COLORS.get(condition, 'gray')
        ax.plot(points.genes_per_cell, points[column], 'o-', color=color, label=label)
        if column == 'mi_mean':
            valid = points.mi_sem.notna()
            ax.errorbar(points.loc[valid, 'genes_per_cell'], points.loc[valid, column],
                        yerr=points.loc[valid, 'mi_sem'], fmt='none', color=color, capsize=3)
        if show_runs:
            raw = RUNS[(RUNS.method == method) & (RUNS.condition == condition)
                       & (RUNS.n_genes == genes) & (RUNS.n_cells == cells)
                       & (RUNS.profile == profile) & (RUNS.coverage == coverage)]
            ax.scatter(raw.genes_per_cell, raw.mi, color=color, s=18, alpha=.25)
    setup_axis(ax, method, ylabel)
    ax.legend(fontsize=9)


# %%
# Primary comparison: identical axis scales and strategy colors in both panels.
if not SUMMARY.empty:
    fig, axes = plt.subplots(1, len(METHODS), figsize=(6 * len(METHODS), 5),
                             sharex=True, sharey=True, squeeze=False)
    for ax, method in zip(axes.flat, METHODS):
        plot_metric(ax, method, 'mi_mean', 'MI [bits], mean ± SEM', show_runs=True)
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(OUT / 'estimator_comparison_mi.png', dpi=180)
    plt.show()

# %%
# Both quantities refer to the full labeled response Y, including for counting.
if not SUMMARY.empty:
    fig, axes = plt.subplots(2, len(METHODS), figsize=(6 * len(METHODS), 8),
                             sharex=True, sharey='row', squeeze=False)
    for col, method in enumerate(METHODS):
        plot_metric(axes[0, col], method, 'full_entropy_mean', 'H(Y) [bits], mean')
        plot_metric(axes[1, col], method, 'noise_mean', 'H(Y | X) [bits], mean')
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(OUT / 'estimator_comparison_entropies.png', dpi=180)
    plt.show()
