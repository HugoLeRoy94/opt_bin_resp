"""Select comparable experimental designs while keeping estimator methods separate."""
import json
from pathlib import Path

import pandas as pd

from tasks.cells.gene_expression.analysis._shared import load_study, summarize


# Deliberate differences in this comparison. Everything else, including the
# environment, cell design, training budget and evaluation budget, must match.
METHOD_SETTINGS = {'entropy', 'measurement_fns', 'final_measurement_fns',
                   'cell_grouped_max_states'}


def design_key(manifest):
    protocol = manifest['protocol']
    return json.dumps(dict(coverage=manifest['coverage'], points=protocol['points'], settings={
        key: value for key, value in protocol['settings'].items()
        if key not in METHOD_SETTINGS}), sort_keys=True)


def method_key(manifest):
    return (manifest['arguments']['entropy'],
            manifest['arguments'].get('evaluation', 'exact'))


def method_title(method):
    objective, evaluation = method
    training = {'grouped_mi': 'Exact MI training', 'grouped_kt_mi': 'Grouped KT training',
                'kt_mi': 'KT training'}[objective]
    measurement = ('Exact grouped evaluation' if evaluation == 'exact'
                   else 'Grouped counting evaluation (plug-in)')
    return f'{training}\n{measurement}'


def select_sweeps(data_root, sweep_dirs=None):
    """Latest compatible design with multiple methods; latest per method/condition.

    Explicit folders allow selecting an older design or pooling replicates. The
    ordinary loader still validates each method's own protocol and deduplicates
    reruns. This function never weakens its general compatibility checks.
    """
    paths = (sorted(Path(data_root).glob('replicates_*/experiment.json'))
             if sweep_dirs is None else [Path(p) / 'experiment.json' for p in sweep_dirs])
    entries = []
    for path in paths:
        manifest = json.loads(path.read_text())
        if manifest['experiment'] != 'replicates':
            raise ValueError(f'Expected a replicates sweep: {path.parent}')
        if method_key(manifest)[0] not in {'grouped_mi', 'grouped_kt_mi', 'kt_mi'}:
            continue
        entries.append((path.parent, manifest))
    if sweep_dirs is not None:
        if len({design_key(m) for _, m in entries}) > 1:
            raise ValueError('Comparison sweeps differ beyond training/evaluation method: '
                             'select matching arrays, environments and budgets.')
        return entries
    designs = {}
    for folder, manifest in entries:
        designs.setdefault(design_key(manifest), []).append((folder, manifest))
    candidates = [group for group in designs.values()
                  if len({method_key(m) for _, m in group}) > 1]
    if not candidates:
        print('No compatible replicates design with multiple estimator methods found.')
        return []
    # Folder suffix is the recorded launch timestamp, independent of condition name.
    chosen = max(candidates, key=lambda group: max('_'.join(p.name.split('_')[-2:])
                                                  for p, _ in group))
    latest = {}
    for folder, manifest in sorted(chosen, key=lambda entry: '_'.join(entry[0].name.split('_')[-2:])):
        latest[(method_key(manifest), manifest['condition'], manifest['coverage'])] = (folder, manifest)
    return list(latest.values())


def load_comparison(data_root, sweep_dirs=None):
    selected = select_sweeps(data_root, sweep_dirs)
    methods = sorted({method_key(m) for _, m in selected},
                     key=lambda m: (m[0] != 'grouped_mi', m))
    runs, summaries = [], []
    for method in methods:
        folders = [p for p, m in selected if method_key(m) == method]
        title = method_title(method)
        print(f'\n{title.replace(chr(10), " / ")}:')
        frame = load_study(data_root, 'replicates', folders)
        if frame.empty:
            continue
        frame = frame.assign(method=title)
        runs.append(frame)
        summaries.append(summarize(frame).assign(method=title))
    if not runs:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(runs, ignore_index=True), pd.concat(summaries, ignore_index=True)
