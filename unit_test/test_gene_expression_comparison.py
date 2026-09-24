"""Estimator comparisons may differ in method, but not silently in biology/budgets."""
import json

import pytest

from tasks.cells.gene_expression.analysis._method_comparison import select_sweeps


def write_sweep(root, stamp, objective='grouped_mi', evaluation='exact',
                condition='heteromers', cells=30, batch=4096, coverage='complete', guard=262144):
    folder = root / f'replicates_{condition}_{coverage}_20260923_{stamp}'
    folder.mkdir()
    manifest = dict(experiment='replicates', condition=condition, coverage=coverage,
                    arguments=dict(entropy=objective, evaluation=evaluation),
                    protocol=dict(points=[[5, cells, 1, 'base']], settings=dict(
                        entropy=objective, batch_size=batch, cell_grouped_max_states=guard,
                        measurement_fns=[evaluation], final_measurement_fns=[evaluation])))
    (folder / 'experiment.json').write_text(json.dumps(manifest))
    return folder


def test_auto_selects_latest_per_method_with_matching_design(tmp_path):
    write_sweep(tmp_path, '110000', guard=1048576)
    exact = write_sweep(tmp_path, '120000', guard=1048576)
    hom = write_sweep(tmp_path, '121000', condition='homomers', guard=1048576)
    kt = write_sweep(tmp_path, '130000', objective='grouped_kt_mi', evaluation='counting')
    # A newer, unmatched array must not replace just one comparison panel.
    write_sweep(tmp_path, '140000', cells=40)
    assert {p for p, _ in select_sweeps(tmp_path)} == {exact, hom, kt}


@pytest.mark.parametrize('changed', [dict(cells=40), dict(batch=8192), dict(coverage='random')])
def test_explicit_selection_rejects_biological_or_budget_differences(tmp_path, changed):
    exact = write_sweep(tmp_path, '110000')
    other = write_sweep(tmp_path, '120000', objective='grouped_kt_mi', evaluation='counting', **changed)
    with pytest.raises(ValueError, match='differ beyond'):
        select_sweeps(tmp_path, [exact, other])


def test_no_matching_methods_returns_empty_instead_of_misleading_comparison(tmp_path):
    write_sweep(tmp_path, '110000')
    write_sweep(tmp_path, '120000', condition='homomers')
    assert select_sweeps(tmp_path) == []

