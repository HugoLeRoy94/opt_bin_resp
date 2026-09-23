"""Grouped estimators agree with binary references without a joint-state table."""
import json
from types import SimpleNamespace

import pytest
import torch

from src.bin_loss import (compute_kt_entropy, compute_kt_upper_entropy,
                          compute_response_conditional_entropy)
from src.counting import SymbolCounter
from src.grouped_loss import (GroupedCellMutualInformationLoss, GroupedKTMutualInformationLoss,
                              GroupedResponseCounter)
from src.response_groups import CellGrouping
from src.IO import ExperimentLogger
from src.run import SimulationRunner, _build_loss, resolve_batch_sizes
from unit_test.test_cell_pipeline import _config


@pytest.mark.parametrize('sizes', [(1, 1, 1), (2, 3, 1), (20,)])
@pytest.mark.parametrize('recompute', [False, True])
def test_grouped_kt_matches_binary_values_and_partial_gradients(sizes, recompute):
    torch.manual_seed(43)
    ids = torch.arange(len(sizes)).repeat_interleave(torch.tensor(sizes))
    grouping = CellGrouping(torch.eye(len(sizes))[ids])
    a = (.02 + .96 * torch.rand(11, len(sizes), dtype=torch.float64))[:, ids].requires_grad_()
    loss = GroupedKTMutualInformationLoss(grouping, collision_chunk_size=4,
                                         recompute_backward=recompute)
    soft = torch.stack((1 - a, a), -1)
    for upper in (False, True):
        fn = compute_kt_upper_entropy if upper else compute_kt_entropy
        for mi in (False, True):
            reference = fn(soft, chunk_size=4, return_mi=mi)
            actual = loss.bound(a, upper=upper, return_mi=mi)
            torch.testing.assert_close(actual, reference)
            torch.testing.assert_close(torch.autograd.grad(actual, a, retain_graph=True)[0],
                                       torch.autograd.grad(reference, a, retain_graph=True)[0])
    torch.testing.assert_close(-loss(a), loss.bound(a, return_mi=True))
    torch.testing.assert_close(loss.compute_entropy(a), loss.bound(a))


def test_conditional_binomial_entropies_match_joint_enumeration():
    ids = torch.tensor([0, 0, 1, 1, 1])
    grouping = CellGrouping(torch.eye(2)[ids])
    a = torch.tensor([[0., .9], [.4, .7], [1., .01]], dtype=torch.float64)[:, ids]
    exact = GroupedCellMutualInformationLoss(grouping).compute_metrics(a)
    hc, hyc, d = grouping.conditional_entropies(grouping.probabilities(a))
    torch.testing.assert_close(hc, exact['grouped_count_conditional_entropy'])
    torch.testing.assert_close(hyc, exact['conditional_entropy_response_grouped'])
    torch.testing.assert_close(d, exact['grouped_label_entropy'])


def test_integer_symbol_counter_streams_without_thresholding_or_overflow():
    # More than 64 coordinates prevents relying on an overflowing packed integer.
    symbols = torch.zeros(8, 80, dtype=torch.int64)
    symbols[:, 0] = torch.tensor([0, 1, 2, 2, 3, 3, 3, 4])
    counter = SymbolCounter()
    counter.update(symbols[:3])
    counter.update(symbols[3:])
    assert counter.symbols.shape == (5, 80)
    torch.testing.assert_close(counter.counts, torch.tensor([1, 1, 2, 3, 1]))
    expected = -(counter.counts.double() / 8 * (counter.counts.double() / 8).log2()).sum()
    assert counter.entropy()[0] == pytest.approx(expected)
    assert counter.n_samples == 8


def test_counting_converges_to_exact_and_preserves_entropy_meanings():
    grouping = CellGrouping(torch.ones(3, 1))
    # Deterministic choice of equally represented inputs; sample only the output.
    a = torch.tensor([[.1] * 3, [.8] * 3], dtype=torch.float64)
    reference = GroupedCellMutualInformationLoss(grouping).compute_metrics(a)
    counter = GroupedResponseCounter(grouping)
    generator = torch.Generator().manual_seed(17)
    rng_before = torch.get_rng_state()
    for _ in range(6):
        counter.update(a.repeat(5000, 1), generator)
    torch.testing.assert_close(torch.get_rng_state(), rng_before)
    metrics = counter.metrics()
    assert metrics['grouped_counting_K_hat'] == 4
    assert metrics['grouped_counting_samples'] == 60000
    for counted, exact in (
        ('grouped_count_entropy_counting_plugin', 'grouped_count_entropy'),
        ('response_entropy_grouped_counting_plugin', 'response_entropy_grouped'),
        ('mutual_information_grouped_counting_plugin', 'mutual_information_grouped'),
    ):
        assert metrics[counted] == pytest.approx(reference[exact].item(), abs=.015)
    assert metrics['response_entropy_grouped_counting_plugin'] - metrics['conditional_entropy_response_grouped_counting'] == pytest.approx(metrics['mutual_information_grouped_counting_plugin'])


def test_large_alphabet_never_enumerated_and_negative_estimates_retained():
    grouping = CellGrouping(torch.eye(70))
    assert grouping.n_states == 2**70
    a = torch.full((2, 70), .5)
    assert GroupedKTMutualInformationLoss(grouping)(a).item() == pytest.approx(0, abs=1e-5)
    counter = GroupedResponseCounter(grouping)
    counter.update(a, torch.Generator().manual_seed(0))
    assert counter.metrics()['mutual_information_grouped_counting_plugin'] < -60
    with pytest.raises(ValueError, match='cell_grouped_max_states'):
        GroupedCellMutualInformationLoss(grouping)


def test_grouped_kt_auto_budget_uses_groups_for_work_not_as_binary_alphabet():
    common = dict(n_receptors=100, entropy_type='grouped_kt_mi',
                  n_physics_receptors=1, mem_budget_bytes=2**30,
                  recompute_backward=True)
    two = resolve_batch_sizes(grouped_n_groups=2, grouped_n_states=51**2, **common)
    many = resolve_batch_sizes(grouped_n_groups=100, grouped_n_states=2**100, **common)
    assert two[0] > many[0]
    assert two[2] > 2**2  # J=2 groups are not two binary variables.
    with pytest.raises(ValueError, match='grouped_n_groups'):
        resolve_batch_sizes(n_receptors=100, entropy_type='grouped_kt_mi')


def test_runner_grouped_kt_counting_bypasses_guard_and_persists(tmp_path):
    cfg = _config(entropy='grouped_kt_mi', cell_receptors=None,
                  cell_gene_sets=((0,), (0,), (1,)), cell_readout='mean', epochs=1,
                  cell_grouped_max_states=1, final_test_batch_size=17, eval_chunk_size=6,
                  measurement_fns=('grouped_counting',),
                  final_measurement_fns=('grouped_counting', 'full_array_entropy',
                                         'mutual_information_kt', 'mutual_information_kt_upper'))
    runner = SimulationRunner(cfg, ExperimentLogger(str(tmp_path)))
    runner.run()
    assert runner.grouped_estimator is None
    results = json.loads((tmp_path / 'test_results.json').read_text())
    assert results['grouped_counting_samples'] == [17] * 10
    assert results['grouped_n_states'] == [6] * 10
    for h, hc, mi in zip(results['full_array_entropy'],
                         results['conditional_entropy_response'], results['mutual_information_kt']):
        assert h - hc == pytest.approx(mi)
    assert 'mutual_information_grouped' not in results


def test_streamed_kt_uses_full_budget_and_preserves_binary_bound(tmp_path, monkeypatch):
    cfg = _config(entropy='grouped_kt_mi', cell_receptors=None,
                  cell_gene_sets=((0,), (0,), (1,)), cell_readout='mean', eval_chunk_size=4,
                  measurement_fns=('full_array_entropy', 'mutual_information_kt',
                                   'mutual_information_kt_upper'))
    W = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    runner = SimulationRunner(cfg, ExperimentLogger(str(tmp_path)))
    loss = _build_loss(cfg, cell_weights=W)
    torch.manual_seed(9)
    a = torch.rand(11, 2)[:, torch.tensor([0, 0, 1])]
    requests = []
    def sample_batch(batch_size, return_dense_conc=False, **kwargs):
        start = sum(requests)
        requests.append(batch_size)
        result = (a[start:start + batch_size], None, None)
        return result + (None,) if return_dense_conc else result
    env = SimpleNamespace(use_interface_model=False, sample_batch=sample_batch)
    monkeypatch.setattr(runner, '_activity', lambda physics, E, *args, **kwargs: E)
    result = runner._eval_stats(env, None, loss, None, 11, 0)
    assert requests == [4, 4, 3]
    soft = torch.stack((1 - a, a), -1)
    assert result['full_array_entropy'] == pytest.approx(compute_kt_entropy(soft).item(), abs=1e-6)
    assert result['mutual_information_kt_upper'] == pytest.approx(compute_kt_upper_entropy(soft, return_mi=True).item(), abs=1e-6)


def test_scalable_script_guard_and_counting_budget_report(tmp_path, monkeypatch):
    from tasks.cells.gene_expression.scripts import replicates
    from tasks.cells.gene_expression.scripts.evaluation_budget import main as budget_main
    from tasks.cells.gene_expression.analysis._shared import load_study, summarize
    large = replicates.main(['--n_genes', '5', '--n_cells', '30', '--coverage', 'complete',
                             '--entropy', 'grouped_kt_mi', '--evaluation', 'counting',
                             '--max_states', '1', '--dry_run'])
    assert max(r['count_states'] for r in large['rows']) > 262144
    with pytest.raises(ValueError, match='count states'):
        replicates.main(['--n_genes', '5', '--n_cells', '30', '--coverage', 'complete',
                         '--entropy', 'grouped_kt_mi', '--evaluation', 'exact', '--dry_run'])
    replicates.main(['--n_genes', '2', '--n_cells', '4', '--coverage', 'complete',
                     '--entropy', 'grouped_kt_mi', '--evaluation', 'counting', '--max_states', '1',
                     '--epochs', '1', '--replicates', '1', '--batch_size', '8',
                     '--final_batch_size', '17', '--eval_chunk_size', '6', '--base_folder', str(tmp_path)])
    runs = load_study(tmp_path, 'replicates')
    assert len(runs) == 2
    assert (summarize(runs).mi_estimator == 'grouped counting (plug-in)').all()
    folder = runs.iloc[0].run_dir
    budget_main(['--run_dirs', folder, '--estimator', 'counting', '--budgets', '7', '19',
                 '--repeats', '2', '--chunk_size', '5', '--device', 'cpu'])
    from pathlib import Path
    report = json.loads(next(Path(folder).glob('grouped_evaluation_budget_*.json')).read_text())
    assert [r['grouped_counting_samples'] for r in report['records']] == [7, 19, 7, 19]
    assert report['arguments']['estimator'] == 'counting'


@pytest.mark.parametrize('settings', [dict(entropy='grouped_kt_mi'),
                                      dict(measurement_fns=('grouped_counting',))])
def test_grouped_options_reject_receptor_only_configs(settings):
    with pytest.raises(ValueError, match='cell mode'):
        _config(cell_receptors=None, **settings)
