"""Count grouping must preserve labeled entropy, MI, gradients, and receptor APIs."""
import json
import math
from types import SimpleNamespace

import pytest
import torch

from src.analysis_helper import full_array_entropy, identity_channel, concentration_channel
from src.bin_loss import (DiscreteExactLoss, KT_EPS, compute_shannon_joint_entropy,
                          compute_response_conditional_entropy)
from src.grouped_loss import GroupedCellMutualInformationLoss
from src.config import RunConfig
from src.IO import ExperimentLogger
from src.run import SimulationRunner, _build_loss, resolve_batch_sizes
from unit_test.test_cell_pipeline import _config


def exact_metrics(a):
    a = a.clamp(KT_EPS, 1 - KT_EPS)
    h = compute_shannon_joint_entropy(torch.stack((1 - a, a), -1))
    hc = compute_response_conditional_entropy(a)
    return h, hc, h - hc


@pytest.mark.parametrize('sizes', [(2, 5, 3), (3, 3, 4), (10,), (1, 1, 1)])
def test_grouped_matches_full_values_and_gradients(sizes):
    torch.manual_seed(29)
    counts = torch.tensor(sizes)
    group_ids = torch.arange(len(sizes)).repeat_interleave(counts)
    W = torch.eye(len(sizes))[group_ids]
    loss = GroupedCellMutualInformationLoss(W)
    base = torch.rand(9, len(sizes), dtype=torch.float64) * .98 + .01
    a = base[:, group_ids].clone().requires_grad_()
    h, hc, mi = exact_metrics(a)
    result = loss.compute_metrics(a)
    torch.testing.assert_close(result['response_entropy_grouped'], h)
    torch.testing.assert_close(result['conditional_entropy_response_grouped'], hc)
    torch.testing.assert_close(result['mutual_information_grouped'], mi)
    torch.testing.assert_close(loss.compute_entropy(a), h)
    torch.testing.assert_close(loss.compute_entropy(a, 'shannon'), h)
    # Includes partial gradients for individual duplicate columns, not just
    # gradients of shared parameters: averaging must distribute them correctly.
    for actual, reference in ((result['response_entropy_grouped'], h),
                              (result['conditional_entropy_response_grouped'], hc),
                              (-loss(a), mi)):
        torch.testing.assert_close(torch.autograd.grad(actual, a, retain_graph=True)[0],
                                   torch.autograd.grad(reference, a, retain_graph=True)[0])
    assert loss.n_states == math.prod(n + 1 for n in sizes)
    assert result['grouped_count_entropy'] <= math.log2(loss.n_states) + 1e-12


def test_two_cell_example_and_entropy_meanings():
    loss = GroupedCellMutualInformationLoss(torch.ones(2, 1))
    a = torch.tensor([[0., 0.], [.5, .5]], dtype=torch.float64)
    result = loss.compute_metrics(a)
    assert result['grouped_count_entropy'].item() == pytest.approx(1.29879494, abs=5e-5)
    assert result['response_entropy_grouped'].item() == pytest.approx(1.54879494, abs=5e-5)
    assert result['grouped_count_conditional_entropy'].item() == pytest.approx(.75, abs=5e-5)
    assert result['conditional_entropy_response_grouped'].item() == pytest.approx(1., abs=5e-5)
    assert result['grouped_label_entropy'].item() == pytest.approx(.25, abs=5e-5)
    assert result['mutual_information_grouped'].item() == pytest.approx(.54879494, abs=5e-5)
    assert full_array_entropy(a, loss)['full_array_entropy'] == pytest.approx(
        result['response_entropy_grouped'].item())
    # The unchanged receptor Shannon API still reports labeled entropy.
    receptor_loss = DiscreteExactLoss('shannon')
    torch.testing.assert_close(receptor_loss.compute_entropy(a.clamp(KT_EPS, 1-KT_EPS)),
                               loss.compute_entropy(a))


@pytest.mark.parametrize('p', [0., .2, .5, 1.])
def test_constant_probabilities_carry_no_information(p):
    loss = GroupedCellMutualInformationLoss(torch.ones(20, 1))
    a = torch.full((8, 20), p, dtype=torch.float64, requires_grad=True)
    value = loss(a)
    assert value.item() == pytest.approx(0, abs=1e-12)
    assert torch.autograd.grad(value, a)[0].abs().max() < 1e-12


def test_grouping_uses_weights_and_guards_allocation():
    W = torch.tensor([[.5, .5], [.5, .5], [.25, .75]])
    loss = GroupedCellMutualInformationLoss(W)
    assert loss.n_groups == 2
    assert loss.n_states == 6
    with pytest.raises(ValueError, match='cell_grouped_max_states'):
        GroupedCellMutualInformationLoss(torch.eye(40))
    with pytest.raises(ValueError, match='cell mode'):
        _config(cell_receptors=None, entropy='grouped_mi')
    with pytest.raises(ValueError, match='cell mode'):
        _config(cell_receptors=None, measurement_fns=('grouped_information',))


def test_grouped_conditionals_preserve_labeled_entropy():
    W = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    grouped = GroupedCellMutualInformationLoss(W)
    reference = DiscreteExactLoss('shannon')
    a = torch.tensor([[.1, .1, .7], [.4, .4, .3], [.9, .9, .2], [.7, .7, .8]],
                     dtype=torch.float64)
    masks = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]])
    assert identity_channel(a, masks, grouped) == pytest.approx(identity_channel(a, masks, reference))
    assert concentration_channel(a, masks, grouped) == pytest.approx(concentration_channel(a, masks, reference))


@pytest.mark.parametrize('training_loss', ['grouped_mi', 'kt_mi'])
def test_streamed_evaluation_combines_probabilities_before_entropy(tmp_path, monkeypatch, training_loss):
    cfg = _config(entropy=training_loss, cell_receptors=None,
                  cell_gene_sets=((0,), (0,), (1,)), cell_readout='mean',
                  batch_size=4, eval_chunk_size=4,
                  measurement_fns=('grouped_information', 'conditional_entropy_response',
                                   'full_array_entropy'))
    runner = SimulationRunner(cfg, ExperimentLogger(str(tmp_path)))
    W = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    runner.cell_array = SimpleNamespace(W=W)
    loss = _build_loss(cfg, cell_weights=W)
    torch.manual_seed(17)
    base = torch.rand(10, 2)
    activities = base[:, torch.tensor([0, 0, 1])]
    requests = []

    def sample_batch(batch_size, return_dense_conc=False, **kwargs):
        start = sum(requests)
        requests.append(batch_size)
        result = (activities[start:start + batch_size], None, None)
        return result + (None,) if return_dense_conc else result

    env = SimpleNamespace(use_interface_model=False, sample_batch=sample_batch)
    monkeypatch.setattr(runner, '_activity', lambda physics, E, *args, **kwargs: E)
    result = runner._eval_stats(env, None, loss, None, 10, 0)
    assert requests == [4, 4, 2]
    h, hc, mi = exact_metrics(activities.double())
    assert result['response_entropy_grouped'] == pytest.approx(h.item(), abs=2e-6)
    assert result['conditional_entropy_response_grouped'] == pytest.approx(hc.item(), abs=2e-6)
    assert result['mutual_information_grouped'] == pytest.approx(mi.item(), abs=2e-6)
    assert result['conditional_entropy_response'] == pytest.approx(hc.item(), abs=2e-6)
    assert result['response_evaluation_samples'] == 10
    if training_loss == 'grouped_mi':
        assert result['full_array_entropy'] == result['response_entropy_grouped']


@pytest.mark.parametrize('periodic', [False, True])
def test_runner_trains_and_persists_grouped_measurements(tmp_path, periodic):
    cfg = _config(entropy='grouped_mi', cell_receptors=None,
                  cell_gene_sets=((0,), (0,), (1,)), cell_readout='mean', epochs=2,
                  per_epoch_measure=periodic, measurement_fns=('full_array_entropy', 'grouped_information'),
                  final_measurement_fns=('grouped_information', 'full_array_entropy'),
                  final_test_batch_size=17, eval_chunk_size=6)
    runner = SimulationRunner(cfg, ExperimentLogger(str(tmp_path)))
    runner.run()
    result = json.loads((tmp_path / 'test_results.json').read_text())
    assert result['response_evaluation_samples'] == [17]   # one final measurement, not repeated
    assert result['grouped_n_states'] == [6]
    assert result['full_array_entropy'] == result['response_entropy_grouped']
    history = (tmp_path / 'stats.csv').read_text()
    assert 'train_mutual_information' in history
    assert 'train_entropy' not in history
    restored = RunConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert restored.entropy == 'grouped_mi'
    assert restored.cell_grouped_max_states == cfg.cell_grouped_max_states


def test_auto_budget_depends_on_count_alphabet():
    kwargs = dict(entropy_type='grouped_mi', grouped_n_states=101,
                  n_physics_receptors=1, mem_budget_bytes=1024**3)
    assert resolve_batch_sizes(n_receptors=100, **kwargs) == resolve_batch_sizes(n_receptors=10000, **kwargs)
    with pytest.raises(ValueError, match='grouped_n_states'):
        resolve_batch_sizes(n_receptors=3, entropy_type='grouped_mi')
