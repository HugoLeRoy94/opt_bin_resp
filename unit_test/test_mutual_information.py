"""MI bounds, stochastic counting, and the runner's actual objective/evaluation."""
import json
from types import SimpleNamespace

import pytest
import torch

from src.bin_loss import (DiscreteExactLoss, KTMutualInformationLoss,
                          compute_kt_entropy, compute_kt_upper_entropy,
                          compute_response_conditional_entropy,
                          compute_shannon_joint_entropy)
from src.analysis_helper import count_sampled_responses, mutual_information_counting
from src.config import RunConfig, SingleRunConfig
from src.IO import ExperimentLogger
from src.run import SimulationRunner, _build_loss, resolve_batch_sizes
from tasks.cells.equivalence.scripts._shared import COMMON, RECEPTORS


def _config(**overrides):
    values = {k: v for k, v in COMMON.items() if k in SingleRunConfig.__dataclass_fields__}
    values.update(batch_size=8, test_batch_size=8, epochs=6,
                  initial_temperature=1.0, per_epoch_measure=False,
                  receptor_indices=RECEPTORS)
    values.update(overrides)
    return SingleRunConfig(**values)


def soft(a):
    return torch.stack((1 - a, a), dim=-1)


@pytest.mark.parametrize("p", [.2, .5, .8])
def test_input_independent_response_has_zero_information(p):
    a = torch.full((8, 3), p, dtype=torch.float64, requires_grad=True)
    mi_loss = KTMutualInformationLoss(collision_chunk_size=3)
    value = mi_loss(a)
    assert abs(value.item()) < 1e-12
    assert torch.autograd.grad(value, a)[0].abs().max() < 1e-12
    assert mi_loss.compute_entropy(a) > 2
    if p != .5:
        old_loss = DiscreteExactLoss(entropy_type='kt')(a)
        assert torch.autograd.grad(old_loss, a)[0].abs().sum() > 1


@pytest.mark.parametrize("chunk", [2, 32])
def test_kt_brackets_exact_mi_and_preserves_entropy_api(chunk):
    torch.manual_seed(31)
    a = .01 + .98 * torch.rand(13, 4, dtype=torch.float64)
    conditional = compute_response_conditional_entropy(a)
    exact = compute_shannon_joint_entropy(soft(a)) - conditional
    lower = compute_kt_entropy(soft(a), chunk_size=chunk, return_mi=True)
    upper = compute_kt_upper_entropy(soft(a), chunk_size=chunk, return_mi=True)
    assert 0 <= lower <= exact <= upper <= 4 - conditional + 1e-12
    torch.testing.assert_close(compute_kt_entropy(soft(a), chunk_size=chunk),
                               lower + conditional)
    torch.testing.assert_close(compute_kt_upper_entropy(soft(a), chunk_size=chunk),
                               upper + conditional)


def test_binary_symmetric_channel_and_noiseless_codes():
    a = torch.tensor([[.1], [.9]], dtype=torch.float64)
    exact = 1 - compute_response_conditional_entropy(a)
    assert compute_kt_entropy(soft(a), return_mi=True) <= exact
    assert exact <= compute_kt_upper_entropy(soft(a), return_mi=True)
    codes = ((torch.arange(8)[:, None] >> torch.arange(3)) & 1).double()
    assert compute_kt_entropy(soft(codes), return_mi=True).item() == pytest.approx(3, abs=.01)


def test_mi_gradients_checkpointing_and_optimization():
    torch.manual_seed(42)
    a = (.1 + .8 * torch.rand(5, 3, dtype=torch.float64)).requires_grad_()
    loss = KTMutualInformationLoss(collision_chunk_size=2)
    assert torch.autograd.gradcheck(loss, (a,))
    checkpointed = KTMutualInformationLoss(collision_chunk_size=2, recompute_backward=True)
    torch.testing.assert_close(checkpointed(a), loss(a))
    torch.testing.assert_close(torch.autograd.grad(checkpointed(a), a),
                               torch.autograd.grad(loss(a), a))
    logits = torch.tensor([[-.4], [.4]], requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=.15)
    initial = 1 - compute_response_conditional_entropy(logits.sigmoid()).item()
    for _ in range(40):  # optimization iterations, not a tensor-math loop
        optimizer.zero_grad()
        loss(logits.sigmoid()).backward()
        optimizer.step()
    actual_mi = 1 - compute_response_conditional_entropy(logits.sigmoid()).item()
    assert actual_mi > .9 and actual_mi > initial + .8


def test_counting_samples_probability_channel():
    torch.manual_seed(73)
    coins = mutual_information_counting(torch.full((32768, 3), .5))
    assert coins['response_entropy_plugin'] == pytest.approx(3, abs=.01)
    assert coins['mutual_information_counting_mm'] == pytest.approx(0, abs=.01)
    a = torch.tensor([[.1], [.9]]).repeat(16384, 1)
    result = mutual_information_counting(a)
    exact = 1 - compute_response_conditional_entropy(a).item()
    assert result['mutual_information_counting_mm'] == pytest.approx(exact, abs=.01)
    assert result['response_counting_samples'] == a.shape[0]


def test_counting_does_not_clip_undersampling_or_enumerate_the_alphabet():
    result = mutual_information_counting(torch.full((1, 1100), .5))
    assert result['mutual_information_counting_mm'] == -1100
    assert result['response_counting_K_hat'] == 1


def test_posthoc_counting_streams_chunks_and_reports_entropy_and_mi():
    """The reloaded-checkpoint path has the same counting semantics as the runner."""
    requests = []

    def sample_batch(batch_size, **kwargs):
        requests.append(batch_size)
        return torch.full((batch_size, 3), .5), None, None

    env = SimpleNamespace(use_interface_model=False, sample_batch=sample_batch)
    result = count_sampled_responses(
        env, lambda energies, *args, **kwargs: energies, None,
        n_samples=32768, fwd_chunk=4096,
    )
    assert requests == [4096] * 8
    assert result['response_counting_samples'] == 32768
    assert result['response_entropy_mm'] == pytest.approx(3, abs=.02)
    assert result['mutual_information_counting_mm'] == pytest.approx(0, abs=.02)


@pytest.mark.parametrize("counting_only", [False, True])
def test_full_batch_metrics_share_samples_and_keep_world_rng(tmp_path, monkeypatch, counting_only):
    names = ('mutual_information_counting',) if counting_only else (
        'entropy_kt', 'entropy_kt_upper', 'mutual_information_kt',
        'mutual_information_kt_upper', 'conditional_entropy_response',
        'codeword_entropy', 'mutual_information_counting')
    cfg = _config(entropy='kt_mi', measurement_fns=names, eval_chunk_size=4)
    runner = SimulationRunner(cfg, ExperimentLogger(str(tmp_path)))
    activities = torch.linspace(.01, .97, 30).reshape(10, 3)
    requests = []

    def sample_batch(batch_size, return_dense_conc=False, **kwargs):
        start = sum(requests)
        requests.append(batch_size)
        output = (activities[start:start + batch_size], None, None)
        return output + (None,) if return_dense_conc else output

    env = SimpleNamespace(use_interface_model=False, sample_batch=sample_batch)
    monkeypatch.setattr(runner, '_activity', lambda physics, E, *a, **kw: E)
    if counting_only:
        def no_kt(*a, **kw):
            pytest.fail("Counting-only evaluation must not compute pairwise KT")
        monkeypatch.setattr('src.run.compute_kt_entropy', no_kt)
        monkeypatch.setattr('src.run.compute_kt_upper_entropy', no_kt)
    rng_before = torch.get_rng_state()
    result = runner._eval_stats(env, None, _build_loss(cfg), None, 10, 0)
    torch.testing.assert_close(torch.get_rng_state(), rng_before)
    assert requests == [4, 4, 2]
    assert result['response_counting_samples'] == result['response_evaluation_samples'] == 10
    conditional = compute_response_conditional_entropy(activities).item()
    assert result['conditional_entropy_response'] == pytest.approx(conditional)
    if not counting_only:
        for suffix, fn in [('', compute_kt_entropy), ('_upper', compute_kt_upper_entropy)]:
            mi = fn(soft(activities), return_mi=True).item()
            assert result['mutual_information_kt' + suffix] == pytest.approx(mi)
            assert result['full_array_entropy_kt' + suffix] == pytest.approx(mi + conditional)


@pytest.mark.parametrize('periodic', [False, True])
def test_runner_trains_mi_and_honors_final_only_counting(tmp_path, periodic):
    cfg = _config(entropy='kt_mi', epochs=2, cell_recalibrate_every=0,
                  per_epoch_measure=periodic,
                  measurement_fns=('mutual_information_kt',),
                  final_measurement_fns=('mutual_information_counting',),
                  final_test_batch_size=17)
    assert isinstance(_build_loss(cfg), KTMutualInformationLoss)
    legacy = _build_loss(_config(entropy='kt'))
    assert not isinstance(legacy, KTMutualInformationLoss)
    a = torch.full((8, 3), .5)
    assert _build_loss(cfg)(a).item() == pytest.approx(0, abs=1e-6)
    assert legacy(a).item() == pytest.approx(-3)
    runner = SimulationRunner(cfg, ExperimentLogger(str(tmp_path)))
    runner.run()
    result = json.loads((tmp_path / 'test_results.json').read_text())
    assert result['response_counting_samples'] == [17] * 10
    assert 'mutual_information_counting_mm' in result
    assert 'mutual_information_kt' not in result
    history = (tmp_path / 'stats.csv').read_text()
    assert 'train_mutual_information' in history
    assert 'train_entropy' not in history


@pytest.mark.parametrize('recompute', [False, True])
def test_automatic_budget_matches_kt(recompute):
    kwargs = dict(n_receptors=20, n_physics_receptors=100,
                  n_ligands=8, k_sub=5, mem_budget_bytes=1024**3,
                  recompute_backward=recompute)
    assert resolve_batch_sizes(entropy_type='kt_mi', **kwargs) == resolve_batch_sizes(
        entropy_type='kt', **kwargs)


def test_final_configuration_round_trip():
    cfg = RunConfig.from_dict(_config(final_test_batch_size=123).to_dict())
    restored = RunConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert restored.final_measurement_fns == cfg.final_measurement_fns
    assert restored.final_test_batch_size == 123
    with pytest.raises(ValueError, match='final_test_batch_size'):
        _config(final_test_batch_size=0)
