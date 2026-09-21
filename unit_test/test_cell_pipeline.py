"""Small regressions for threshold-cell construction and numerical execution paths."""
import sys
import json
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cells import CellArray, CellReadout, cell_activity
from src.config import RunConfig, SingleRunConfig
from src.environment import LigandEnvironment, LogNormalConcentration
from src.IO import ExperimentLogger, SweepLogger, SweepLoader, SingleRunLoader
from src.physics import BinaryReceptor
from src.run import SimulationRunner
from tasks.cells.equivalence.scripts._shared import COMMON, RECEPTORS


def _config(**overrides):
    values = {k: v for k, v in COMMON.items() if k in SingleRunConfig.__dataclass_fields__}
    values.update(batch_size=8, test_batch_size=8, epochs=6,
                  initial_temperature=1.0, per_epoch_measure=False,
                  cell_receptors=tuple((r,) for r in RECEPTORS))
    values.update(overrides)
    return SingleRunConfig(**values)


@pytest.mark.parametrize("swept", [False, True])
def test_explicit_repertoire_config_roundtrip_and_logging(tmp_path, swept):
    # Ten cells with three pentamers each exceeded the old filename limit.
    uniform = (tuple(RECEPTORS[:3]),) * 10
    mixed = (tuple(RECEPTORS[:1]), tuple(RECEPTORS[:2]))
    repertoires = [uniform, mixed] if swept else uniform
    config = RunConfig(**{**COMMON, "base_folder": str(tmp_path),
                          "cell_receptors": repertoires})
    restored = RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored.cell_receptors == repertoires
    assert restored.is_sweep() == swept
    original_steps = next(config.generate_trajectories())
    restored_steps = next(restored.generate_trajectories())
    assert restored_steps == original_steps

    logger = SweepLogger(restored)
    expected_labels = ["receptors_per_cell_3", "receptors_per_cell_1-2"]
    for step, label in zip(restored_steps, expected_labels):
        run_logger = logger.get_run_logger(step, "20260921_120000")
        assert Path(run_logger.run_dir).parent.name == label
        assert SingleRunLoader(run_logger.run_dir).load_config() == step
    loaded = SweepLoader(logger.sweep_root)
    assert loaded.config.cell_receptors == repertoires
    assert len(list(loaded.iter_run_dirs())) == len(original_steps)


@pytest.mark.parametrize("interface", [False, True])
def test_runner_preserves_explicit_repertoires(tmp_path, interface):
    runner = SimulationRunner(_config(use_interface_model=interface), ExperimentLogger(str(tmp_path)))
    runner._initialize()
    assert runner.cell_array.pool_size == len(RECEPTORS)
    torch.testing.assert_close(runner.cell_array.W.cpu(), torch.eye(len(RECEPTORS)))
    assert runner.readout.mode == "threshold"


@pytest.mark.parametrize("interface", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 3])
def test_threshold_outputs_and_gradients_match_gather(interface, chunk_size):
    torch.manual_seed(12)
    array = CellArray([[0, 1], [1, 2]], 5, use_interface_model=interface)
    env = LigandEnvironment(
        n_genes=3, n_families=2, conc_model=LogNormalConcentration(4, 0., 1.),
        n_ligands=4, latent_dim=2, family_spread=.3, avg_family_distance=1.,
        n_presence_blocks=1, mu_sources=1., mu_ligands_per_source=1.,
        observation_noise_sigma=0., distribution_type="gaussian",
        affinity_kernel="gaussian", kernel_params=[1.], use_interface_model=interface,
    )
    physics = BinaryReceptor(3, 5, temperature=1.)
    readout = CellReadout(array.W, mode="threshold", threshold=.2,
                          temperature=.15, learnable_threshold=False)
    coords = torch.randn(7, 2, 2)
    concs = torch.full((7, 2), 10.)
    params = tuple(env.parameters())
    reference = cell_activity(
        physics, readout, env._compute_energies(coords, array.receptor_indices),
        concs, array.receptor_indices, pre_gathered=interface,
    )
    reference_grads = torch.autograd.grad(reference.square().sum(), params)
    env.bind_receptors(array.receptor_indices)
    actual = cell_activity(
        physics, readout, env._compute_energies(coords, array.receptor_indices),
        concs, array.receptor_indices, composition=env.composition,
        chunk_size=chunk_size, recompute=chunk_size is not None,
    )
    actual_grads = torch.autograd.grad(actual.square().sum(), params)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual_grads, reference_grads, atol=2e-6, rtol=2e-5)
    assert torch.cat([g.flatten() for g in actual_grads]).abs().sum() > 0


@pytest.mark.parametrize("epochs,split", [(6, .5), (2, .5), (1, .5)])
def test_final_cell_temperature_is_used_and_saved(tmp_path, monkeypatch, epochs, split):
    cfg = _config(epochs=epochs, cell_phase_split=split, cell_recalibrate_every=0,
                  cell_threshold=.2)
    runner = SimulationRunner(cfg, ExperimentLogger(str(tmp_path)))
    monkeypatch.setattr("src.cells.calibrate_cell_readout", lambda *a, **kw: {
        "drive_scale": .1, "theta": .2, "n_silent": 0, "n_saturated": 0,
    })
    seen_temperatures = []
    original_activity = runner._activity

    def activity(*a, **kw):
        seen_temperatures.append(runner.readout.temperature)
        return original_activity(*a, **kw)

    def final_test(env, physics, loss_fn, receptor_indices, n_samples):
        assert physics.temperature == cfg.temperature
        assert runner.readout.temperature == pytest.approx(.1 * cfg.cell_temperature)
        return {}

    monkeypatch.setattr(runner, "_activity", activity)
    monkeypatch.setattr(runner, "_test", final_test)
    runner.run()
    if epochs > 1:
        assert seen_temperatures[-1] == pytest.approx(.1 * cfg.cell_temperature)
    saved = torch.load(tmp_path / "best_model.pt", map_location="cpu", weights_only=True)
    assert saved["readout_temperature"] == pytest.approx(.1 * cfg.cell_temperature)
