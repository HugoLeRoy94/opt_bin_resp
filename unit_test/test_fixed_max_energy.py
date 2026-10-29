"""fixed_max_energy: E_max is a constant buffer, never a trainable parameter."""
import math

import pytest
import torch
import torch.nn.functional as F

from src.environment import LigandEnvironment, LogNormalConcentration


def _make_env(fixed=7.5, shared=True, n_genes=3, kernel="gaussian"):
    conc = LogNormalConcentration(20, init_mean=0.0, init_scale=1.0)
    return LigandEnvironment(
        n_genes=n_genes, n_families=1, conc_model=conc, n_ligands=20,
        observation_noise_sigma=0.0, latent_dim=4, family_spread=1.0,
        distribution_type="gaussian", avg_family_distance=1.0, n_presence_blocks=1,
        mu_sources=1.0, mu_ligands_per_source=1.0, affinity_kernel=kernel,
        kernel_params=[4.0], shared_unit_energies=shared, fixed_max_energy=fixed)


@pytest.mark.parametrize("shared", [True, False])
def test_value_and_not_trainable(shared):
    env = _make_env(shared=shared)
    emax = F.softplus(env.max_energies_raw())
    assert emax.shape == (3,)
    assert torch.allclose(emax, torch.full((3,), 7.5), atol=1e-5)
    assert all(p is not env.max_energy_u_raw_fixed for p in env.parameters())
    assert not any("max_energy" in n for n, _ in env.named_parameters())


def test_survives_training_and_reload():
    env = _make_env()
    opt = torch.optim.Adam(env.parameters(), lr=0.1)
    for _ in range(5):
        opt.zero_grad()
        F.softplus(env.max_energies_raw()).sum().mul(env.base_energies().sum()).backward()
        opt.step()
    assert torch.allclose(F.softplus(env.max_energies_raw()), torch.full((3,), 7.5), atol=1e-5)
    env2 = _make_env()
    env2.load_state_dict(env.state_dict())
    assert torch.allclose(env2.max_energies_raw(), env.max_energies_raw())


@pytest.mark.parametrize("shared", [True, False])
def test_clone_keeps_fixed(shared):
    new = _make_env(shared=shared).clone_with_extra_units(2)
    assert new.fixed_max_energy == 7.5
    assert torch.allclose(F.softplus(new.max_energies_raw()), torch.full((5,), 7.5), atol=1e-5)


def test_rejects_quadratic():
    with pytest.raises(NotImplementedError):
        _make_env(kernel="quadratic")


def test_default_still_learnable():
    env = _make_env(fixed=None)
    assert any("max_energy" in n for n, _ in env.named_parameters())
