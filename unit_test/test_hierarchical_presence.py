"""
Tests for the hierarchical source→ligand presence sampler in LigandEnvironment.

Run with:
    cd opt_bin_resp && python -m pytest unit_test/test_hierarchical_presence.py -v
"""
import math
import json
import tempfile
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment import LigandEnvironment, LogNormalConcentration
from src.config import SingleRunConfig, RunConfig
from src.IO import SingleRunLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(n_ligands=40, n_presence_blocks=4,
              mu_sources=1.0, mu_ligands_per_source=1.0,
              n_genes=3, n_families=2, latent_dim=4):
    conc = LogNormalConcentration(n_ligands, init_mean=-6.0, init_scale=1.0)
    return LigandEnvironment(
        n_genes=n_genes,
        n_families=n_families,
        conc_model=conc,
        n_ligands=n_ligands,
        mu_sources=mu_sources,
        mu_ligands_per_source=mu_ligands_per_source,
        observation_noise_sigma=0.1,
        latent_dim=latent_dim,
        family_spread=0.5,
        distribution_type="gaussian",
        avg_family_distance=1.0,
        n_presence_blocks=n_presence_blocks,
        affinity_kernel="gaussian",
        kernel_params=[1.0],
    )


def _masks(env, B=2048):
    return env._sample_masks(B)


# ---------------------------------------------------------------------------
# Test 1 — No empty rows over a range of (mu_sources, mu_ligands_per_source)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mu_s,mu_l", [
    (1e-3, 1e-3),
    (1.0,  1.0),
    (5.0,  5.0),
    (0.5,  3.0),
    (3.0,  0.5),
])
def test_no_empty_rows(mu_s, mu_l):
    env = _make_env(n_ligands=40, n_presence_blocks=4,
                    mu_sources=mu_s, mu_ligands_per_source=mu_l)
    masks = _masks(env, B=2048)
    assert masks.sum(-1).min() >= 1, (
        f"Empty row found for mu_sources={mu_s}, mu_ligands_per_source={mu_l}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Singleton limit: both knobs → ~1e-3 gives P(S==1) ~ 1
# ---------------------------------------------------------------------------

def test_singleton_limit():
    env = _make_env(n_ligands=40, n_presence_blocks=40,
                    mu_sources=1e-3, mu_ligands_per_source=1e-3)
    masks = _masks(env, B=4096)
    row_sums = masks.sum(-1)
    mean_s = row_sums.mean().item()
    frac_singleton = (row_sums == 1).float().mean().item()
    assert mean_s < 1.1, f"mean S={mean_s:.3f}, expected ~1"
    assert frac_singleton > 0.95, f"P(S==1)={frac_singleton:.3f}, expected ~1"


# ---------------------------------------------------------------------------
# Test 3 — Monotonicity: mean S increases in each knob separately
# ---------------------------------------------------------------------------

def test_monotonicity_mu_sources():
    mus_vals = [0.5, 1.0, 2.0, 4.0]
    means = []
    for mu_s in mus_vals:
        env = _make_env(n_ligands=40, n_presence_blocks=8,
                        mu_sources=mu_s, mu_ligands_per_source=1.0)
        means.append(_masks(env, B=4096).sum(-1).float().mean().item())
    for i in range(len(means) - 1):
        assert means[i] < means[i + 1], (
            f"Non-monotone in mu_sources: {mus_vals[i]}→{mus_vals[i+1]}: "
            f"{means[i]:.2f}→{means[i+1]:.2f}"
        )


def test_monotonicity_mu_ligands():
    mul_vals = [0.5, 1.0, 2.0, 4.0]
    means = []
    for mu_l in mul_vals:
        env = _make_env(n_ligands=40, n_presence_blocks=4,
                        mu_sources=1.0, mu_ligands_per_source=mu_l)
        means.append(_masks(env, B=4096).sum(-1).float().mean().item())
    for i in range(len(means) - 1):
        assert means[i] < means[i + 1], (
            f"Non-monotone in mu_ligands_per_source: {mul_vals[i]}→{mul_vals[i+1]}: "
            f"{means[i]:.2f}→{means[i+1]:.2f}"
        )


# ---------------------------------------------------------------------------
# Test 4 — Within-block covariance > 0; cross-block ~0 (small negative allowed)
# ---------------------------------------------------------------------------

def test_block_correlation_structure():
    K = 4
    env = _make_env(n_ligands=40, n_presence_blocks=K,
                    mu_sources=1.5, mu_ligands_per_source=2.0)
    masks = _masks(env, B=8192).numpy()  # (B, L)
    block_ids = env.presence_block_id.numpy()  # (L,)

    # Pick two ligand pairs: one within block 0, one across blocks 0 and 1
    idx_b0 = (block_ids == 0).nonzero()[0]
    idx_b1 = (block_ids == 1).nonzero()[0]
    assert len(idx_b0) >= 2 and len(idx_b1) >= 1

    i0, i1 = int(idx_b0[0]), int(idx_b0[1])
    j0 = int(idx_b1[0])

    cov_within = float(
        ((masks[:, i0] - masks[:, i0].mean()) *
         (masks[:, i1] - masks[:, i1].mean())).mean()
    )
    cov_cross = float(
        ((masks[:, i0] - masks[:, i0].mean()) *
         (masks[:, j0] - masks[:, j0].mean())).mean()
    )

    assert cov_within > 0.0, f"Within-block cov={cov_within:.4f}, expected > 0"
    # Cross-block cov can be slightly negative (without-replacement over K sources)
    # but should be small in magnitude compared to within-block cov.
    assert abs(cov_cross) < cov_within, (
        f"|cross-block cov|={abs(cov_cross):.4f} >= within-block cov={cov_within:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Determinism under a fixed torch seed
# ---------------------------------------------------------------------------

def test_determinism():
    env = _make_env(n_ligands=20, n_presence_blocks=4,
                    mu_sources=1.5, mu_ligands_per_source=2.0)
    torch.manual_seed(42)
    m1 = env._sample_masks(64)
    torch.manual_seed(42)
    m2 = env._sample_masks(64)
    assert torch.equal(m1, m2), "Masks differ under the same torch seed"


# ---------------------------------------------------------------------------
# Test 6 — Old config.json (containing p_presence/rho_block, missing mu_*)
#           loads without error and yields valid defaults
# ---------------------------------------------------------------------------

def test_old_config_compat():
    # Build a minimal old-style config dict that has the removed fields but
    # lacks the new mu_* fields.
    old_cfg = {
        "n_families": 2,
        "n_ligands": 20,
        "latent_dim": 4,
        "family_spread": 0.5,
        "average_family_distance": 1.0,
        "environment_geometry": "default",
        "distribution_type": "gaussian",
        "observation_noise_sigma": 0.1,
        "n_presence_blocks": 2,
        "rho_block": 0.3,            # old field — must be silently dropped
        "block_shared_conc_mean": False,
        "conc_model_type": "lognormal",
        "conc_mean": [-6.0] * 20,
        "conc_std": [1.0] * 20,
        "p_presence": [0.5] * 20,    # old field — must be silently dropped
        "n_genes": 3,
        "k_sub": 5,
        "temperature": 1.0,
        "affinity_kernel": "gaussian",
        "kernel_params": [1.0],
        "batch_size": 256,
        "entropy": "renyi",
        "cov_weight": None,
        "penalty_type": None,
        "n_c_bins": 10,
        "epochs": 10,
        "lr": 1e-3,
        "use_scheduler": False,
        "test_batch_size": 512,
        "measurement_fns": [],
    }

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = os.path.join(tmp, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(old_cfg, f)
        loader = SingleRunLoader(tmp)
        cfg = loader.load_config()

    assert cfg.mu_sources == 1.0
    assert cfg.mu_ligands_per_source == 1.0
    assert not hasattr(cfg, "rho_block")
    assert not hasattr(cfg, "p_presence")
