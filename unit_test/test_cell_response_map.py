"""Response maps must use the full pool and preserve the shared chemical coordinates."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import analysis_helper as ah
from src.cells import CellArray, CellReadout
from src.environment import LigandEnvironment, LogNormalConcentration
from src.physics import BinaryReceptor


@pytest.mark.parametrize("interface", [False, True])
def test_isolated_ligands_match_dense_mixtures(interface):
    torch.manual_seed(7)
    array = CellArray([[0, 1], [1, 2]], 5, use_interface_model=interface)
    env = LigandEnvironment(
        3, 2, LogNormalConcentration(4, 0., 1.), n_ligands=4,
        observation_noise_sigma=0., latent_dim=2, family_spread=.3,
        distribution_type="gaussian", avg_family_distance=1., n_presence_blocks=1,
        mu_sources=1., mu_ligands_per_source=1., use_interface_model=interface,
        affinity_kernel="gaussian", kernel_params=[1.],
    )
    physics = BinaryReceptor(3, 5, temperature=3.)
    readout = CellReadout(array.W, threshold=.1, temperature=.2, learnable_threshold=False)
    # Independent reference: four dense mixtures, each with a different ligand
    # present. Use the composition fast path, unlike the isolated-ligand helper.
    env.bind_receptors(array.receptor_indices)
    coords = env.ligand_latent[None].expand(4, -1, -1)
    energies = env._compute_energies(coords, array.receptor_indices)
    p = physics(energies, 2. * torch.eye(4), array.receptor_indices,
                composition=env.composition)
    expected = readout(p)
    actual = ah.cell_ligand_responses(env, physics, array.receptor_indices, readout, 2.)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert actual.shape == (4, 2)
    assert not actual.requires_grad
    with pytest.raises(ValueError, match="positive"):
        ah.cell_ligand_responses(env, physics, array.receptor_indices, readout, 0.)


def test_panels_reuse_coordinates_and_probability_scale(monkeypatch):
    def no_refitting(*args, **kwargs):
        raise AssertionError("A supplied embedding must not be refitted")
    monkeypatch.setattr(ah, "build_latent_umap", no_refitting)
    embedding = dict(families=np.array([[0., 0.]]), ligands=np.array([[1., 0.], [0., 1.]]),
                     receptors=np.array([[.3, .5]]),
                     samples=np.random.default_rng(0).normal(size=(30, 2)),
                     sample_labels=np.zeros(30, dtype=int), ligand_assignments=np.zeros(2, dtype=int),
                     latent_dim=3)
    base_fig, _ = ah.plot_latent_umap(None, None, embedding=embedding)
    response = np.array([[0., .2], [1., .8]])
    fig, axes = ah.plot_cell_response_umap(embedding, response, [[0], [0]])
    for cell, ax in enumerate(axes.flat):
        diamonds = next(c for c in ax.collections
                        if isinstance(c, PathCollection) and c.get_array() is not None)
        np.testing.assert_array_equal(diamonds.get_offsets(), embedding['ligands'])
        np.testing.assert_array_equal(diamonds.get_array(), response[:, cell])
        assert diamonds.get_clim() == (0, 1)
    assert axes[0, 0].get_xlim() == axes[0, 1].get_xlim()
    assert axes[0, 0].get_ylim() == axes[0, 1].get_ylim()
    plt.close(base_fig)
    plt.close(fig)
