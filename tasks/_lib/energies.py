"""Binding energies out of a trained environment, in natural-log units.

E_max is a fixed hyperparameter. E_base of a homomer is the mean of its two
faces, or the one shared scalar under shared_unit_energies.

The simulator stores concentrations in log10 while the analytics is in natural
log. Anything combining these with conc_mean or conc_std must convert; see
theory.from_log10. Getting it wrong rescales nu by 2.3.
"""
import numpy as np


def emax(env) -> np.ndarray:
    """Selectivity ceiling per gene, shape (n_genes,): the same fixed value."""
    return np.full(env.n_genes, env.max_energy.item())


def ebase(env) -> np.ndarray:
    """Base energy of each gene's homomer pocket, shape (n_genes,)."""
    e_plus, e_minus = env.face_base_energies()
    return (0.5 * (e_plus + e_minus)).detach().cpu().numpy()
