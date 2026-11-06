"""Helpers for the two_families task.

Theory and run loading are reused from analytic_baseline/_lib and large_R/_lib.
Natural-log concentration units throughout.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_REC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REC / "analytic_baseline" / "_lib"))
sys.path.insert(0, str(_REC / "large_R" / "_lib"))

import theory as T                                             # noqa: E402,F401
import runs as RU                                              # noqa: E402,F401
import large_r as LR                                           # noqa: E402,F401

GOAL = "two_families"


def h2(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return float(-p * math.log2(p) - (1 - p) * math.log2(1 - p))


def interp_G(R_known, G_known):
    """Single-family MI G(n) for any n >= 0: G(0) = 0, G(1) = 1 (one balanced
    receptor), measured values elsewhere, linear in log2 n in between."""
    R_known = np.asarray(R_known, float)
    G_known = np.asarray(G_known, float)
    keep = R_known > 1
    x = np.concatenate([[0.0], np.log2(R_known[keep])])
    y = np.concatenate([[1.0], G_known[keep]])
    order = np.argsort(x)
    x, y = x[order], y[order]

    def G(n):
        n = np.asarray(n, float)
        out = np.interp(np.log2(np.maximum(n, 1)), x, y, right=np.nan)
        return np.where(n == 0, 0.0, out)
    return G


def best_split(G, R: int, pi: float) -> tuple[int, float]:
    """R_1 maximising pi G(R_1) + (1-pi) G(R-R_1), and that maximum."""
    r1 = np.arange(R + 1)
    v = pi * G(r1) + (1 - pi) * G(R - r1)
    i = int(np.nanargmax(v))
    return int(r1[i]), float(v[i])


def array_of(env, ri) -> dict:
    """Receptor positions and energies of a homomer array, plus the ligand pool."""
    ri = np.asarray(ri.cpu() if hasattr(ri, "cpu") else ri)
    gene = ri[:, 0]
    assert np.all(ri == gene[:, None]), "homomer-only task"
    mu = env.concentration_model.mu.detach().cpu().numpy()
    sd = np.exp(env.concentration_model.log_sigma.detach().cpu().numpy())
    return {
        "pos": env.unit_latent.detach().cpu().numpy()[gene],
        "Ebase": RU._ebase(env)[gene],
        "Emax": RU._emax(env)[gene],
        "lam": float(np.atleast_1d(env.kernel_params)[0]),
        "centres": env.family_latent.detach().cpu().numpy(),
        "lig": env.ligand_latent.detach().cpu().numpy(),
        "fam": env.ligand_family_assignments.detach().cpu().numpy(),
        "m": float(mu.mean()) * T.LN10,
        "sigma_c": float(sd.mean()) * T.LN10,
    }


def _H(codes: np.ndarray) -> float:
    _, c = np.unique(codes, return_counts=True)
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p)))


def decompose(a: dict, n: int = 2 ** 20, chunk: int = 2 ** 16, seed: int = 0) -> dict:
    """Monte Carlo of the learned array with a hard threshold, one ligand per sniff.

    Returns H(A), H(F), H(A|F=f), H(F|A), the allocation (receptors nearest each
    centre), the saturation probability P(s > Ebase + Emax), and the firing
    probability of each group for ligands of the OTHER family.
    """
    rng = np.random.default_rng(seed)
    pos, lam2 = a["pos"], a["lam"] ** 2
    R = len(pos)
    ceil = a["Ebase"] + a["Emax"]
    group = np.argmin(((pos[:, None, :] - a["centres"][None]) ** 2).sum(-1), axis=1)
    nf = len(a["centres"])
    weights = (np.uint64(1) << np.arange(R, dtype=np.uint64))

    codes, fams = [], []
    cross_hits = np.zeros(nf)
    cross_n = np.zeros(nf)
    sat = 0
    for start in range(0, n, chunk):
        b = min(chunk, n - start)
        idx = rng.integers(0, len(a["lig"]), b)
        s = a["m"] + a["sigma_c"] * rng.standard_normal(b)
        d2 = ((a["lig"][idx, None, :] - pos[None]) ** 2).sum(-1)
        act = s[:, None] > a["Ebase"] + a["Emax"] * (1 - np.exp(-d2 / lam2))
        f = a["fam"][idx]
        codes.append((act.astype(np.uint64) * weights).sum(1))
        fams.append(f)
        sat += int(np.sum(s > ceil[0]))          # energies are shared
        for g in range(nf):
            other = f != g
            cross_hits[g] += act[np.ix_(other, group == g)].sum()
            cross_n[g] += other.sum() * np.sum(group == g)
    codes = np.concatenate(codes)
    fams = np.concatenate(fams)

    H_A = _H(codes)
    pi = np.array([np.mean(fams == g) for g in range(nf)])
    H_F = float(-np.sum(pi[pi > 0] * np.log2(pi[pi > 0])))
    # H(F|A) = sum_a P(a) H(F | A=a), from the family counts of each codeword
    _, inv = np.unique(codes, return_inverse=True)
    n_a = np.bincount(inv)
    post = np.stack([np.bincount(inv, fams == g, minlength=len(n_a)) for g in range(nf)]) / n_a
    with np.errstate(divide="ignore", invalid="ignore"):
        H_post = -np.nansum(np.where(post > 0, post * np.log2(post), 0.0), axis=0)
    H_F_given_A = float(np.sum(n_a / n * H_post))
    H_A_given_f = np.array([_H(codes[fams == g]) for g in range(nf)])
    return {
        "R": R, "H_A": H_A, "H_F": H_F, "pi": pi,
        "H_A_given_f": H_A_given_f,
        "H_F_given_A": H_F_given_A,
        "R_f": np.bincount(group, minlength=nf),
        "p_sat": sat / n,
        "p_cross": np.divide(cross_hits, cross_n, out=np.full(nf, np.nan), where=cross_n > 0),
    }
