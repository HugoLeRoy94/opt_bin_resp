"""Load analytic_baseline runs and pull out the geometry the analytics predicts.

Keeps the three analysis scripts from repeating the same checkpoint surgery.
Everything returned is in NATURAL log-concentration units.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import theory as T                                             # noqa: E402
from src.plotlib import load_runs, load_model                  # noqa: E402

GOAL = "analytic_baseline"


def load(goal: str = GOAL, **filters):
    """All complete runs of the task, newest sweeps included."""
    return load_runs(goal, **filters)


def _emax(env) -> np.ndarray:
    """Learned selectivity ceiling per gene, natural-log units.

    Goes through env.max_energies_raw(), which returns (n_genes,) whether the
    underlying parameter is per-gene or the single shared scalar of
    shared_unit_energies. Reading the parameter directly would break in shared
    mode, where no attribute of that name exists.
    """
    if hasattr(env, "max_energies_raw"):
        return F.softplus(env.max_energies_raw().detach()).cpu().numpy()
    return 0.5 * (F.softplus(env.max_energy_u_raw_plus.detach())
                  + F.softplus(env.max_energy_u_raw_minus.detach())).cpu().numpy()


def _ebase(env) -> np.ndarray:
    if hasattr(env, "base_energies"):
        return env.base_energies().detach().cpu().numpy()
    return 0.5 * (env.base_energy_u_plus.detach()
                  + env.base_energy_u_minus.detach()).cpu().numpy()


def geometry(run_row) -> dict:
    """Receptor geometry and the matching closed-form prediction for one run.

    run_row: a one-row DataFrame (or a frame filtered to one run), as accepted
    by src.plotlib.load_model.

    Returned keys that matter:
      U          (R,) distance of each homomer from the family centre
      dirs       (R,D) unit vectors from the centre to each homomer
      Emax       (R,) learned ceiling;  Ebase (R,) learned baseline
      pred_init  closed form evaluated at the INITIAL Emax (10)
      pred_meas  closed form evaluated at the MEASURED mean Emax
      exact_meas exact-route optimum at the measured Emax (no delta method)
    """
    env, physics, ri = load_model(run_row)
    cfg = run_row.iloc[0] if hasattr(run_row, "iloc") else run_row

    centre = env.family_latent.detach().cpu().numpy()[0]          # (D,)
    units = env.unit_latent.detach().cpu().numpy()                # (n_genes, D)

    # Homomers: receptor r is gene r repeated k_sub times, so the receptor
    # coordinate is just that gene's latent coordinate.
    ri_np = np.asarray(ri.cpu() if torch.is_tensor(ri) else ri)
    gene_of_receptor = ri_np[:, 0]
    assert np.all(ri_np == gene_of_receptor[:, None]), \
        "this task is homomer-only; found a mixed receptor"

    pos = units[gene_of_receptor] - centre                        # (R, D)
    U = np.linalg.norm(pos, axis=1)
    dirs = pos / np.maximum(U, 1e-12)[:, None]

    emax = _emax(env)[gene_of_receptor]
    ebase = _ebase(env)[gene_of_receptor]

    D = int(cfg["latent_dim"])
    sigma = float(cfg["family_spread"])
    lam = float(np.atleast_1d(cfg["kernel_params"])[0])
    sigma_c = float(cfg["conc_std"]) * T.LN10
    m = float(cfg["conc_mean"]) * T.LN10

    def _pred(Emax_val):
        p = T.Params(D=D, sigma=sigma, lam=lam, Emax=float(Emax_val),
                     sigma_c=sigma_c, m=m)
        o = T.lambert_optimum(p)
        o["Params"] = p
        o["eta"] = T.eta(o["nu_min"])
        return o

    pred_init = _pred(T.Params(D=D, sigma=sigma, lam=lam, Emax=10.0,
                               sigma_c=sigma_c, m=m).Emax)
    pred_meas = _pred(float(np.mean(emax)))

    return {
        "run_dir": str(run_row["path"].iloc[0]) if hasattr(run_row, "iloc") else None,
        "R": int(len(U)), "D": D, "sigma": sigma, "lam": lam,
        "sigma_c": sigma_c, "m": m, "conc_std": float(cfg["conc_std"]),
        "centre": centre, "pos": pos, "U": U, "dirs": dirs,
        "Emax": emax, "Ebase": ebase,
        "shared_energies": bool(getattr(env, "shared_unit_energies", False)),
        "Emax_drift": float(np.mean(emax) / 10.0 - 1.0),
        "zbar_meas": np.array([T.zbar_from_ebase(
            T.Params(D=D, sigma=sigma, lam=lam, Emax=float(e), sigma_c=sigma_c, m=m), b)
            for e, b in zip(emax, ebase)]),
        "pred_init": pred_init,
        "pred_meas": pred_meas,
    }


def exact_pred(g: dict) -> dict:
    """Exact-route optimum at the measured Emax. Slower; use when selfcons < 2."""
    p = T.Params(D=g["D"], sigma=g["sigma"], lam=g["lam"],
                 Emax=float(np.mean(g["Emax"])), sigma_c=g["sigma_c"], m=g["m"])
    return T.exact_optimum(p)


def welch_bound(R: int, D: int) -> float:
    """Smallest achievable maximum |cos| between R unit vectors in D dimensions.

    Zero for R <= D, where exact orthogonality is possible. Above D it is the
    floor the optimiser cannot beat, so the orthogonality check must compare
    against this and not against zero.
    """
    if R <= D:
        return 0.0
    return math.sqrt((R - D) / (D * (R - 1)))


def random_cos(D: int) -> float:
    """E|cos| between two independent uniform directions in D dimensions."""
    return math.sqrt(2.0 / (math.pi * D))
