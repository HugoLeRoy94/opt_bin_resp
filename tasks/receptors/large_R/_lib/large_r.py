"""Helpers for the large_R task.

Theory and run loading are reused from analytic_baseline/_lib. Natural-log
concentration units throughout.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_AB = Path(__file__).resolve().parents[2] / "analytic_baseline" / "_lib"
sys.path.insert(0, str(_AB))

import theory as T                                             # noqa: E402,F401
import runs as RU                                              # noqa: E402,F401

GOAL = "large_R"
RHO = math.sqrt(0.5)          # sigma sqrt(D) / lambda, as in analytic_baseline


def lam_for(D: int, sigma: float = 1.0, rho: float = RHO) -> float:
    """Kernel range giving the same geometry ratio rho at every D."""
    return sigma * math.sqrt(D) / rho


def log2_shell_bound(R: int, D: int) -> float:
    """Equal norms: R hyperplanes through one point of R^{D+1}."""
    return math.log2(2 * sum(math.comb(R - 1, i) for i in range(D + 1)))


def log2_ladder_bound(R: int, D: int) -> float:
    """Unequal norms: R hyperplanes in general position in R^{D+1}."""
    return math.log2(sum(math.comb(R, i) for i in range(D + 2)))


def c_law(p, U: float, ebase: float, n: int = 400_000, seed: int = 0) -> dict:
    """Law of c = tau/||l|| for a shell of radius U: interior sd and atom weights."""
    rng = np.random.default_rng(seed)
    l2 = p.sigma ** 2 * rng.chisquare(p.D, n)
    s = p.m + p.sigma_c * rng.standard_normal(n)
    arg = 1.0 - (s - ebase) / p.Emax
    with np.errstate(divide="ignore", invalid="ignore"):
        B = np.where(arg > 0, -p.lam ** 2 * np.log(arg), np.inf)
    c = (l2 - B + U ** 2) / (2 * U * np.sqrt(l2))
    inside = np.abs(c) < 1
    return {"sd_c": float(c[inside].std()), "mean_c": float(c[inside].mean()),
            "p_on": float(np.mean(c <= -1)), "p_off": float(np.mean(c >= 1))}
