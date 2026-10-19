"""Closed-form homomer-array predictions from analytics/homomer_array_mi.tex.

Gaussian (saturating) kernel throughout. Equation numbers refer to that document.
All energies are in NATURAL log-concentration units. The simulator stores
conc_mean / conc_std in log10, so convert with `from_log10`.

Two routes are provided and they must agree where both are valid:

  delta route  nu(w) from the delta-method expansion of ln Z (Eq. 32), optimum in
               closed form through Lambert W (Eq. 39). Valid while the
               self-consistency group of Eq. 41 is large.
  exact route  product quadrature over the two shared scalars (||l||^2, s),
               handling the saturation atoms of Step 1d exactly. No expansion of
               ln Z. Still assumes A4 (shared threshold independent of the
               private projections).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import lambertw, logsumexp, gammaln
from scipy.stats import norm, chi2
from scipy.optimize import brentq, minimize_scalar

LN10 = math.log(10.0)

_QCLIP_LO = 0.0
_QCLIP_HI = 1.0


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    """Environment and receptor parameters, natural-log energy units.

    D        latent dimension
    sigma    per-coordinate spread of the ligand cloud (RunConfig.family_spread)
    lam      affinity length scale (RunConfig.kernel_params[0])
    Emax     selectivity ceiling, ln units (softplus(max_energy_u_raw))
    sigma_c  std of ln c
    m        mean of ln c
    """
    D: int
    sigma: float
    lam: float
    Emax: float
    sigma_c: float
    m: float = 0.0

    @property
    def rho2(self) -> float:
        """(sigma sqrt(D) / lambda)^2 — cloud width over kernel range, squared."""
        return self.D * self.sigma ** 2 / self.lam ** 2

    @property
    def rho(self) -> float:
        return math.sqrt(self.rho2)

    @property
    def Gamma(self) -> float:
        """Emax / sigma_c — receptor dynamic range (Eq. 10)."""
        return self.Emax / self.sigma_c

    @property
    def P(self) -> float:
        """Radial coefficient of Eq. 32."""
        return self.rho2 / 2.0

    @property
    def Q(self) -> float:
        """Concentration coefficient of Eq. 32."""
        return self.D * math.exp(2 * self.rho2) / (4 * self.rho2 * self.Gamma ** 2)


def from_log10(conc_mean_log10: float, conc_std_log10: float) -> tuple[float, float]:
    """Simulator conc_mean / conc_std (log10) -> (m, sigma_c) in natural log."""
    return conc_mean_log10 * LN10, conc_std_log10 * LN10


def ebase_from_zbar(p: Params, zbar: float) -> float:
    """Invert zbar = 1 - (m - Ebase)/Emax  (Eq. 10)."""
    return p.m - p.Emax * (1.0 - zbar)


def zbar_from_ebase(p: Params, ebase: float) -> float:
    return 1.0 - (p.m - ebase) / p.Emax


# --------------------------------------------------------------------------- #
# delta route: nu(w) and the Lambert optimum
# --------------------------------------------------------------------------- #
def nu_of_w(p: Params, w: float | np.ndarray) -> float | np.ndarray:
    """Eq. 32. w = U^2 / lambda^2."""
    w = np.asarray(w, dtype=float)
    return np.sqrt((p.P + p.Q * np.exp(2 * w)) / w)


def nu_quadratic_of_w(p: Params, w: float | np.ndarray) -> float | np.ndarray:
    """Eq. 34, the quadratic-kernel limit. Monotone decreasing: no optimum."""
    w = np.asarray(w, dtype=float)
    return np.sqrt((p.P + p.Q * math.exp(-2 * p.rho2)) / w)


def lambert_optimum(p: Params) -> dict:
    """Eqs. 37, 39, 41. Optimal lever arm and the resulting nu, delta route."""
    A = p.P / (p.Q * math.e)
    Lam = float(np.real(lambertw(A)))
    w_star = (1.0 + Lam) / 2.0
    nu_min = math.sqrt(p.rho2 / Lam)
    zbar = math.exp(-(p.rho2 + w_star))
    return {
        "A": A,
        "Lambda": Lam,
        "w_star": w_star,
        "U_star": p.lam * math.sqrt(w_star),
        "nu_min": nu_min,
        "zbar": zbar,
        "Ebase": ebase_from_zbar(p, zbar),
        # Eq. 41: Gamma*zbar at the optimum, independent of Gamma.
        "selfcons": math.sqrt(Lam * p.D / (2 * p.rho2 ** 2)),
        "sat_prob": float(norm.cdf(-math.sqrt(Lam * p.D / (2 * p.rho2 ** 2)))),
        # asymptotic forms, Eq. 40
        # asymptotic forms are only meaningful for Gamma > 1
        "U_star_asym": p.lam * math.sqrt(math.log(p.Gamma)) if p.Gamma > 1 else float("nan"),
        "nu_min_asym": p.rho / math.sqrt(2 * math.log(p.Gamma)) if p.Gamma > 1 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# quadrature nodes for q
# --------------------------------------------------------------------------- #
def q_nodes_delta(nu: float, n: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """q = Phibar(nu z), z ~ N(0,1). Returns (q, normalised weights).

    A uniform grid in z with pdf weights, refined so that the kernel width 1/nu
    is always resolved. Gauss-Hermite is NOT used: its weights overflow above a
    few hundred nodes, silently returning NaN. Pass n to force a node count.
    """
    zmax = 12.0
    if n <= 0:
        h = min(2e-3, 5e-2 / max(float(nu), 1e-9))
        n = int(2 * zmax / h) + 1
        n = int(min(max(n, 4001), 600001))
    z = np.linspace(-zmax, zmax, n)
    w = norm.pdf(z)
    w = w / w.sum()
    return norm.sf(nu * z), w


def q_nodes_exact(p: Params, U: float, ebase: float,
                  n_r2: int = 200, n_s: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Exact route: product quadrature over (||l||^2, s), atoms handled explicitly.

    Implements the recipe of the tex Section "The exact route, without A7":
        Z <= 0  -> q = 1  (saturated on)
        Z >  1  -> q = 0  (silent)
        else    -> q = Phibar(tau/sigma), tau = (||l||^2 + lam^2 ln Z + U^2)/(2U)
    """
    # Gauss-Legendre in quantile space, so each marginal is integrated exactly.
    ur, wr = np.polynomial.legendre.leggauss(n_r2)
    ur = 0.5 * (ur + 1.0)
    wr = 0.5 * wr
    r2 = p.sigma ** 2 * chi2.ppf(ur, p.D)

    us, ws = np.polynomial.legendre.leggauss(n_s)
    us = 0.5 * (us + 1.0)
    ws = 0.5 * ws
    s = p.m + p.sigma_c * norm.ppf(us)

    R2, S = np.meshgrid(r2, s, indexing="ij")
    Wt = np.outer(wr, ws)

    Z = (ebase + p.Emax - S) / p.Emax
    q = np.empty_like(Z)

    on = Z <= 0.0
    off = Z > 1.0
    mid = ~(on | off)
    q[on] = 1.0
    q[off] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = (R2[mid] + p.lam ** 2 * np.log(Z[mid]) + U ** 2) / (2.0 * U)
    q[mid] = norm.sf(tau / p.sigma)

    return q.ravel(), (Wt / Wt.sum()).ravel()


# --------------------------------------------------------------------------- #
# entropy and mutual information from q nodes
# --------------------------------------------------------------------------- #
def _h2(q: np.ndarray) -> np.ndarray:
    """Binary entropy in bits, with h2(0) = h2(1) = 0."""
    q = np.asarray(q, dtype=float)
    out = np.zeros_like(q)
    m = (q > 0.0) & (q < 1.0)
    qq = q[m]
    out[m] = -(qq * np.log2(qq) + (1 - qq) * np.log2(1 - qq))
    return out


def eta_from_nodes(q: np.ndarray, w: np.ndarray) -> float:
    """Per-receptor information efficiency E[h2(q)] (Eq. 48)."""
    return float(np.sum(w * _h2(q)))


def eta(nu: float, n: int = 0) -> float:
    """eta(nu) for the delta route."""
    q, w = q_nodes_delta(nu, n)
    return eta_from_nodes(q, w)


def mean_firing(q: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * q))


def mi_from_nodes(q: np.ndarray, w: np.ndarray, R: int) -> dict:
    """Exact MI of the exchangeable array, Eqs. 44-46.

    Returns MI by both routes plus the count distribution, so the two can be
    cross-checked. Everything in log space: q^k (1-q)^(R-k) underflows for R>=40.
    """
    k = np.arange(R + 1)
    with np.errstate(divide="ignore"):
        lq = np.log(q)
        l1q = np.log1p(-q)
    lw = np.log(np.maximum(w, 1e-300))

    # log Pi_k = logsumexp_i [ log w_i + k log q_i + (R-k) log(1-q_i) ].
    # q = 0 and q = 1 occur exactly, at the grid edges and at the saturation
    # atoms of the exact route. There 0 * log(0) must be 0, not NaN, so the two
    # products are built with the zero-exponent rows masked out rather than
    # multiplied.
    A = np.zeros((R + 1, q.size))
    nzk = k > 0
    A[nzk] = k[nzk, None] * lq[None, :]
    B = np.zeros((R + 1, q.size))
    nzr = (R - k) > 0
    B[nzr] = (R - k)[nzr, None] * l1q[None, :]
    terms = lw[None, :] + A + B
    log_pi = logsumexp(terms, axis=1)

    log_binom = gammaln(R + 1) - gammaln(k + 1) - gammaln(R - k + 1)
    log_pk = log_binom + log_pi
    pk = np.exp(log_pk)
    pk = pk / pk.sum()

    nz = pk > 0
    H_K = float(-np.sum(pk[nz] * np.log2(pk[nz])))
    D_term = float(np.sum(pk * log_binom / math.log(2.0)))
    mi_route1 = H_K + D_term                      # Eq. 45

    eta_v = eta_from_nodes(q, w)
    mi_route2_IAq = mi_route1 - R * eta_v          # Eq. 46, solved for I(A;q)

    return {
        "MI": mi_route1,
        "H_K": H_K,
        "which_term": D_term,
        "eta": eta_v,
        "R_eta": R * eta_v,
        "I_A_q": mi_route2_IAq,
        "P_K": pk,
        "norm_err": float(abs(np.exp(logsumexp(log_pk)) - 1.0)),
    }


def mi(nu: float, R: int, n: int = 0) -> dict:
    """MI of an R-receptor balanced array at control parameter nu."""
    q, w = q_nodes_delta(nu, n)
    return mi_from_nodes(q, w, R)


# --------------------------------------------------------------------------- #
# exact route: balance and optimal lever arm by numerical search
# --------------------------------------------------------------------------- #
def balance_ebase(p: Params, U: float, n_r2: int = 120, n_s: int = 240,
                  lo: float = -200.0, hi: float = 200.0) -> float:
    """Solve E[q] = 1/2 for Ebase at fixed U, on the exact nodes (Eq. 26)."""
    def f(eb):
        q, w = q_nodes_exact(p, U, eb, n_r2, n_s)
        return mean_firing(q, w) - 0.5
    a = p.m + lo
    b = p.m + hi
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(f"balance not bracketed: f({a:.3g})={fa:.3g}, f({b:.3g})={fb:.3g}")
    return float(brentq(f, a, b, xtol=1e-10, rtol=1e-12))


def exact_optimum(p: Params, w_lo: float = 1e-3, w_hi: float = 12.0,
                  n_r2: int = 120, n_s: int = 240) -> dict:
    """Maximise eta over the lever arm on the exact nodes, re-balancing at each U.

    This is the prediction to use when the self-consistency group of Eq. 41 is
    not large, i.e. when rho is of order 1 or above.
    """
    def neg_eta(logw):
        w = math.exp(logw)
        U = p.lam * math.sqrt(w)
        eb = balance_ebase(p, U, n_r2, n_s)
        q, wt = q_nodes_exact(p, U, eb, n_r2, n_s)
        return -eta_from_nodes(q, wt)

    res = minimize_scalar(neg_eta, bounds=(math.log(w_lo), math.log(w_hi)),
                          method="bounded", options={"xatol": 1e-4})
    w_star = math.exp(res.x)
    U_star = p.lam * math.sqrt(w_star)
    eb = balance_ebase(p, U_star, n_r2, n_s)
    q, wt = q_nodes_exact(p, U_star, eb, n_r2, n_s)
    eta_v = eta_from_nodes(q, wt)
    # report the nu that would give this eta through the delta route
    try:
        nu_eff = brentq(lambda nu: eta(nu) - eta_v, 1e-4, 1e3)
    except ValueError:
        nu_eff = float("nan")
    return {
        "w_star": w_star,
        "U_star": U_star,
        "Ebase": eb,
        "zbar": zbar_from_ebase(p, eb),
        "eta": eta_v,
        "nu_eff": nu_eff,
        "mean_firing": mean_firing(q, wt),
        "sat_on": float(np.sum(wt[q >= 1.0])),
        "sat_off": float(np.sum(wt[q <= 0.0])),
    }


# --------------------------------------------------------------------------- #
# asymptotic laws, for plotting
# --------------------------------------------------------------------------- #
ETA_SMALL_NU_COEF = 1.0 / (math.pi * math.log(2.0))     # Eq. 51
ETA_LARGE_NU_COEF = norm.pdf(0.0) * 2.6060764896686517  # Eq. 54, phi(0) * A
ETA_AT_ONE = 1.0 / (2.0 * math.log(2.0))                # Eq. 52


def eta_small_nu(nu):
    return 1.0 - ETA_SMALL_NU_COEF * np.asarray(nu, dtype=float) ** 2


def eta_large_nu(nu):
    return ETA_LARGE_NU_COEF / np.asarray(nu, dtype=float)


def mi_predicted(nu: float, R: int, n: int = 0) -> float:
    """MI including the I(A;q) term, the quantity to compare with a measurement."""
    return mi(nu, R, n)["MI"]
