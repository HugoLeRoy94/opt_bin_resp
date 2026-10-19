#!/usr/bin/env python3
"""Tier 0: self-consistency of the closed form. CPU only, seconds, no src/ import.

Nothing downstream is trustworthy until this passes. Run:
    python3 tasks/receptors/analytic_baseline/scripts/selftest_theory.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_lib"))
import theory as T  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


print("\n=== eta(nu) ===")
check("eta(1) == 1/(2 ln 2)", abs(T.eta(1.0) - T.ETA_AT_ONE) < 1e-9,
      f"{T.eta(1.0):.12f} vs {T.ETA_AT_ONE:.12f}")
check("quadrature stable under refinement",
      abs(T.eta(1.0) - T.eta(1.0, 200001)) < 1e-9)
check("eta in [0,1]", all(0.0 <= T.eta(n) <= 1.0 for n in [0.01, 0.5, 1, 3, 10, 50]))
check("eta strictly decreasing",
      all(T.eta(a) > T.eta(b) for a, b in zip([.1, .5, 1, 2, 5], [.5, 1, 2, 5, 10])))
check("small-nu law within 1% for nu <= 0.2",
      all(abs(T.eta(n) / T.eta_small_nu(n) - 1) < 0.01 for n in [0.05, 0.1, 0.2]),
      f"nu=0.2: {T.eta(0.2):.6f} vs {T.eta_small_nu(0.2):.6f}")
check("large-nu law within 5% for nu >= 5",
      all(abs(T.eta(n) / T.eta_large_nu(n) - 1) < 0.05 for n in [5, 10, 30]),
      f"nu=5: {T.eta(5):.6f} vs {T.eta_large_nu(5):.6f}")

# The tex quotes A = 2.606 from a coarse Simpson rule. Recompute it.
from scipy.integrate import quad                                   # noqa: E402
from scipy.stats import norm                                       # noqa: E402
# T._h2 defines h2(0) = h2(1) = 0; an unguarded 0*log2(0) returns NaN out here.
A_hi = quad(lambda x: float(T._h2(np.array([norm.sf(x)]))[0]), -40, 40, limit=800)[0]
check("constant A matches the tex to 1e-6", abs(A_hi - 2.6060764896686517) < 1e-6,
      f"A = {A_hi:.10f}")

print("\n=== MI (count distribution) ===")
for nu in [0.01, 0.5, 1.0, 3.0, 10.0]:
    for R in [2, 6, 12, 20, 40]:
        d = T.mi(nu, R)
        check(f"nu={nu:<5} R={R:<3} normalised, MI<=R, routes agree",
              d["norm_err"] < 1e-10 and d["MI"] <= R + 1e-9
              and abs(d["MI"] - (d["R_eta"] + d["I_A_q"])) < 1e-9,
              f"MI={d['MI']:.4f}  R*eta={d['R_eta']:.4f}  I(A;q)={d['I_A_q']:.4f}")
check("MI -> R as nu -> 0", abs(T.mi(1e-3, 12)["MI"] - 12) < 1e-3)
check("MI -> ~1 bit as nu -> inf", abs(T.mi(1e4, 12)["MI"] - 1.0) < 0.15,
      f"MI(nu=1e4, R=12) = {T.mi(1e4, 12)['MI']:.4f}")

print("\n=== delta route vs exact route ===")
print("  the exact route drops the delta-method expansion of ln Z and keeps the")
print("  saturation atoms. Eq. 41 predicts they agree when selfcons >> 1.")
rows = [(10, 1, 5, 10, 1), (10, 1, 5, 100, 1), (8, 1, 4, 10, 1), (10, 1, 2, 10, 1)]
for D, sg, lam, Emax, sc in rows:
    p = T.Params(D=D, sigma=sg, lam=lam, Emax=Emax, sigma_c=sc)
    L, E = T.lambert_optimum(p), T.exact_optimum(p)
    dU = abs(E["U_star"] / L["U_star"] - 1)
    de = abs(E["eta"] / T.eta(L["nu_min"]) - 1)
    print(f"  D={D} lam={lam} Gamma={p.Gamma:.0f} rho={p.rho:.2f} selfcons={L['selfcons']:.2f}"
          f" | U*: {L['U_star']:.3f} vs {E['U_star']:.3f} ({100*dU:.1f}%)"
          f" | eta: {T.eta(L['nu_min']):.4f} vs {E['eta']:.4f} ({100*de:.1f}%)"
          f" | sat_on={E['sat_on']:.1e}")
    if L["selfcons"] > 3:
        check(f"  routes agree where selfcons>3 (D={D},lam={lam},G={p.Gamma:.0f})",
              dU < 0.05 and de < 0.02)
check("exact route balances to 1/2",
      abs(T.exact_optimum(T.Params(D=8, sigma=1, lam=4, Emax=10, sigma_c=1))
          ["mean_firing"] - 0.5) < 1e-6)

print("\n=== degeneracy of the quadratic kernel (Eq. 34) ===")
p = T.Params(D=8, sigma=1, lam=4, Emax=10, sigma_c=1)
w = np.array([0.1, 0.5, 1, 2, 4, 8, 16])
nq = T.nu_quadratic_of_w(p, w)
check("quadratic-kernel nu is monotone decreasing in w (no optimum)",
      bool(np.all(np.diff(nq) < 0)), f"nu(w) = {np.round(nq, 3)}")
ng = T.nu_of_w(p, w)
check("gaussian-kernel nu has an interior minimum",
      int(np.argmin(ng)) not in (0, len(w) - 1), f"nu(w) = {np.round(ng, 3)}")

print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
