# %%
"""Check 3: the information itself, and the two regimes in R.

Only run this after checks 1 and 2. If the geometry is wrong, an MI agreement
here is a coincidence and a disagreement has two possible causes.

Measured quantity: codeword_entropy. The analytics assumes a deterministic
threshold readout (A3), so MI = H(A) exactly, and codeword_entropy is the
hard-binarised H(A). Miller-Madow is the less biased of its two estimates, but
neither is a bound: check the Good-Turing missing mass before believing any
point, and remember the hard ceiling log2 B.

Predictions:
  R <= D   MI = R*eta(nu) + I(A;q),  Eq. 48; eta from Eq. 48 at the run's own nu
  R >  D   the exchangeable reduction fails (A5). MI must fall BELOW the
           extrapolated line and bend toward the geometric saturation
           O(D log2 R) of Section 13
  collapse (MI - I(A;q))/R against nu lands on one curve for every (R, D).
           This is the real claim: nu is the only control parameter.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TASK / "_lib"))
import runs as RU                                              # noqa: E402
import theory as T                                             # noqa: E402

MI_COL = "codeword_entropy_mm_mean"
MI_ALT = "codeword_entropy_plugin_mean"
MISS = "codeword_entropy_missing_mass_mean"
CEIL = "codeword_entropy_log2B_mean"

# %%
df = RU.load()
missing = [c for c in (MI_COL, MISS, CEIL) if c not in df.columns]
if missing:
    raise SystemExit(f"missing columns {missing}: rerun the sweep with "
                     f"'codeword_entropy' in final_measurement_fns")

# per-run predicted nu, taken at the MEASURED Emax (it is learnable)
rows = []
for path, grp in df.groupby("path"):
    g = RU.geometry(grp)
    pm = g["pred_meas"]
    rows.append(dict(path=path, nu=pm["nu_min"], eta_pred=pm["eta"],
                     selfcons=pm["selfcons"], Emax_drift=g["Emax_drift"]))
P = pd.DataFrame(rows)
df = df.merge(P, on="path")
df["MI"] = df[MI_COL]
df["MI_theory"] = [T.mi(nu, int(R))["MI"] for nu, R in zip(df["nu"], df["R"])]
df["I_A_q"] = [T.mi(nu, int(R))["I_A_q"] for nu, R in zip(df["nu"], df["R"])]
D = int(df["latent_dim"].iloc[0])

# %%
# Estimator health first. A point whose missing mass is above 1e-3, or that sits
# within 2 bits of log2 B, is not a measurement of anything.
bad = df[(df[MISS] > 1e-3) | (df["MI"] > df[CEIL] - 2)]
print(f"{len(bad)} of {len(df)} runs flagged as unreliable estimates")
if len(bad):
    print(bad.groupby("R")[[MISS, CEIL, "MI"]].mean().round(4))
ok = df.drop(bad.index)

# %%
# MI against R, one line per nu, with theory.
fig, ax = plt.subplots(figsize=(6.2, 4.2))
rr = np.arange(1, df["R"].max() + 1)
for nu_g, sub in ok.groupby(np.round(ok["nu"], 2)):
    a = sub.groupby("R")["MI"].agg(["mean", "std", "count"])
    line = ax.errorbar(a.index, a["mean"], yerr=a["std"], fmt="o", capsize=3,
                       label=f"nu={nu_g:.2f}")
    col = line.lines[0].get_color()
    th = [T.mi(nu_g, int(r))["MI"] for r in rr]
    ax.plot(rr[rr <= D], np.array(th)[rr <= D], "-", color=col, lw=1.2)
    ax.plot(rr[rr >= D], np.array(th)[rr >= D], ":", color=col, lw=1.0)
ax.plot(rr, rr, "k--", lw=.8, label="R bits (ideal)")
ax.axvline(D, color="grey", lw=.8)
ax.text(D * 1.02, .5, "R = D", rotation=90, fontsize=8, color="grey")
ax.set_xlabel("R"); ax.set_ylabel("MI  [bits]")
ax.set_title("Check 3: solid = theory where A5 holds, dotted = extrapolation past R=D")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

# %%
# The same thing as a residual, which is where a small systematic shows up.
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for nu_g, sub in ok.groupby(np.round(ok["nu"], 2)):
    a = sub.groupby("R").apply(
        lambda s: pd.Series({"d": (s["MI"] - s["MI_theory"]).mean(),
                             "e": (s["MI"] - s["MI_theory"]).std()}),
        include_groups=False)
    ax.errorbar(a.index, a["d"], yerr=a["e"], fmt="o-", capsize=3,
                label=f"nu={nu_g:.2f}")
ax.axhline(0, color="k", lw=.8)
ax.axvline(D, color="grey", lw=.8)
ax.set_xlabel("R"); ax.set_ylabel("measured - predicted  [bits]")
ax.set_title("residual; it must turn negative past R = D as A5 fails")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

# %%
# The collapse. This is the claim that nu is the only control parameter.
sub = ok[ok["R"] <= D].copy()
sub["eta_meas"] = (sub["MI"] - sub["I_A_q"]) / sub["R"]
fig, ax = plt.subplots(figsize=(5.8, 4.0))
for R, s in sub.groupby("R"):
    ax.plot(s["nu"], s["eta_meas"], "o", ms=5, label=f"R={R}")
nug = np.logspace(-1, 1.2, 300)
ax.plot(nug, [T.eta(n) for n in nug], "k-", lw=1.4, label=r"$\eta(\nu)$, Eq. 48")
ax.plot(nug, T.eta_small_nu(nug), "r--", lw=.9, label=r"$1-\nu^2/\pi\ln 2$")
ax.plot(nug, T.eta_large_nu(nug), "b--", lw=.9, label=r"$1.04/\nu$")
ax.plot([1.0], [T.ETA_AT_ONE], "k*", ms=13, label=r"$\eta(1)=1/2\ln 2$")
ax.set_xscale("log"); ax.set_ylim(0, 1.05)
ax.set_xlabel(r"$\nu$"); ax.set_ylabel(r"$(\mathrm{MI}-I(A;q))\,/\,R$")
ax.set_title("Check 3b: collapse onto one curve (R <= D only)")
ax.legend(fontsize=7, ncol=2); plt.tight_layout(); plt.show()

# %%
# Past R = D the prediction is a change of SHAPE, not a value. Normalising by R
# makes the knee visible: flat below D, falling above.
fig, ax = plt.subplots(figsize=(5.8, 3.6))
for nu_g, s in ok.groupby(np.round(ok["nu"], 2)):
    a = s.groupby("R")["MI"].mean()
    ax.plot(a.index, a.values / a.index, "o-", label=f"nu={nu_g:.2f}")
    ax.axhline(T.eta(nu_g), color=ax.lines[-1].get_color(), ls=":", lw=.8)
ax.axvline(D, color="grey", lw=.8)
ax.set_xlabel("R"); ax.set_ylabel("MI / R   [bits per receptor]")
ax.set_title("Check 3c: bits per receptor; dotted = eta(nu), the R<=D plateau")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

# %%
# Estimator diagnostics, so a shortfall is never mistaken for physics.
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for nu_g, s in df.groupby(np.round(df["nu"], 2)):
    a = s.groupby("R")[[MISS, "MI"]].mean()
    axes[0].semilogy(a.index, np.maximum(a[MISS], 1e-12), "o-", label=f"nu={nu_g:.2f}")
    axes[1].plot(a.index, a["MI"], "o-", label=f"nu={nu_g:.2f}")
axes[0].axhline(1e-3, color="r", ls="--", lw=.9, label="usability threshold")
axes[0].set_ylabel("Good-Turing missing mass"); axes[0].set_xlabel("R")
axes[1].plot(df.groupby("R")[CEIL].mean(), "k--", lw=1, label=r"$\log_2 B$ ceiling")
axes[1].set_ylabel("MI [bits]"); axes[1].set_xlabel("R")
for a in axes:
    a.legend(fontsize=7)
fig.suptitle("estimator health: both panels must be comfortable before believing check 3",
             y=1.03)
plt.tight_layout(); plt.show()
