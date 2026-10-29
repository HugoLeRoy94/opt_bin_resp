# %%
"""Check 3: the information itself, and the two regimes in R.

Only run this after checks 1 and 2. If the geometry is wrong, an MI agreement
here is a coincidence and a disagreement has two possible causes.

Measured quantity: codeword_entropy. The analytics assumes a deterministic
threshold readout (A3), so MI = H(A) exactly, and codeword_entropy is the
hard-binarised H(A). Miller-Madow is the less biased of its two estimates, but
neither is a bound: check the Good-Turing missing mass before believing any
point, and remember the hard ceiling log2 B.

Predictions, now stated at nu_eff rather than nu. Check 2 showed the directions are
zero-sum (mean cos = -1/(R-1)), not orthogonal, so assumption A5 is replaced by

    nu_eff^2 = (cos + nu^2)/(1 - cos)  ->  (nu^2 (R-1) - 1)/R   at the floor

The exchangeable reduction itself is untouched: the drive covariance stays of
one-factor form, only its two coefficients change. nu_eff -> nu as R grows, so the
correction is a SMALL-R effect, and it is the sign of the residual that tests it.

  R <= D   MI = R*eta(nu_eff) + I(A;q). The uncorrected nu is plotted too, so the
           gap between the two curves is visible.
  R <= 1 + 1/nu^2   the clip does not bind, the outputs can be made exactly
           independent, and MI = R bits exactly (T.R_lossless).
  R >  D   the exchangeable reduction degrades: pairs can no longer all share one
           cosine (Check 2, right panel), so MI must fall BELOW the extrapolated
           line and bend toward the geometric saturation O(D log2 R) of Section 13.
  collapse (MI - I(A;q))/R against nu_eff lands on one curve for every (R, D).
           This is the real claim: one control parameter, not two.
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
# A5 version (orthogonal directions) and the corrected version, side by side.
df["nu_eff"] = [T.nu_eff(nu, int(R)) for nu, R in zip(df["nu"], df["R"])]
df["MI_orth"] = [T.mi(nu, int(R))["MI"] for nu, R in zip(df["nu"], df["R"])]
df["MI_theory"] = [T.mi(nu, int(R))["MI"] for nu, R in zip(df["nu_eff"], df["R"])]
df["I_A_q"] = [T.mi(nu, int(R))["I_A_q"] for nu, R in zip(df["nu_eff"], df["R"])]
D = int(df["latent_dim"].iloc[0])

# Group on the SWEPT variable. nu is recomputed per run at the measured Emax, which
# drifts differently in every run, so grouping on round(nu, 2) produced ~67 groups
# and a legend with 67 entries. conc_std is the axis that was actually swept.
NU = df.groupby("conc_std")["nu"].mean().round(2).to_dict()
LAB = {cs: f"nu~{v:.2f}" for cs, v in NU.items()}

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
for cs, sub in ok.groupby("conc_std"):
    nu_g = NU[cs]
    a = sub.groupby("R")["MI"].agg(["mean", "std", "count"])
    line = ax.errorbar(a.index, a["mean"], yerr=a["std"], fmt="o", capsize=3,
                       label=LAB[cs])
    col = line.lines[0].get_color()
    th = [T.mi_eff(nu_g, int(r))["MI"] for r in rr]          # corrected
    th_o = [T.mi(nu_g, int(r))["MI"] for r in rr]            # old A5
    ax.plot(rr[rr <= D], np.array(th)[rr <= D], "-", color=col, lw=1.4)
    ax.plot(rr[rr >= D], np.array(th)[rr >= D], ":", color=col, lw=1.0)
    ax.plot(rr, th_o, "-", color=col, lw=.7, alpha=.45)      # faint: orthogonal
ax.plot(rr, rr, "k--", lw=.8, label="R bits (ideal)")
ax.axvline(D, color="grey", lw=.8)
ax.text(D * 1.02, .5, "R = D", rotation=90, fontsize=8, color="grey")
ax.set_xlabel("R"); ax.set_ylabel("MI  [bits]")
ax.set_title("Check 3: solid = theory where A5 holds, dotted = extrapolation past R=D")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

# %%
# The same thing as a residual, which is where a small systematic shows up.
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for cs, sub in ok.groupby("conc_std"):
    a = sub.groupby("R").apply(
        lambda s: pd.Series({"d": (s["MI"] - s["MI_theory"]).mean(),
                             "e": (s["MI"] - s["MI_theory"]).std(),
                             "o": (s["MI"] - s["MI_orth"]).mean()}),
        include_groups=False)
    line = ax.errorbar(a.index, a["d"], yerr=a["e"], fmt="o-", capsize=3,
                       label=LAB[cs])
    ax.plot(a.index, a["o"], "s--", ms=3, lw=.7, alpha=.5,
            color=line.lines[0].get_color())
ax.axhline(0, color="k", lw=.8)
ax.axvline(D, color="grey", lw=.8)
ax.set_xlabel("R"); ax.set_ylabel("measured - predicted  [bits]")
ax.set_title("residual: circles = corrected (nu_eff), faint squares = orthogonal.\n"
             "The orthogonal residual is positive and decays with R; that decay IS "
             "the simplex signature")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

# %%
# The collapse. This is the claim that nu is the only control parameter.
sub = ok[ok["R"] <= D].copy()
sub["eta_meas"] = (sub["MI"] - sub["I_A_q"]) / sub["R"]
fig, ax = plt.subplots(figsize=(5.8, 4.0))
for R, s in sub.groupby("R"):
    # x axis is nu_eff: with zero-sum directions that is the control parameter.
    # Plotting against raw nu smears the small-R points to the right.
    ax.plot(s["nu_eff"], s["eta_meas"], "o", ms=5, label=f"R={R}")
nug = np.logspace(-1, 1.2, 300)
ax.plot(nug, [T.eta(n) for n in nug], "k-", lw=1.4, label=r"$\eta(\nu)$, Eq. 48")
ax.plot(nug, T.eta_small_nu(nug), "r--", lw=.9, label=r"$1-\nu^2/\pi\ln 2$")
ax.plot(nug, T.eta_large_nu(nug), "b--", lw=.9, label=r"$1.04/\nu$")
ax.plot([1.0], [T.ETA_AT_ONE], "k*", ms=13, label=r"$\eta(1)=1/2\ln 2$")
ax.set_xscale("log"); ax.set_ylim(0, 1.05)
ax.set_xlabel(r"$\nu_{\rm eff}$"); ax.set_ylabel(r"$(\mathrm{MI}-I(A;q))\,/\,R$")
ax.set_title(r"Check 3b: collapse onto one curve vs $\nu_{\rm eff}$ (R <= D only)")
ax.legend(fontsize=7, ncol=2); plt.tight_layout(); plt.show()

# %%
# Past R = D the prediction is a change of SHAPE, not a value. Normalising by R
# makes the knee visible: flat below D, falling above.
fig, ax = plt.subplots(figsize=(5.8, 3.6))
for cs, s in ok.groupby("conc_std"):
    a = s.groupby("R")["MI"].mean()
    line, = ax.plot(a.index, a.values / a.index, "o-", label=LAB[cs])
    # the plateau is now R-dependent, since nu_eff depends on R
    ax.plot(a.index, [T.eta(T.nu_eff(NU[cs], int(r))) for r in a.index],
            ":", lw=.9, color=line.get_color())
ax.axvline(D, color="grey", lw=.8)
ax.set_xlabel("R"); ax.set_ylabel("MI / R   [bits per receptor]")
ax.set_title(r"Check 3c: bits per receptor; dotted = $\eta(\nu_{\rm eff}(R))$")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

# %%
# Estimator diagnostics, so a shortfall is never mistaken for physics.
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for cs, s in df.groupby("conc_std"):
    a = s.groupby("R")[[MISS, "MI"]].mean()
    axes[0].semilogy(a.index, np.maximum(a[MISS], 1e-12), "o-", label=LAB[cs])
    axes[1].plot(a.index, a["MI"], "o-", label=LAB[cs])
axes[0].axhline(1e-3, color="r", ls="--", lw=.9, label="usability threshold")
axes[0].set_ylabel("Good-Turing missing mass"); axes[0].set_xlabel("R")
axes[1].plot(df.groupby("R")[CEIL].mean(), "k--", lw=1, label=r"$\log_2 B$ ceiling")
axes[1].set_ylabel("MI [bits]"); axes[1].set_xlabel("R")
for a in axes:
    a.legend(fontsize=7)
fig.suptitle("estimator health: both panels must be comfortable before believing check 3",
             y=1.03)
plt.tight_layout(); plt.show()

# %%
