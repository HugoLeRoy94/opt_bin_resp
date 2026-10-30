# %%
"""large_R: the small-R theory and the large-R expansion against simulation.

Predictions, for R >> D (README):
  MI vs R      below the shell bound (~D log2 R) if the norms stay equal, up to
               the ladder bound (~(D+1) log2 R) if they spread. The expansion
               theory.mi_large_R describes a shell with uniform directions.
  slope        dMI/dlog2R tends to D (shell) or D+1 (ladder).
  norms        std(U)/<U> grows with R once a shell wastes a dimension.
  operating    the expansion is maximal for sd(c) ~ 1/D, i.e. nu ~ 1/sqrt(D).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Anchored on tasks/receptors, not on this file: a shared interactive kernel can
# keep __file__ from another task's script.
_RECEPTORS = next(p for p in Path(__file__).resolve().parents if p.name == "receptors")
sys.path.insert(0, str(_RECEPTORS / "large_R" / "_lib"))
import large_r as LR                                           # noqa: E402

T, RU = LR.T, LR.RU
MI_COL = "codeword_entropy_mm_mean"
MISS = "codeword_entropy_missing_mass_mean"
CEIL = "codeword_entropy_log2B_mean"

# %%
df = RU.load(LR.GOAL)
rows = []
for path, grp in df.groupby("path"):
    g = RU.geometry(grp)
    r = grp.iloc[0]
    D, R = g["D"], g["R"]
    p = T.Params(D=D, sigma=g["sigma"], lam=g["lam"], Emax=float(np.mean(g["Emax"])),
                 sigma_c=g["sigma_c"], m=g["m"])
    U, eb = float(g["U"].mean()), float(np.mean(g["Ebase"]))
    C = g["dirs"] @ g["dirs"].T
    cl = LR.c_law(p, U, eb)
    rows.append(dict(path=path, D=D, R=R, conc_std=g["conc_std"],
                     MI=r[MI_COL], miss=r[MISS], ceil=r[CEIL],
                     U=U, Ucv=float(g["U"].std() / U), Ebase=eb,
                     cos_mean=float(C[np.triu_indices(R, 1)].mean()) if R > 1 else np.nan,
                     zero_sum=float(np.linalg.norm(g["dirs"].sum(0)) / R),
                     sd_c=cl["sd_c"], p_atoms=cl["p_on"] + cl["p_off"],
                     params=p))
X = pd.DataFrame(rows)

bad = X[(X["miss"] > 1e-3) | (X["MI"] > X["ceil"] - 2)]
print(f"{len(bad)} of {len(X)} runs flagged as unreliable estimates")
if len(bad):
    print(bad.groupby(["D", "R"])[["miss", "ceil", "MI"]].mean().round(4))
ok = X.drop(bad.index)
DS = sorted(X["D"].unique())
CS = sorted(X["conc_std"].unique())
COL = dict(zip(CS, plt.rcParams["axes.prop_cycle"].by_key()["color"]))

# %%
# MI vs R, one panel per D. Colour = conc_std, line style = which theory.
#   solid   small-R closed form (simplex, nu_eff), valid for R <= D+1
#   dotted  the same formula extrapolated past R = D+1
#   dashed  large-R expansion at U*, valid for R >> D^2
fig, axes = plt.subplots(1, len(DS), figsize=(6.2 * len(DS), 4.4), squeeze=False)
for ax, D in zip(axes[0], DS):
    sub = ok[ok["D"] == D]
    rr = np.arange(2, X["R"].max() + 1)
    top = sub["MI"].max()
    for cs, s in sub.groupby("conc_std"):
        col = COL[cs]
        a = s.groupby("R")["MI"].agg(["mean", "std"])
        ax.errorbar(a.index, a["mean"], yerr=a["std"].fillna(0), fmt="o", color=col,
                    ms=6, capsize=3, zorder=3)
        p0 = s["params"].iloc[0]
        o = T.lambert_optimum(p0)
        small = np.array([T.mi_eff(o["nu_min"], int(r))["MI"] for r in rr])
        ax.plot(rr[rr <= D + 1], small[rr <= D + 1], "-", color=col, lw=2.2)
        ax.plot(rr[rr >= D + 1], small[rr >= D + 1], ":", color=col, lw=1.5)
        big = T.mi_large_R(p0, o["U_star"], o["Ebase"], rr)
        ax.plot(rr, big, "--", color=col, lw=1.5)
        top = max(top, big.max())
        ax.plot([], [], "s", color=col, label=f"conc_std = {cs:g}  (nu = {o['nu_min']:.2f})")
    ax.plot([], [], "ko", label="simulation")
    ax.plot([], [], "k-", lw=2.2, label="small-R theory (R <= D+1)")
    ax.plot([], [], "k:", lw=1.5, label="small-R theory, extrapolated")
    ax.plot([], [], "k--", lw=1.5, label="large-R expansion")
    ax.axvline(D ** 2, color="grey", lw=1.0)
    ax.text(D ** 2 * 1.04, 0.3, "R = D$^2$", color="grey", fontsize=8)
    # the extrapolated small-R line reaches R bits; let it leave the frame
    ax.set_xscale("log", base=2); ax.set_ylim(0, 1.25 * top)
    ax.set_xlabel("R"); ax.set_ylabel("MI [bits]"); ax.set_title(f"D = {D}")
    ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# %%
# Local slope dMI/dlog2R between consecutive R.
fig, axes = plt.subplots(1, len(DS), figsize=(5.6 * len(DS), 3.6), squeeze=False)
for ax, D in zip(axes[0], DS):
    for cs, s in ok[ok["D"] == D].groupby("conc_std"):
        a = s.groupby("R")["MI"].mean()
        lr = np.log2(a.index.values.astype(float))
        mid = 2 ** ((lr[1:] + lr[:-1]) / 2)
        ax.plot(mid, np.diff(a.values) / np.diff(lr), "o-", color=COL[cs],
                label=f"conc_std={cs:g}")
    ax.axhline(D, color="k", lw=.9, label="D (shell)")
    ax.axhline(D + 1, color="grey", lw=.9, label="D+1 (ladder)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("R"); ax.set_ylabel("dMI / dlog2 R"); ax.set_title(f"D = {D}")
    ax.legend(fontsize=7)
plt.tight_layout(); plt.show()

# %%
# Geometry: norm spread (ladder) and zero-sum directions.
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for (D, cs), s in X.groupby(["D", "conc_std"]):
    a = s.groupby("R")[["Ucv", "zero_sum"]].mean()
    ls = "-" if D == DS[0] else "--"
    axes[0].plot(a.index, a["Ucv"], "o" + ls, color=COL[cs], label=f"D={D}, conc_std={cs:g}")
    axes[1].plot(a.index, a["zero_sum"], "o" + ls, color=COL[cs])
axes[0].set_ylabel("std(U) / <U>")
axes[1].set_ylabel("||sum_r u_r|| / R")
for a in axes:
    a.set_xscale("log", base=2); a.set_xlabel("R")
axes[0].legend(fontsize=7)
plt.tight_layout(); plt.show()

# %%
# Operating point: spread of c at the measured U, Ebase, against the 1/D optimum.
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for (D, cs), s in X.groupby(["D", "conc_std"]):
    a = s.groupby("R")[["sd_c", "p_atoms"]].mean()
    ls = "-" if D == DS[0] else "--"
    axes[0].plot(a.index, a["sd_c"] * D, "o" + ls, color=COL[cs], label=f"D={D}, conc_std={cs:g}")
    axes[1].plot(a.index, a["p_atoms"], "o" + ls, color=COL[cs])
axes[0].axhline(1, color="k", lw=.9)
axes[0].set_ylabel("D * sd(c)   (expansion optimum: 1)")
axes[1].set_ylabel("P(|c| >= 1): all on or all off")
for a in axes:
    a.set_xscale("log", base=2); a.set_xlabel("R")
axes[0].legend(fontsize=7)
plt.tight_layout(); plt.show()

# %%
