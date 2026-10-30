# %%
"""large_R: the large-R expansion and the counting bounds against simulation.

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_lib"))
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
                     exp_meas=float(T.mi_large_R(p, U, eb, R)),
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
# MI vs R, one panel per D.
fig, axes = plt.subplots(1, len(DS), figsize=(5.6 * len(DS), 4.2), squeeze=False)
for ax, D in zip(axes[0], DS):
    sub = ok[ok["D"] == D]
    rr = np.unique(np.geomspace(2, X["R"].max(), 40).astype(int))
    for cs, s in sub.groupby("conc_std"):
        col = COL[cs]
        a = s.groupby("R")["MI"].agg(["mean", "std"])
        ax.errorbar(a.index, a["mean"], yerr=a["std"], fmt="o", color=col, capsize=3,
                    label=f"conc_std={cs:g}")
        e = s.groupby("R")["exp_meas"].mean()
        ax.plot(e.index, e.values, "x", color=col, ms=7)
        p0 = s["params"].iloc[0]
        o = T.lambert_optimum(p0)
        ax.plot(rr, T.mi_large_R(p0, o["U_star"], o["Ebase"], rr), color=col, lw=1.2,
                dashes=(6, 2))
        small = rr[rr <= D + 1]
        ax.plot(small, [T.mi_eff(o["nu_min"], int(r))["MI"] for r in small], "-",
                color=col, lw=1.0)
    ax.plot(rr, [LR.log2_shell_bound(int(r), D) for r in rr], "k-", lw=.9,
            label="shell bound")
    ax.plot(rr, [LR.log2_ladder_bound(int(r), D) for r in rr], "-", color="grey",
            lw=.9, label="ladder bound")
    ax.plot([], [], "k", lw=1.2, dashes=(6, 2), label="expansion at U*")
    ax.plot([], [], "kx", label="expansion at measured U, Ebase")
    ax.plot([], [], "k-", lw=1.0, label="small-R theory (R <= D+1)")
    ax.axvline(D ** 2, color="grey", lw=.6, ls=":")
    ax.set_xscale("log", base=2); ax.set_ylim(bottom=0)
    ax.set_xlabel("R"); ax.set_ylabel("MI [bits]"); ax.set_title(f"D = {D}")
    ax.legend(fontsize=7)
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
