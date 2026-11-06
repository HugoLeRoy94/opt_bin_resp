# %%
"""two_families: does a two-family array behave as two single-family arrays?

G(n) is the single-family MI of the large_R sweep at the same parameters.
Prediction (README):
  MI_2(R)     = H(F) + max_{R_1} [ pi G(R_1) + (1-pi) G(R-R_1) ] - H(F|A)
  allocation  R_1* from that max: all on family 1 while pi dG(R_1) > (1-pi) G(1)
  core claim  H(A | F=f) = G(R_f): each group is an optimal single-family array
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Anchored on tasks/receptors, not on this file: a shared interactive kernel can
# keep __file__ from another task's script.
_RECEPTORS = next(p for p in Path(__file__).resolve().parents if p.name == "receptors")
sys.path.insert(0, str(_RECEPTORS / "two_families" / "_lib"))
import two_fam as TF                                           # noqa: E402

T, RU, LR = TF.T, TF.RU, TF.LR
from src.plotlib import load_model                             # noqa: E402

MI_COL = "codeword_entropy_mm_mean"
MISS = "codeword_entropy_missing_mass_mean"
CEIL = "codeword_entropy_log2B_mean"

# %%
df = RU.load(TF.GOAL)
rows = []
for path, grp in df.groupby("path"):
    r = grp.iloc[0]
    env, _, ri = load_model(grp)
    a = TF.array_of(env, ri)
    dec = TF.decompose(a)
    c = a["centres"]
    rows.append(dict(path=path, R=int(r["n_genes"]), MI=r[MI_COL], miss=r[MISS],
                     ceil=r[CEIL], D=int(r["latent_dim"]), conc_std=float(r["conc_std"]),
                     lam=a["lam"], Emax=float(a["Emax"].mean()),
                     centre_dist=float(np.linalg.norm(c[0] - c[1])),
                     centre_origin=np.linalg.norm(c, axis=1), **dec))
X = pd.DataFrame(rows).sort_values("R").reset_index(drop=True)
D, CS = int(X["D"].iloc[0]), float(X["conc_std"].iloc[0])
bad = X[(X["miss"] > 1e-3) | (X["MI"] > X["ceil"] - 2)]
print(f"{len(bad)} of {len(X)} runs flagged as unreliable estimates")
print(f"simulator MI - Monte Carlo H(A), max |diff|: {(X['MI'] - X['H_A']).abs().max():.3f}")

one = RU.load(LR.GOAL)
one = one[(one["latent_dim"] == D) & (np.isclose(one["conc_std"], CS))]
g1 = one.groupby("n_genes")[MI_COL].agg(["mean", "std"])
G = TF.interp_G(g1.index.values, g1["mean"].values)

X["pi1"] = X["pi"].apply(lambda p: p[0])
X["R1"] = X["R_f"].apply(lambda v: int(v[0]))
X["R1_opt"] = [TF.best_split(G, R, p)[0] for R, p in zip(X["R"], X["pi1"])]
X["pred_naive"] = [TF.h2(p) + TF.best_split(G, R, p)[1] for R, p in zip(X["R"], X["pi1"])]
X["pred"] = X["pred_naive"] - X["H_F_given_A"]
X["pred_meas_split"] = [TF.h2(p) + p * G(r1) + (1 - p) * G(R - r1) - h
                        for p, r1, R, h in zip(X["pi1"], X["R1"], X["R"], X["H_F_given_A"])]
print(X[["R", "centre_dist", "centre_origin", "pi1", "R1", "R1_opt"]].round(2).to_string())

# %%
# MI vs R. Points: simulation. Lines: two-family prediction built from G.
fig, ax = plt.subplots(figsize=(7, 4.6))
hollow = X.index.isin(bad.index)
ax.plot(X["R"][~hollow], X["MI"][~hollow], "o", color="C3", ms=6, label="simulation, 2 families")
ax.plot(X["R"][hollow], X["MI"][hollow], "o", color="C3", mfc="white", ms=6,
        label="2 families, estimate unreliable")
ax.errorbar(g1.index, g1["mean"], yerr=g1["std"], fmt="s", color="C0", ms=5, capsize=3,
            label="large_R, 1 family = G(R)")
ax.plot(X["R"], X["pred_naive"], "--", color="C3", lw=1.5, label="H(F) + optimal split of G")
ax.plot(X["R"], X["pred"], "-", color="C3", lw=1.5, label="  same, minus H(F|A)")
ax.plot(X["R"], X["pred_meas_split"], ":", color="k", lw=1.2,
        label="  same, at the measured split")
ax.set_xscale("log", base=2)
ax.set_xlabel("R"); ax.set_ylabel("MI [bits]")
ax.set_title(f"D = {D}, conc_std = {CS:g}, pi = {X['pi1'].mean():.2f}")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# %%
# Core claim, and allocation.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
for g, mk in ((0, "o"), (1, "^")):
    Rf = X["R_f"].apply(lambda v: int(v[g])).values
    Hf = X["H_A_given_f"].apply(lambda v: v[g]).values
    sc = ax.scatter(G(Rf), Hf, c=np.log2(X["R"]), marker=mk, cmap="viridis",
                    label=f"family {g + 1}")
lim = [0, float(np.nanmax(G(X["R"].values))) * 1.05]
ax.plot(lim, lim, "k-", lw=.8)
ax.set_xlabel("G(R_f)  (large_R, 1 family)"); ax.set_ylabel("H(A | F=f)  (two-family array)")
ax.legend(fontsize=8)
plt.colorbar(sc, ax=ax, label="log2 R")

ax = axes[1]
ax.plot(X["R"], X["R1"] / X["R"], "o-", color="C3", label="measured R_1 / R")
ax.plot(X["R"], X["R1_opt"] / X["R"], "k--", lw=1, label="optimal split of G")
ax.axhline(X["pi1"].mean(), color="grey", lw=.8, label="pi (log-regime limit)")
ax.set_xscale("log", base=2); ax.set_ylim(0, 1.05)
ax.set_xlabel("R"); ax.set_ylabel("fraction of receptors on family 1")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# %%
# Side terms: ambiguity H(F|A), saturation, firing on the other family, and the
# realised geometry (centres are random, one environment per R).
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(X["R"], X["H_F_given_A"], "o-", label="H(F|A) [bits]")
ax.plot(X["R"], X["p_sat"], "s-", label="P(s > ceiling)")
for g, mk in ((0, "^"), (1, "v")):
    ax.plot(X["R"], X["p_cross"].apply(lambda v: v[g]), mk + "-",
            label=f"P(group {g + 1} fires | other family)")
ax.set_xscale("log", base=2); ax.set_xlabel("R")
ax.legend(fontsize=8)
ax = axes[1]
ax.plot(X["R"], X["centre_dist"], "o-", label="distance between centres")
ax.plot(X["R"], X["centre_origin"].apply(max), "s-", label="farthest centre from origin")
ax.set_xscale("log", base=2); ax.set_xlabel("R")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

print(X[["R", "MI", "pred", "pred_meas_split", "H_F_given_A", "R1", "R1_opt"]].round(2).to_string())

# %%
