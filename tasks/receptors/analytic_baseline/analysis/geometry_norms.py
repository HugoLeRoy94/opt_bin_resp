# %%
"""Check 1: where the optimiser puts the receptors.

The most exact part of the closed form is the geometry, so it is tested first.
Three predictions, in increasing order of how much they can fail:

  (a) every receptor sits at the SAME distance U from the family centre
      (Lemma 1: at fixed Ebase and Emax, balance has one solution in U)
  (b) that distance equals U* = lambda sqrt(w*), w* = (1+Lambda)/2   (Eq. 39)
  (c) the operating point zbar matches exp(-(rho^2 + w*))            (Eq. 30)

(a) is the cleanest: it needs no closed form at all, only isotropy. If (a) holds
and (b) fails, the lever-arm argument of Section 7 is wrong but the reduction of
Sections 1 to 5 is fine. Above nu = 1 the prediction INVERTS: Section 12 says the
norms should spread into a ladder, so a rising spread there is a success, not a
failure.

Emax is learnable, so the closed form is evaluated twice: at the initial Emax=10
and at the measured one. Only the second is a fair test.
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

# %%
df = RU.load()
print(f"{len(df)} runs, R in {sorted(df['R'].unique())}, "
      f"conc_std in {sorted(df['conc_std'].unique())}")

# %%
rows = []
for path, grp in df.groupby("path"):
    g = RU.geometry(grp)
    pm, pi = g["pred_meas"], g["pred_init"]
    rows.append(dict(
        path=path, R=g["R"], D=g["D"], conc_std=g["conc_std"],
        nu_pred=pm["nu_min"], nu_pred_init=pi["nu_min"],
        selfcons=pm["selfcons"],
        U_mean=g["U"].mean(), U_std=g["U"].std(),
        U_spread=g["U"].std() / max(g["U"].mean(), 1e-12),
        U_star=pm["U_star"], U_star_init=pi["U_star"],
        U_ratio=g["U"].mean() / pm["U_star"],
        zbar_meas=float(np.mean(g["zbar_meas"])), zbar_pred=pm["zbar"],
        Emax_mean=float(np.mean(g["Emax"])), Emax_drift=g["Emax_drift"],
        Ebase_mean=float(np.mean(g["Ebase"])), Ebase_pred=pm["Ebase"],
    ))
G = pd.DataFrame(rows).sort_values(["conc_std", "R"])
print(G.groupby("conc_std")[["nu_pred", "Emax_drift", "U_ratio",
                             "U_spread", "selfcons"]].mean().round(4))

# %%
# Emax drift first: if the ceiling moved, every prediction below shifts with it.
# A large positive drift is itself a result, see the README.
fig, ax = plt.subplots(figsize=(5, 3.2))
for cs, sub in G.groupby("conc_std"):
    ax.plot(sub["R"], 100 * sub["Emax_drift"], "o-",
            label=f"nu={sub['nu_pred'].iloc[0]:.2f}")
ax.axhline(0, color="k", lw=.8)
ax.set_xlabel("R"); ax.set_ylabel("Emax drift from init  [%]")
ax.set_title("did the optimiser move the selectivity ceiling?")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

# %%
# (b) mean distance against the predicted optimum.
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
for cs, sub in G.groupby("conc_std"):
    lab = f"nu={sub['nu_pred'].iloc[0]:.2f}"
    axes[0].plot(sub["R"], sub["U_mean"], "o-", label=lab)
    axes[0].plot(sub["R"], sub["U_star"], "--", color=axes[0].lines[-1].get_color(),
                 lw=.9)
    axes[1].plot(sub["R"], sub["U_ratio"], "o-", label=lab)
axes[0].set_ylabel(r"$\langle U\rangle$   (dashed: predicted $U_\star$)")
axes[1].axhline(1.0, color="k", lw=.8)
axes[1].set_ylabel(r"$\langle U\rangle\,/\,U_\star$")
for a in axes:
    a.set_xlabel("R")
    a.axvline(G["D"].iloc[0], color="grey", ls=":", lw=.8)
axes[1].legend(fontsize=8, title="at measured Emax")
fig.suptitle("Check 1b: lever arm against Eq. 39", y=1.02)
plt.tight_layout(); plt.show()

# %%
# (a) are the receptors at the same distance? And does the spread rise past nu=1?
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for cs, sub in G.groupby("conc_std"):
    ax.plot(sub["R"], sub["U_spread"], "o-",
            label=f"nu={sub['nu_pred'].iloc[0]:.2f}")
ax.set_xlabel("R"); ax.set_ylabel(r"std$(U)\,/\,\langle U\rangle$")
ax.set_title("Check 1a: equal norms below nu=1, ladder above (Section 12)")
ax.axvline(G["D"].iloc[0], color="grey", ls=":", lw=.8)
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

# %%
# The spread as a function of nu is the sharper version of the same plot: the
# staggering transition is predicted at nu = 1 and should show as a kink.
fig, ax = plt.subplots(figsize=(5.2, 3.4))
agg = G.groupby("nu_pred")["U_spread"].agg(["mean", "std", "count"])
ax.errorbar(agg.index, agg["mean"], yerr=agg["std"], fmt="o-", capsize=3)
ax.axvline(1.0, color="r", ls="--", lw=1, label=r"predicted transition $\nu=1$")
ax.set_xscale("log"); ax.set_xlabel(r"$\nu$ (predicted)")
ax.set_ylabel(r"std$(U)\,/\,\langle U\rangle$")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

# %%
# (c) operating point. zbar is set by Ebase, which is learnable, so this tests
# the balance condition Eq. 26 rather than any approximation.
fig, ax = plt.subplots(figsize=(4.2, 4))
ax.scatter(G["zbar_pred"], G["zbar_meas"], c=np.log10(G["nu_pred"]),
           cmap="viridis", s=26)
lim = [0, max(G["zbar_pred"].max(), G["zbar_meas"].max()) * 1.1]
ax.plot(lim, lim, "k--", lw=.8)
ax.set_xlabel(r"predicted $\bar z = e^{-(\rho^2+w_\star)}$")
ax.set_ylabel(r"measured $\bar z$ from $E_{base}$")
ax.set_title("Check 1c: operating point")
plt.colorbar(ax.collections[0], ax=ax, label=r"$\log_{10}\nu$")
plt.tight_layout(); plt.show()

# %%
# Rows where the delta route is not quantitative get the exact route instead.
bad = G[G["selfcons"] < 2.0]
if len(bad):
    print(f"\n{len(bad)} runs with selfcons < 2 (Eq. 41): delta route not "
          f"quantitative, recomputing with the exact route")
    for path in bad["path"].unique()[:4]:
        g = RU.geometry(df[df["path"] == path])
        e = RU.exact_pred(g)
        print(f"  R={g['R']:3d} cs={g['conc_std']:.4f}  measured U={g['U'].mean():.3f}"
              f"  delta U*={g['pred_meas']['U_star']:.3f}"
              f"  exact U*={e['U_star']:.3f}  sat_on={e['sat_on']:.2e}")
