# %%
"""Check 2: are the receptor directions orthogonal?

Lemma 2 says orthogonal directions make the private projections independent,
which is what turns the array into R independent coins and gives the whole
exchangeable reduction. It is assumption A5 and it is only available while
R <= D.

The comparison must be made against the right reference, and there are two:

  Welch bound   for R > D exact orthogonality is impossible. The smallest
                achievable maximum |cos| is sqrt((R-D)/(D(R-1))). An optimiser
                that reaches it is doing as well as geometry permits, so the
                bound, not zero, is the target above R = D.
  random        E|cos| between independent directions is sqrt(2/(pi D)). If the
                measured value sits here, the optimiser has not spread the
                receptors at all and the agreement of any other check is luck.

The interesting number is therefore where the measurement sits BETWEEN those
two lines, not its absolute size.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TASK / "_lib"))
import runs as RU                                              # noqa: E402

# %%
df = RU.load()
rows, grams = [], {}
for path, grp in df.groupby("path"):
    g = RU.geometry(grp)
    V = g["dirs"]                                   # (R, D) unit vectors
    Gm = V @ V.T
    off = Gm[~np.eye(len(V), dtype=bool)]
    rows.append(dict(
        path=path, R=g["R"], D=g["D"], conc_std=g["conc_std"],
        nu_pred=g["pred_meas"]["nu_min"],
        cos_mean=float(np.abs(off).mean()),
        cos_max=float(np.abs(off).max()),
        cos_rms=float(np.sqrt((off ** 2).mean())),
        welch=RU.welch_bound(g["R"], g["D"]),
        rand=RU.random_cos(g["D"]),
    ))
    grams[(g["R"], g["conc_std"])] = Gm
O = pd.DataFrame(rows).sort_values(["conc_std", "R"])
D = int(O["D"].iloc[0])
print(O.groupby("R")[["cos_mean", "cos_max", "welch", "rand"]].mean().round(4))

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
for cs, sub in O.groupby("conc_std"):
    lab = f"nu={sub['nu_pred'].iloc[0]:.2f}"
    a = sub.groupby("R")[["cos_mean", "cos_max"]].mean()
    axes[0].plot(a.index, a["cos_mean"], "o-", label=lab)
    axes[1].plot(a.index, a["cos_max"], "o-", label=lab)

ref = O.groupby("R")[["welch", "rand"]].mean()
for k, a in enumerate(axes):
    a.plot(ref.index, ref["welch"], "k--", lw=1.1,
           label="Welch bound (best possible)")
    a.axhline(ref["rand"].iloc[0], color="grey", ls=":", lw=1.1,
              label=r"random directions, $\sqrt{2/\pi D}$")
    a.axvline(D, color="grey", lw=.8)
    a.set_xlabel("R")
    a.set_yscale("log")
axes[0].set_ylabel(r"mean $|\cos\theta_{rr'}|$")
axes[1].set_ylabel(r"max $|\cos\theta_{rr'}|$")
axes[1].legend(fontsize=7)
fig.suptitle(f"Check 2: orthogonality of receptor directions (D = {D}), "
             f"vertical line at R = D", y=1.03)
plt.tight_layout(); plt.show()

# %%
# How far between "random" and "as good as geometry allows" is the array?
# 0 means the optimiser did nothing, 1 means it saturated the Welch bound.
O["quality"] = np.where(
    O["R"] <= D,
    1.0 - O["cos_mean"] / O["rand"],                       # target is 0
    1.0 - (O["cos_max"] - O["welch"]) / np.maximum(O["rand"] - O["welch"], 1e-9),
)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for cs, sub in O.groupby("conc_std"):
    a = sub.groupby("R")["quality"].mean()
    ax.plot(a.index, a, "o-", label=f"nu={sub['nu_pred'].iloc[0]:.2f}")
ax.axhline(1, color="k", ls="--", lw=.8)
ax.axhline(0, color="grey", ls=":", lw=.8)
ax.axvline(D, color="grey", lw=.8)
ax.set_xlabel("R"); ax.set_ylabel("spreading quality  (1 = optimal, 0 = random)")
ax.set_title("Check 2b: how well the directions are spread")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

# %%
# One Gram matrix below R = D and one above, as a sanity picture.
keys = sorted(grams, key=lambda k: k[0])
pick = [k for k in keys if k[0] <= D][-1:] + [k for k in keys if k[0] > D][-1:]
if pick:
    fig, axes = plt.subplots(1, len(pick), figsize=(4.4 * len(pick), 3.8))
    axes = np.atleast_1d(axes)
    for a, k in zip(axes, pick):
        im = a.imshow(grams[k], vmin=-1, vmax=1, cmap="coolwarm")
        a.set_title(f"R = {k[0]} ({'<=' if k[0] <= D else '>'} D = {D})")
        plt.colorbar(im, ax=a, fraction=.046)
    fig.suptitle(r"Gram matrix $\hat u_r \cdot \hat u_{r'}$", y=1.02)
    plt.tight_layout(); plt.show()
