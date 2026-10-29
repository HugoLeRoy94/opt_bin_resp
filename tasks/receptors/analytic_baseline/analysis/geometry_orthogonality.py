# %%
"""Check 2: how are the receptor DIRECTIONS arranged?

This check used to ask whether the directions are orthogonal, following assumption
A5. The data says they are not, and that A5 was wrong: the measured mean pairwise
cosine sits on -1/(R-1) to three decimals at every R. The array is a zero-sum
(regular-simplex) configuration, the most anti-aligned arrangement there is.

Why that is the right thing to do. Lemma 2 says orthogonal directions make the
PROJECTIONS p_r independent. It says nothing about the OUTPUTS, which share the
threshold tau, so orthogonality leaves the entire common-mode correlation standing
(Section 6: Cov(a_r,a_r') = Var(q) > 0). The cosine is a free knob that can cancel
it, since for an equiangular array the drive correlation is

    rho_x = (cos + nu^2) / (1 + nu^2)

which vanishes at cos = -nu^2. Geometry clips the knob at <cos> >= -1/(R-1),
because ||sum_r u_r||^2 = R + R(R-1)<cos> >= 0, with equality iff the directions sum
to zero. Over most of this grid nu^2 > 1/(R-1), so the clip binds and the optimum is
the zero-sum configuration.

Three things are therefore plotted, and the references are new:

  mean cos      must sit ON -1/(R-1), at EVERY R. This bound is dimension-free, so
                it binds above R = D too.
  zero-sum      ||sum_r u_r|| / R. The same statement as a residual: zero means the
                bound is saturated.
  cos spread    std of the off-diagonals. Zero below R = D+1, where all pairs can be
                equal (a regular simplex spans R-1 dimensions). Necessarily positive
                above, since the mean stays pinned but the pairs can no longer all
                match: this is why max |cos| rises past R = D while the mean does not.

The Welch bound is kept only as a packing reference for max |cos| above R = D+1. It
is NOT the target: it answers "how far apart can R vectors be", and the array is
solving "how anti-aligned can they be", which is a different question.
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
rows, grams = [], {}
for path, grp in df.groupby("path"):
    g = RU.geometry(grp)
    V = g["dirs"]                                   # (R, D) unit vectors
    Gm = V @ V.T
    off = Gm[~np.eye(len(V), dtype=bool)]           # signed off-diagonals
    R = g["R"]
    rows.append(dict(
        path=path, R=R, D=g["D"], conc_std=g["conc_std"],
        nu=g["pred_meas"]["nu_min"],
        cos_mean=float(off.mean()),                 # SIGNED, this is the test
        cos_std=float(off.std()),
        cos_absmean=float(np.abs(off).mean()),
        cos_absmax=float(np.abs(off).max()),
        floor=T.mean_cos_floor(R),                  # -1/(R-1), the target
        cos_opt=T.cos_opt(g["pred_meas"]["nu_min"], R),
        zero_sum=float(np.linalg.norm(V.sum(axis=0)) / R),
        welch=RU.welch_bound(R, g["D"]),
    ))
    grams.setdefault(R, Gm)
O = pd.DataFrame(rows).sort_values(["conc_std", "R"])
D = int(O["D"].iloc[0])
NU = O.groupby("conc_std")["nu"].mean().round(2).to_dict()


def _lab(cs):
    return f"nu~{NU[cs]:.2f}"


print("mean signed cosine against the dimension-free floor -1/(R-1):")
print(O.groupby("R")[["cos_mean", "floor", "cos_std", "zero_sum"]].mean().round(4))

# %%
# The headline: signed mean cosine on the floor, at every R.
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
for cs, sub in O.groupby("conc_std"):
    a = sub.groupby("R")[["cos_mean", "cos_std"]].mean()
    axes[0].plot(a.index, a["cos_mean"], "o-", label=_lab(cs))
    axes[1].plot(a.index, a["cos_std"], "o-", label=_lab(cs))
rr = np.array(sorted(O["R"].unique()))
axes[0].plot(rr, [T.mean_cos_floor(int(r)) for r in rr], "k--", lw=1.3,
             label=r"floor $-1/(R-1)$  (zero-sum directions)")
axes[0].axhline(0, color="grey", ls=":", lw=.9, label="orthogonal (old A5)")
axes[0].set_ylabel(r"mean $\cos\theta_{rr'}$  (signed)")
axes[1].set_ylabel(r"std of $\cos\theta_{rr'}$ over pairs")
axes[1].axvline(D + 1, color="r", ls="--", lw=.9, label="R = D+1")
for a in axes:
    a.axvline(D, color="grey", lw=.8)
    a.set_xlabel("R")
    a.legend(fontsize=7)
fig.suptitle("Check 2: the array is zero-sum, not orthogonal. Left: the mean "
             "saturates a dimension-free bound.\nRight: pairs can only all be "
             "equal (a true simplex) while R <= D+1.", y=1.08, fontsize=9)
plt.tight_layout(); plt.show()

# %%
# The same claim as a residual. 0 = bound saturated, 1/sqrt(R) = random directions.
fig, ax = plt.subplots(figsize=(5.4, 3.5))
for cs, sub in O.groupby("conc_std"):
    a = sub.groupby("R")["zero_sum"].mean()
    ax.semilogy(a.index, np.maximum(a, 1e-6), "o-", label=_lab(cs))
ax.plot(rr, 1 / np.sqrt(rr), "k:", lw=1,
        label=r"random directions, $1/\sqrt{R}$")
ax.axvline(D, color="grey", lw=.8)
ax.set_xlabel("R"); ax.set_ylabel(r"$\|\sum_r \hat u_r\| / R$")
ax.set_title("Check 2b: zero-sum residual (0 = optimal anti-alignment)")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

# %%
# Does the array sit at the CANCELLING cosine or at the geometric floor? They differ
# only while nu^2 < 1/(R-1), i.e. for arrays small enough to be lossless.
fig, ax = plt.subplots(figsize=(5.4, 3.5))
for cs, sub in O.groupby("conc_std"):
    a = sub.groupby("R")[["cos_mean", "cos_opt"]].mean()
    line, = ax.plot(a.index, a["cos_mean"], "o-", label=_lab(cs))
    ax.plot(a.index, a["cos_opt"], "--", lw=.9, color=line.get_color())
ax.set_xlabel("R"); ax.set_ylabel(r"$\cos\theta$: measured (o) vs optimal (--)")
ax.set_title(r"Check 2c: optimal is $-\min(\nu^2,\,1/(R-1))$")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

# %%
# Above R = D+1 the mean is still pinned, so the interesting number becomes the
# largest |cos|, and there the packing reference is the right one.
hi = O[O["R"] > D + 1]
if len(hi):
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for cs, sub in hi.groupby("conc_std"):
        a = sub.groupby("R")["cos_absmax"].mean()
        ax.plot(a.index, a, "o-", label=_lab(cs))
    ref = hi.groupby("R")["welch"].mean()
    ax.plot(ref.index, ref, "k--", lw=1.2, label="Welch bound (packing floor)")
    ax.set_xlabel("R"); ax.set_ylabel(r"max $|\cos\theta_{rr'}|$")
    ax.set_title("Check 2d: crowding above R = D+1, where pairs cannot all match")
    ax.legend(fontsize=8); plt.tight_layout(); plt.show()

# %%
# Gram matrices: one small (the simplex is unmistakable), one at R = D, one above.
want = [r for r in (4, D, max(O["R"])) if r in grams]
pick = sorted(set(want))
if pick:
    fig, axes = plt.subplots(1, len(pick), figsize=(4.3 * len(pick), 3.9))
    axes = np.atleast_1d(axes)
    for a, k in zip(axes, pick):
        Gm = grams[k].copy()
        np.fill_diagonal(Gm, np.nan)              # the 1s swamp the colour scale
        lim = 2.5 * abs(T.mean_cos_floor(k))      # centre the scale on the floor
        im = a.imshow(Gm, vmin=-lim, vmax=lim, cmap="coolwarm")
        a.set_title(f"R = {k}  ({'<=' if k <= D else '>'} D = {D})\n"
                    f"floor = {T.mean_cos_floor(k):+.3f}", fontsize=9)
        plt.colorbar(im, ax=a, fraction=.046)
    fig.suptitle(r"Gram matrix $\hat u_r\cdot\hat u_{r'}$, diagonal masked, "
                 r"scale centred on $-1/(R-1)$", y=1.04)
    plt.tight_layout(); plt.show()

# %%
