# %%
"""Did cell mode reduce to the receptor model?

Compares the runs from scripts/as_cells.py and scripts/as_receptors.py. A cell holding
exactly one receptor has drive p when W == I. With the mean readout, activity = W @ p = p,
so the cell and receptor scripts describe the same stochastic binary channel.

## Which numbers to compare, and why not the KT entropies

Both runs report the same soft opening probabilities p:

    MI lower = entropy KT lower - H(response | full input)
    MI upper = entropy KT upper - H(response | full input)

Both optimize KT MI, and sampled-output counting estimates the same stochastic channel.
Hard codewords remain a useful deterministic diagnostic, while the MI measurements
compare the full conditional response distributions.

## Reading a mismatch

  receptor_indices differ    -> the pool is not the receptor list (ordering/canonicalisation)
  codewords differ a lot     -> inspect pool ordering, sampling and optimization
  everything differs         -> also check initial environment and seeds
"""
import sys
from pathlib import Path
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")

import json
import numpy as np
import matplotlib.pyplot as plt
from src.plotlib import load_run, DATA_ROOT
from src.IO import find_latest_sweep, SweepLoader

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

ROOT = DATA_ROOT / "equivalence"
RUNS = {}
for name in ("as_cells", "as_receptors"):
    sweep = find_latest_sweep(str(ROOT), prefix=name)[0]
    RUNS[name] = list(SweepLoader(sweep).iter_run_dirs())[0][1]
    print(f"{name:14s} {RUNS[name]}")

# %%
# ── the arrays must be the same receptors in the same order ──────────────────
cfgs, hists, tests = {}, {}, {}
for name, run_dir in RUNS.items():
    cfgs[name], hists[name] = load_run(run_dir=run_dir)
    with open(Path(run_dir) / "test_results.json") as f:
        tests[name] = json.load(f)

ri_c, ri_r = cfgs["as_cells"].receptor_indices, cfgs["as_receptors"].receptor_indices
same_array = ri_c == ri_r
print(f"receptor_indices identical : {same_array}   ({len(ri_c)} vs {len(ri_r)} receptors)")
if not same_array:
    print("  cells    :", ri_c)
    print("  receptors:", ri_r)
c = cfgs["as_cells"]
print(f"cell readout               : {c.cell_readout}")

# %%
# ── final measurements, side by side ─────────────────────────────────────────
def final(test, key):
    v = test.get(key, float("nan"))
    return float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v)


# These measurements describe the same channel and should agree up to numerical and
# finite-sample variation.
KEYS = ("codeword_entropy_K_hat", "codeword_entropy_mm",
        "mutual_information_kt", "mutual_information_kt_upper",
        "conditional_entropy_response", "mutual_information_counting_mm",
        "response_counting_unique_fraction")
TOL_BITS = 0.10     # hard-codeword entropy agreement, in bits
TOL_KHAT = 0.10     # distinct-codeword count agreement, relative

print(f"\n{'metric':<30}{'cells':>14}{'receptors':>14}{'|diff|':>12}")
rows = []
for k in KEYS:
    a, b = final(tests["as_cells"], k), final(tests["as_receptors"], k)
    rows.append((k, a, b, abs(a - b)))
    print(f"{k:<30}{a:>14.6f}{b:>14.6f}{abs(a - b):>12.2e}")

# %%
# ── verdict ──────────────────────────────────────────────────────────────────
vals = {k: (a, b) for k, a, b, _ in rows}
mm_gap = abs(vals["codeword_entropy_mm"][0] - vals["codeword_entropy_mm"][1])
kh_a, kh_b = vals["codeword_entropy_K_hat"]
kh_rel = abs(kh_a - kh_b) / max(kh_b, 1.0)
channel_keys = ("mutual_information_kt", "mutual_information_kt_upper",
                "conditional_entropy_response", "mutual_information_counting_mm")
channel_gap = max(abs(vals[k][0] - vals[k][1]) for k in channel_keys)

checks = [
    ("cell readout is mean", c.cell_readout == "mean"),
    ("same receptors, same order", same_array),
    ("stochastic-channel metrics agree", channel_gap < TOL_BITS),
    ("hard-codeword entropies agree", mm_gap < TOL_BITS),
    ("distinct-codeword counts agree", kh_rel < TOL_KHAT),
]
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

if all(ok for _, ok in checks):
    print("\nCell mean readout and receptor model agree.")
elif not same_array:
    print("\nNOT COMPARABLE: the two runs used different receptor arrays — fix that first.")
else:
    print(f"\nCHANNELS DIFFER (largest stochastic metric gap {channel_gap:.3f} bits, "
          f"hard-entropy gap {mm_gap:.3f} bits, K_hat {kh_a:.0f} vs {kh_b:.0f}). "
          "Check receptor ordering, seeds and numerical differences in the cell map.")

# %%
# ── compare training trends ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, style in (("as_receptors", "-"), ("as_cells", "--")):
    h = hists[name]
    for col in [c for c in h.columns if "mutual_information" in c or c == "conditional_entropy_response"]:
        ax.plot(h["epoch"], h[col], style, label=f"{name}: {col}", alpha=.8)
ax.set_xlabel("epoch"); ax.set_ylabel("bits")
ax.set_title("Cell array (one receptor per cell) vs receptor array")
ax.legend(fontsize=7)
plt.savefig(FIGURES / "equivalence.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
