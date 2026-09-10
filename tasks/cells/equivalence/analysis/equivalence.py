# %%
"""Did cell mode reduce to the receptor model?

Compares the runs from scripts/as_cells.py and scripts/as_receptors.py. A cell holding
exactly one receptor IS that receptor, so with W == I cell mode should reproduce the
receptor model — same environment draw, same sniffs, same receptors.

## Which numbers to compare, and why not the KT entropies

The cell run uses the real `threshold` readout, so its activity is near-BINARY while the
receptor run reports SOFT open probabilities p. Those are not on the same footing:

    receptor KT  =  H(code)  +  conditional entropy of the soft p
    cell     KT  =  H(code)  +  ~0

So **expect receptor KT >= cell KT**, by roughly the receptor's conditional-entropy term.
That gap is not a failure — it is the soft/hard difference, and it shrinks as the
receptor temperature anneals. The like-for-like comparison is `codeword_entropy_*`,
which binarises BOTH sides at 0.5; those are the checks below.

## Reading a mismatch

  receptor_indices differ    -> the pool is not the receptor list (ordering/canonicalisation)
  codewords differ a lot     -> theta is not separating OFF from ON. Check whether it hit
                                the 1/N floor (the `[cell]` line in the run log) and see
                                doc/theory/09 §9.8.1
  everything differs         -> the runs did not share an environment (seeding broken)

The trajectories are close but NOT identical: the cell run adds a phase-2 hardening the
receptor run has no need for. This tests convergence to the same answer, not step-for-step
agreement. For the bit-exact plumbing check, run as_cells.py with cell_readout="mean".
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
print(f"theta floor 1/N            : {1.0 / c.cell_n_molecules:.1e}"
      f"   (N = {c.cell_n_molecules:.0e} receptor molecules/cell)")
print(f"phase split                : {c.cell_phase_split}  "
      f"(phase 1 anneals the receptor; phase 2 hardens the cell)")

# %%
# ── final measurements, side by side ─────────────────────────────────────────
def final(test, key):
    v = test.get(key, float("nan"))
    return float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v)


# hard-code metrics first: those are the like-for-like comparison
KEYS = ("codeword_entropy_K_hat", "codeword_entropy_mm",
        "full_array_entropy_kt", "full_array_entropy_kt_upper")
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
kt_cell, kt_rec = vals["full_array_entropy_kt"]

checks = [
    ("same receptors, same order", same_array),
    ("hard-codeword entropies agree", mm_gap < TOL_BITS),
    ("distinct-codeword counts agree", kh_rel < TOL_KHAT),
    # soft vs hard: the cell (binary) must not report MORE than the soft receptor
    ("cell KT does not exceed receptor KT", kt_cell <= kt_rec + TOL_BITS),
]
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
print(f"\n  KT gap (receptor - cell) = {kt_rec - kt_cell:+.4f} bits"
      f"   <- expected >= 0: the receptor's soft p carries a conditional-entropy term")

if all(ok for _, ok in checks):
    print("\nEQUIVALENT: one receptor per cell reproduces the receptor model.")
elif not same_array:
    print("\nNOT COMPARABLE: the two runs used different receptor arrays — fix that first.")
elif mm_gap >= TOL_BITS or kh_rel >= TOL_KHAT:
    print(f"\nCODES DIFFER (mm gap {mm_gap:.3f} bits, K_hat {kh_a:.0f} vs {kh_b:.0f}). "
          "theta is not separating OFF from ON. Check the [cell] line in the run log for "
          "whether it hit the 1/N floor; see doc/theory/09 §9.8.1.")
else:
    print("\nCell KT exceeds receptor KT, which should be impossible for a hard code "
          "against a soft one — suspect the cell did not harden (check T_cell in phase 2).")

# %%
# ── training curves should lie on top of each other ──────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, style in (("as_receptors", "-"), ("as_cells", "--")):
    h = hists[name]
    for col in [c for c in h.columns if "entropy" in c]:
        ax.plot(h["epoch"], h[col], style, label=f"{name}: {col}", alpha=.8)
ax.set_xlabel("epoch"); ax.set_ylabel("bits")
ax.set_title("Cell array (one receptor per cell) vs receptor array")
ax.legend(fontsize=7)
plt.savefig(FIGURES / "equivalence.png", dpi=150, bbox_inches="tight")
plt.show()
