# %%
"""Verdict for the cell convergence check — did it reach the KNOWN answer?

Reads the run written by tasks/cells/convergence/scripts/convergence.py. That script
builds a world whose entropy is a number we can write down (one ligand per sniff, drawn
uniformly from N_LIG, fixed concentration), so the target is

    identity_channel  ->  log2(N_LIG)      the real information
    concentration_channel -> 0             everything left is readout softness
    codeword_entropy_K_hat -> N_LIG        one codeword per ligand

The checks are TWO-SIDED. Overshooting is as much a failure as falling short: a cell
parked at activity 0.5 is a coin, contributing a full bit of entropy while saying nothing
about the sniff, and an array of them reports the MAXIMUM (doc/theory/07 §3b.6-3b.7).
That degenerate solution shows up here as concentration_channel > 0 alongside an
identity_channel short of target — while full_array_entropy_kt looks excellent.
"""
import sys
from pathlib import Path
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")

import math
import numpy as np
import matplotlib.pyplot as plt
from src.plotlib import load_run, DATA_ROOT
from src.IO import find_latest_sweep, SweepLoader

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

TOL = 0.15          # allowed shortfall in bits before a check is called FAIL

SWEEP = find_latest_sweep(str(DATA_ROOT / "convergence"), prefix="cell_convergence")[0]
RUN_DIR = list(SweepLoader(SWEEP).iter_run_dirs())[0][1]
print(f"run_dir: {RUN_DIR}")

# %%
# ── the target, derived from the config (not hard-coded) ─────────────────────
cfg, hist = load_run(run_dir=RUN_DIR)
n_lig, n_cells = cfg.n_ligands, len(cfg.cell_gene_sets)
ceiling = min(math.log2(n_lig), n_cells)

print(f"{n_lig} ligands, one per sniff  ->  world entropy = {math.log2(n_lig):.4f} bits")
print(f"{n_cells} cells                  ->  array capacity = {n_cells} bits")
print(f"TARGET: {ceiling:.4f} bits and {n_lig} distinct codewords")
print(f"pool: R_pool={len(cfg.receptor_indices)}, genes/cell="
      f"{min(len(g) for g in cfg.cell_gene_sets)}-{max(len(g) for g in cfg.cell_gene_sets)}")

# %%
# ── final measurement ────────────────────────────────────────────────────────
# SimulationRunner.run() writes test_results.json directly (no loader method for it).
# _test() repeats the measurement test_epochs times, so every value is a list.
# Key names come from _eval_stats: a measurement returning a dict contributes its own
# keys (codeword_entropy, entropy_kt), one returning a scalar is keyed by its fn name.
import json, os
with open(os.path.join(RUN_DIR, "test_results.json")) as f:
    test = json.load(f)


def final(key, default=float("nan")):
    """Mean over the test repeats of one measurement key."""
    v = test.get(key, default)
    return float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v)


info    = final("identity_channel")        # I(A ; which ligand)
softness = final("concentration_channel")  # H(A | which ligand): pure readout softness
kt_lo   = final("full_array_entropy_kt")
kt_hi   = final("full_array_entropy_kt_upper")
k_hat   = final("codeword_entropy_K_hat")

print(f"  identity_channel      {info:8.4f} bits   (target {ceiling:.4f})")
print(f"  concentration_channel {softness:8.4f} bits   (target 0)")
print(f"  KT bracket            [{kt_lo:.4f}, {kt_hi:.4f}]")
print(f"  distinct codewords    {k_hat:8.0f}        (target {n_lig})")

# %%
# ── verdict ──────────────────────────────────────────────────────────────────
checks = [
    ("information reached the world's entropy", info >= ceiling - TOL),
    ("information did not exceed it (no free bits)", info <= ceiling + TOL),
    ("cells are decided, not coins", softness < TOL),
    ("codewords separate the ligands", k_hat >= n_lig - 1),
]
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

if all(ok for _, ok in checks):
    print("\nCONVERGED to the expected result.")
elif info < ceiling - TOL and softness < TOL and k_hat >= n_lig - 1:
    # every bit found is real and the code is nearly complete: it just needs longer.
    print("\nUNDER-TRAINED, not broken: the information is real (softness ~ 0) and the "
          "codewords are nearly all separated. Raise `epochs` and rerun.")
else:
    print("\nDID NOT converge. Softness > 0 with information short of target is the "
          "fair-coin degeneracy — see doc/theory/07 §3b.6-3b.7, not an epoch budget.")

# %%
# ── convergence curve: is it still climbing? ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for col, style in (("identity_channel", "-"), ("concentration_channel", "--"),
                   ("full_array_entropy_kt", ":")):
    if col in hist.columns:
        ax.plot(hist["epoch"], hist[col], style, label=col)
ax.axhline(ceiling, color="k", lw=1, ls="-.", label=f"target = log2({n_lig})")
if "cell_theta" in hist.columns:                      # phase-2 hardening starts here
    ax.axvline(cfg.cell_phase_split * cfg.epochs, color="grey", lw=1, alpha=.6)
    ax.text(cfg.cell_phase_split * cfg.epochs, ax.get_ylim()[1], " phase 2",
            va="top", fontsize=8, color="grey")
ax.set_xlabel("epoch"); ax.set_ylabel("bits")
ax.set_title(f"Cell convergence — {n_cells} cells, {n_lig} ligands")
ax.legend(fontsize=8)
plt.savefig(FIGURES / "cell_convergence.png", dpi=150, bbox_inches="tight")
plt.show()
