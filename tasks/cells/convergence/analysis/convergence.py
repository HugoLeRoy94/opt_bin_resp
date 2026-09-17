# %%
"""Qualitative convergence diagnostic: approach to an approximate identity target.

Reads the run written by tasks/cells/convergence/scripts/convergence.py. That script
builds an approximately singleton world with small concentration variation. The
reference target (not a guaranteed achievable answer) is

    identity_channel  ->  log2(N_LIG)      the real information
    conditional_entropy_response -> 0      only if responses become deterministic
    codeword_entropy_K_hat -> N_LIG        one codeword per ligand

The actual loss maximizes KT MI. H(response | mixture mask), historically named
concentration_channel, includes concentration information AND response noise; it
must not be labelled pure softness. Falling short can reflect geometry, readout,
sampling, or optimization. These checks do not diagnose a unique cause.

The training loop records about 100 evaluation points. Older runs stored their
*logging index* (0..99) as ``epoch`` rather than the corresponding optimization
step; this notebook reconstructs the latter so the phase-2 boundary is drawn in
the right place.
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

print(f"{n_lig} ligands, approximately one per sniff -> identity reference = {math.log2(n_lig):.4f} bits")
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
softness = final("conditional_entropy_response")  # H(A | full sampled input)
kt_lo   = final("mutual_information_kt")
kt_hi   = final("mutual_information_kt_upper")
k_hat   = final("codeword_entropy_K_hat")

print(f"  identity_channel      {info:8.4f} bits   (target {ceiling:.4f})")
print(f"  response noise entropy {softness:8.4f} bits   (zero only in deterministic limit)")
print(f"  KT MI bracket          [{kt_lo:.4f}, {kt_hi:.4f}]")
print(f"  counting MI (MM)       {final('mutual_information_counting_mm'):.4f} bits")
print(f"  distinct / samples     {final('response_counting_unique_fraction'):.4f}")
print(f"  distinct codewords    {k_hat:8.0f}        (target {n_lig})")

# %%
# ── verdict ──────────────────────────────────────────────────────────────────
checks = [
    ("information reached the world's entropy", info >= ceiling - TOL),
    ("information did not exceed it (no free bits)", info <= ceiling + TOL),
    ("codewords separate the ligands", k_hat >= n_lig - 1),
]
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

if all(ok for _, ok in checks):
    print("\nCONVERGED to the expected result.")
else:
    print("\nReference not reached. Compare seeds, sample budgets, geometry and readout "
          "before attributing the gap to optimization. Softness alone is not a failure.")

# %%
# ── training trajectory: did MI ever begin to improve? ───────────────────────
def optimization_epochs(history, config):
    """Return real optimization steps, including compatibility with old stats.csv.

    SimulationRunner evaluates every ``max(1, epochs // 100)`` updates. Its older
    logger saved 0, 1, ..., N-1 instead of those update numbers. Recognise exactly
    that sequence rather than blindly rescaling a future, correctly logged file.
    """
    saved = history["epoch"].to_numpy()
    expected_indices = np.arange(len(history))
    interval = max(1, config.epochs // 100)
    if interval > 1 and np.allclose(saved, expected_indices):
        return expected_indices * interval, interval, True
    return saved, interval, False


epochs, eval_interval, reconstructed_epochs = optimization_epochs(hist, cfg)
if reconstructed_epochs:
    print(f"history: {len(hist)} evaluations every {eval_interval} updates; "
          "reconstructed optimization-step axis from legacy logging indices")
else:
    print(f"history: {len(hist)} evaluations (nominal interval {eval_interval} updates)")


def values(column):
    """A numeric history column, or None when an older run did not log it."""
    return hist[column].to_numpy(dtype=float) if column in hist.columns else None


kt_curve = values("mutual_information_kt")
if kt_curve is not None:
    print(f"KT lower MI during training: first={kt_curve[0]:.4f}, "
          f"last={kt_curve[-1]:.4f}, range=[{np.nanmin(kt_curve):.4f}, "
          f"{np.nanmax(kt_curve):.4f}] bits")

hard_k = values("codeword_entropy_K_hat")
if hard_k is not None and np.allclose(hard_k, 1):
    print("DIAGNOSTIC: K_hat stayed at 1 at every evaluation: the hard cell array "
          "emitted one constant code throughout training. This is an initialization/"
          "signal-or-gradient failure, not a late convergence plateau.")

theta = values("cell_theta")
theta_floor = 1.0 / cfg.cell_n_molecules
if theta is not None and np.allclose(theta, theta_floor):
    print(f"DIAGNOSTIC: shared cell threshold remained at its physical floor "
          f"(1/N = {theta_floor:.2e}). Inspect the initial drive distribution and "
          "silent/saturated-cell calibration before changing the optimizer.")


def phase_marker(ax):
    """Show when receptor sharpness stops changing and cell hardening begins."""
    phase2 = cfg.cell_phase_split * cfg.epochs
    ax.axvline(phase2, color="0.45", lw=1, ls="--", alpha=.8)
    ax.text(phase2, 0.98, " phase 2: cell hardening", transform=ax.get_xaxis_transform(),
            va="top", ha="left", fontsize=8, color="0.35")


# Top: all information quantities are measured using the FINAL sharpness, even
# during phase 1. The grey training surrogate is at the current annealed
# sharpness, so it is useful for optimisation diagnosis but is not comparable
# point-for-point with the final-sharpness KT curves.
fig, (ax_mi, ax_code) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                     gridspec_kw={"height_ratios": (3, 2)})
for column, label, style in (
    ("mutual_information_kt", "KT MI lower", {"color": "tab:blue", "lw": 2}),
    ("mutual_information_kt_upper", "KT MI upper", {"color": "tab:blue", "ls": "--"}),
    ("identity_channel", "identity MI", {"color": "tab:orange", "lw": 1.5}),
    ("train_mutual_information", "training surrogate (annealed)",
     {"color": "0.45", "lw": 1, "alpha": .8}),
):
    series = values(column)
    if series is not None:
        ax_mi.plot(epochs, series, label=label, **style)
ax_mi.axhline(ceiling, color="k", lw=1, ls=":", label=f"reference = {ceiling:.2f} bits")
phase_marker(ax_mi)
ax_mi.set_ylabel("mutual information [bits]")
ax_mi.set_title(f"Cell convergence diagnostic — {n_cells} cells, {n_lig} ligands")
ax_mi.legend(fontsize=8, ncol=2)
ax_mi.grid(axis="y", alpha=.2)

# Bottom: distinguish "the estimator is noisy" from "the array is constant".
# H(hard code) and log2(K_hat) both vanish for a single response code; response
# conditional entropy isolates stochasticity of that response.
for column, label, style in (
    ("codeword_entropy_plugin", "H(hard code)", {"color": "tab:green", "lw": 2}),
    ("conditional_entropy_response", "H(response | sniff)",
     {"color": "tab:red", "ls": "--"}),
):
    series = values(column)
    if series is not None:
        ax_code.plot(epochs, series, label=label, **style)
if hard_k is not None:
    ax_code.plot(epochs, np.log2(np.maximum(hard_k, 1)), color="tab:purple", ls=":",
                 label="log2(K_hat)")
ax_code.axhline(math.log2(n_lig), color="k", lw=1, ls=":",
                label=f"{n_lig} ligand codes")
phase_marker(ax_code)
ax_code.set_xlabel("optimization update")
ax_code.set_ylabel("bits")
ax_code.legend(fontsize=8, ncol=2)
ax_code.grid(axis="y", alpha=.2)

fig.tight_layout()
plt.savefig(FIGURES / "cell_convergence.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
