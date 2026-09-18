# %%
"""Qualitative convergence diagnostic: approach to an approximate identity target.

Reads the run written by tasks/cells/convergence/scripts/convergence.py. That script
builds an approximately singleton world with small concentration variation. The
reference target (not a guaranteed achievable answer) is

    identity_channel  ->  log2(N_LIG)      the real information

The actual loss maximizes KT MI. H(response | mixture mask), historically named
concentration_channel, includes concentration information AND response noise; it
must not be labelled pure softness. Falling short can reflect geometry, readout,
sampling, or optimization. These checks do not diagnose a unique cause.

The mean drive is interpreted as a stochastic firing probability, so nonzero
conditional response entropy is expected and is removed by the MI objective. The
training loop records about 100 evaluation points. Older runs stored their *logging
index* (0..99) as ``epoch`` rather than the corresponding optimization step; this
notebook reconstructs the latter.
"""
import sys
from pathlib import Path
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.plotlib import load_run, load_model, DATA_ROOT
from src.IO import find_latest_sweep, SweepLoader, SingleRunLoader
from src.cells import CellReadout
from src.analysis_helper import (build_latent_umap, plot_latent_umap,
                                 cell_ligand_responses, plot_cell_response_umap)

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
print(f"TARGET: {ceiling:.4f} bits of ligand-identity information")
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
print(f"  distinct hard codes   {k_hat:8.0f}        (diagnostic only for mean readout)")

# %%
# ── verdict ──────────────────────────────────────────────────────────────────
checks = [
    ("information reached the world's entropy", info >= ceiling - TOL),
    ("information did not exceed it (no free bits)", info <= ceiling + TOL),
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
          "emitted one constant code throughout training. With mean readout this does "
          "not by itself imply zero stochastic MI; compare the MI curves above.")


def annealing_marker(ax):
    """Show when the receptor temperature reaches its final value."""
    annealing_end = 0.8 * cfg.epochs
    ax.axvline(annealing_end, color="0.45", lw=1, ls="--", alpha=.8)
    ax.text(annealing_end, 0.98, " receptor annealing ends", transform=ax.get_xaxis_transform(),
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
annealing_marker(ax_mi)
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
                label=f"deterministic reference: {n_lig} codes")
annealing_marker(ax_code)
ax_code.set_xlabel("optimization update")
ax_code.set_ylabel("bits")
ax_code.legend(fontsize=8, ncol=2)
ax_code.grid(axis="y", alpha=.2)

fig.tight_layout()
#plt.savefig(FIGURES / "cell_convergence.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# ── final latent-space map ───────────────────────────────────────────────────
# plot_latent_umap is the project's shared visualisation: family regions and
# centres, the fixed ligands, and receptor centroids. For cell simulations, show
# one homomer per gene: label g represents [g, g, ..., g]. In the interface model
# its position is the homotypic pocket midpoint (v_plus[g] + v_minus[g]) / 2.
# Include every gene, even genes absent from the sampled cells' repertoires.
env, physics, receptor_indices = load_model(run_dir=RUN_DIR)
plot_receptors = receptor_indices
if cfg.is_cell_mode():
    plot_receptors = torch.arange(env.n_genes, device=receptor_indices.device)[:, None].expand(
        -1, cfg.k_sub
    )
print(f"UMAP: {env.n_families} families, {env.n_ligands} ligands, "
      f"{len(plot_receptors)} displayed receptors (simulation R_pool={len(receptor_indices)})")
fig, ax = plt.subplots(figsize=(9, 7))
latent_embedding = build_latent_umap(env, plot_receptors)
plot_latent_umap(env, plot_receptors, ax=ax, embedding=latent_embedding)
ax.set_title(f"Latent-space UMAP — {env.n_genes} gene homomers (labels = gene IDs)"
             if cfg.is_cell_mode() else "Latent-space UMAP — receptors")
fig.tight_layout()
#plt.savefig(FIGURES / "cell_convergence_latent_umap.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# ── cell responses on the same chemical map ──────────────────────────────────
# Restore the actual trained mean readout and its deterministic receptor weights.
CONCENTRATION = 1.0
checkpoint = SingleRunLoader(RUN_DIR).load_checkpoint(map_location="cpu")
readout = CellReadout(
    checkpoint["readout_state"]["W"], mode=checkpoint["readout_mode"],
    temperature=checkpoint["readout_temperature"], k_sub=cfg.k_sub,
    learnable_threshold=cfg.cell_threshold_learnable,
)
readout.load_state_dict(checkpoint["readout_state"])
readout.eval()
responses = cell_ligand_responses(env, physics, receptor_indices, readout, CONCENTRATION)
fig, axes = plot_cell_response_umap(latent_embedding, responses, cfg.cell_gene_sets,
                                   concentration=CONCENTRATION)
fig.savefig(FIGURES / "cell_convergence_response_umap.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
