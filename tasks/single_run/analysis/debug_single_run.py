# %%
import sys
from pathlib import Path
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")

import matplotlib.pyplot as plt
from src.plotlib import load_run, load_model, DATA_ROOT
from src.analysis_helper import plot_latent_umap
from src.IO import find_latest_sweep, SweepLoader

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)

# Point at a single_run_* directory (no runs.db needed).
# find_latest_sweep picks the most recent one; change the index to pick older runs.
RUN_DIR = list(SweepLoader(find_latest_sweep(str(DATA_ROOT), prefix="single_run")[0])
               .iter_run_dirs())[0][1]
print(f"run_dir: {RUN_DIR}")

# %%
# ── training curves ───────────────────────────────────────────────────────────
cfg, hist = load_run(run_dir=RUN_DIR)
print(f"run: n_genes={cfg.n_genes}, n_receptors={cfg.n_receptors}, "
      f"entropy={cfg.entropy}, epochs={cfg.epochs}")
print(f"history: {len(hist)} rows, cols: {list(hist.columns)}")

fig, ax = plt.subplots()
for col in [c for c in hist.columns if "entropy" in c]:
    ax.plot(hist["epoch"], hist[col], label=col)
ax.set_xlabel("epoch")
ax.set_ylabel("H(A)  [bits]")
ax.legend(fontsize=8)
ax.set_title(f"Entropy over training — n_genes={cfg.n_genes}")
plt.savefig(FIGURES / "entropy_over_training.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# ── final state: UMAP ────────────────────────────────────────────────────────
env, physics, ri = load_model(run_dir=RUN_DIR)
print(f"receptor_indices shape: {ri.shape}")
print(f"interface model: {getattr(env, 'use_interface_model', False)}")

fig, ax = plt.subplots(figsize=(7, 6))
plot_latent_umap(env, ri, ax=ax)
ax.set_title(f"Latent UMAP — R={ri.shape[0]}")
plt.savefig(FIGURES / "latent_umap.png", dpi=150, bbox_inches="tight")
plt.show()
# %%
