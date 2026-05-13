# %%
"""
Comparative analysis: homomers vs heteromers.
Loads trained objects from both experiment types and plots entropy metrics
as a function of the number of receptors, stratified by n_families and latent_dim.

Homomer directories  (from opt_homomers.py  → SweepRunner):
  ../data/homomers_w/homomer_sweep_{timestamp}/
    latent_dim_{val}/sample_{id}/n_genes_{val}/

Heteromer directories (from opt_heteromers.py → custom loop):
  ../data/heteromers/families_{nf}/dim_{ld}/n_genes_{ng}/n_receptors_{nr}/sample_{s}/
"""
import sys
sys.path.append('../')
import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.IO import SingleRunLoader, SweepLoader, find_latest_sweep
from src.config import SingleRunConfig
from src.environment import LigandEnvironment, SymmetricLigandEnvironment, LogNormalConcentration, NormalConcentration
from src.physics import BinaryReceptor
from src.bin_loss import DiscreteExactLoss
from src.family_mi_loss import MaximizeMutualInformationLigandLoss
from src.concentration_mi_loss import MaximizeMutualInformationConcentrationLoss
from src.run import _build_loss

ENV_REGISTRY = {
    "asymmetric": LigandEnvironment,
    "symmetric":  SymmetricLigandEnvironment,
}


def load_run_objects(run_dir: str):
    """
    Load and reconstruct PyTorch objects from a single run directory.
    Returns: env, physics, loss_fn, receptor_indices, stats_df, config
    """
    loader     = SingleRunLoader(run_dir)
    config     = loader.load_config()
    checkpoint = loader.load_checkpoint(filename="best_model.pt", map_location="cpu")
    stats_df   = loader.load_history()

    conc_cls = NormalConcentration if config.conc_model_type == "normal" else LogNormalConcentration
    conc_model = conc_cls(
        n_ligands=config.n_ligands,
        init_mean=config.conc_mean,
        init_scale=config.conc_std,
    )

    env = ENV_REGISTRY[config.environment_geometry](
        config.n_genes,
        config.n_families,
        conc_model=conc_model,
        n_ligands=config.n_ligands,
        p_presence=config.p_presence,
        observation_noise_sigma=config.observation_noise_sigma,
        latent_dim=config.latent_dim,
        family_spread=config.family_spread,
        avg_family_distance=config.average_family_distance,
        affinity_length_scale=config.affinity_length_scale,
        distribution_type=config.distribution_type,
    )
    env.load_state_dict(checkpoint["env_state"])
    env.eval()

    physics = BinaryReceptor(config.n_genes, config.k_sub, temperature=config.temperature)
    physics.load_state_dict(checkpoint["physics_state"])
    physics.eval()

    loss_fn = _build_loss(config)
    loss_fn.eval()

    receptor_indices = checkpoint["receptor_indices"]
    if isinstance(receptor_indices, torch.Tensor):
        receptor_indices = receptor_indices.cpu()

    return env, physics, loss_fn, receptor_indices, stats_df, config


# %%  Load homomer experiments via SweepLoader.iter_run_dirs()

base_paths = {
    "homomers":  Path("../data/homomers_w"),
    "heteromers": Path("../data/heteromers"),
}

experiments = {}  # (exp_type, n_families, latent_dim, n_genes, n_receptors) → data dict

# --- Homomers ---
homo_base = base_paths["homomers"]
if homo_base.exists():
    for sweep_dir in sorted(homo_base.iterdir()):
        if not sweep_dir.is_dir():
            continue
        try:
            sweep_loader = SweepLoader(str(sweep_dir))
        except FileNotFoundError:
            continue

        for meta, single_cfg, run_dir in sweep_loader.iter_run_dirs():
            if not os.path.exists(run_dir):
                continue
            try:
                env, physics, loss_fn, receptor_indices, stats_df, config = load_run_objects(run_dir)
                # Homomers: n_receptors = n_genes (one receptor per gene)
                n_receptors = config.n_genes
                key = ("homomers", config.n_families, config.latent_dim,
                       config.n_genes, n_receptors)

                if key not in experiments:
                    experiments[key] = {"samples": [], "test_entropies": []}

                test_json = Path(run_dir) / "test_results.json"
                metrics   = {"full_array_entropy": []}
                if test_json.exists():
                    with open(test_json) as f:
                        data = json.load(f)
                    metrics["full_array_entropy"] = data.get("full_array_entropy", [])

                experiments[key]["samples"].append({
                    "env": env, "physics": physics,
                    "receptor_indices": receptor_indices,
                    "stats_df": stats_df, "config": config,
                    "test_entropies": metrics["full_array_entropy"],
                })
                experiments[key]["test_entropies"].extend(metrics["full_array_entropy"])

            except Exception as e:
                print(f"Error loading {run_dir}: {e}")

# --- Heteromers (custom directory structure from opt_heteromers.py) ---
het_base = base_paths["heteromers"]
if het_base.exists():
    for fam_dir in sorted(het_base.glob("families_*")):
        n_families = int(fam_dir.name.split("_")[1])
        for dim_dir in sorted(fam_dir.glob("dim_*")):
            latent_dim = int(dim_dir.name.split("_")[1])
            for ng_dir in sorted(dim_dir.glob("n_genes_*")):
                n_genes = int(ng_dir.name.split("_")[2])
                for nr_dir in sorted(ng_dir.glob("n_receptors_*")):
                    n_receptors = int(nr_dir.name.split("_")[2])
                    for sample_dir in sorted(nr_dir.glob("sample_*")):
                        try:
                            env, physics, loss_fn, receptor_indices, stats_df, config = \
                                load_run_objects(str(sample_dir))
                            key = ("heteromers", n_families, latent_dim, n_genes, n_receptors)
                            if key not in experiments:
                                experiments[key] = {"samples": [], "test_entropies": []}

                            test_json = sample_dir / "test_results.json"
                            metrics   = {"full_array_entropy": []}
                            if test_json.exists():
                                with open(test_json) as f:
                                    data = json.load(f)
                                metrics["full_array_entropy"] = data.get("full_array_entropy", [])

                            experiments[key]["samples"].append({
                                "env": env, "physics": physics,
                                "receptor_indices": receptor_indices,
                                "stats_df": stats_df, "config": config,
                                "test_entropies": metrics["full_array_entropy"],
                            })
                            experiments[key]["test_entropies"].extend(metrics["full_array_entropy"])

                        except Exception as e:
                            print(f"Error loading {sample_dir}: {e}")

print(f"Loaded {len(experiments)} unique configurations.")

# %% Plotting
dims_to_plot    = [3, 7, 10]
families_to_plot = [5, 10, 30]
het_units_to_plot = [2, 3, 5, 10]

metrics_info = [
    {"key": "test_entropies", "label": "Test Entropy", "title": "Full Array Entropy"},
]


def get_plot_data(exp_dict, exp_type, fam, dim, metric_key, n_units=None):
    x, y_means, y_stds = [], [], []
    for (etype, n_fam, l_dim, u, n_rec), exp_data in exp_dict.items():
        if etype != exp_type or n_fam != fam or l_dim != dim:
            continue
        if n_units is not None and u != n_units:
            continue
        sample_vals = [s[metric_key] for s in exp_data["samples"]
                       if s.get(metric_key) is not None]
        processed = []
        for v in sample_vals:
            if isinstance(v, (list, np.ndarray)):
                if len(v): processed.append(np.mean(v))
            else:
                processed.append(v)
        if processed:
            x.append(n_rec)
            y_means.append(np.mean(processed))
            y_stds.append(np.std(processed))
    if not x:
        return None
    idx = np.argsort(x)
    return np.array(x)[idx], np.array(y_means)[idx], np.array(y_stds)[idx]


color_configs = []
for fam in families_to_plot:
    color_configs.append(("homomers", fam, None))
    for u in het_units_to_plot:
        color_configs.append(("heteromers", fam, u))
colors    = plt.cm.tab20(np.linspace(0, 1, len(color_configs)))
color_map = {cfg: colors[i] for i, cfg in enumerate(color_configs)}

n_dims = len(dims_to_plot)
fig, axes = plt.subplots(nrows=n_dims, ncols=len(metrics_info),
                         figsize=(4 * len(metrics_info), 3 * n_dims), squeeze=False)

for row_idx, target_dim in enumerate(dims_to_plot):
    for col_idx, m_info in enumerate(metrics_info):
        ax   = axes[row_idx, col_idx]
        mkey = m_info["key"]

        for fam in families_to_plot:
            c_key     = ("homomers", fam, None)
            homo_data = get_plot_data(experiments, "homomers", fam, target_dim, mkey)
            if homo_data:
                x, m, s = homo_data
                ax.errorbar(x, m, yerr=s, marker="o", capsize=3,
                            color=color_map[c_key], label=f"Homo, Fam={fam}",
                            linestyle="-", linewidth=2)

            for u in het_units_to_plot:
                c_key    = ("heteromers", fam, u)
                het_data = get_plot_data(experiments, "heteromers", fam, target_dim, mkey, n_units=u)
                if het_data:
                    x, m, s = het_data
                    ax.errorbar(x, m, yerr=s, marker="x", capsize=3,
                                color=color_map[c_key], label=f"Het (u={u}), Fam={fam}",
                                linestyle="--", alpha=0.7)

        if row_idx == 0:
            ax.set_title(m_info["title"], fontsize=14, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel(f"Dim {target_dim}\n\nBits", fontsize=12, fontweight="bold")
        if row_idx == n_dims - 1:
            ax.set_xlabel("Number of receptors")
        if row_idx == 0 and col_idx == len(metrics_info) - 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")

plt.tight_layout()
plt.savefig("all_entropy.svg", bbox_inches="tight")
plt.show()
