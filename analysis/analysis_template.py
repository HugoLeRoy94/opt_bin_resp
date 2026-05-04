# %%
import sys
sys.path.append('../')
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch

from src.IO import SingleRunLoader, SweepLoader
from src.config import SingleRunConfig
from src.environment import LigandEnvironment, SymmetricLigandEnvironment, LogNormalConcentration
from src.physics import BinaryReceptor
from src.bin_loss import DiscreteExactLoss, DiscreteProxyLoss
from src.family_mi_loss import MaximizeMutualInformationFamilyLoss
from src.concentration_mi_loss import MaximizeMutualInformationConcentrationLoss
from src import plot_latent_umap

# Loss registry to reconstruct loss functions
LOSS_REGISTRY = {
    "exact": DiscreteExactLoss,
    "proxy": DiscreteProxyLoss,
    "family": MaximizeMutualInformationFamilyLoss,
    "conc": MaximizeMutualInformationConcentrationLoss
}

# Environment registry
ENV_REGISTRY = {
    "asymmetric": LigandEnvironment,
    "symmetric": SymmetricLigandEnvironment
}


def load_run_objects(run_dir: str) -> tuple:
    """
    Load and reconstruct objects from a single run directory.
    Returns: env, physics, loss_fn, receptor_indices, stats_df, config
    """
    loader = SingleRunLoader(run_dir)
    config = loader.load_config()
    checkpoint = loader.load_checkpoint(filename="best_model.pt", map_location="cpu")
    stats_df = loader.load_history()
    
    # Reconstruct concentration model
    conc_model = LogNormalConcentration(
        n_families=config.n_families,
        init_mean=config.init_means
    )
    
    # Reconstruct environment
    env_class = ENV_REGISTRY[config.env_type]
    env = env_class(
        n_units=config.n_units,
        n_families=config.n_families,
        conc_model=conc_model,
        latent_dim=config.latent_dim,
        shape_sigma=config.shape_sigma,
        avg_family_distance=config.average_family_distance,
        use_sensitivity=config.use_sensitivity
    )
    env.load_state_dict(checkpoint['env_state'])
    env.eval()
    
    # Reconstruct physics
    physics = BinaryReceptor(
        n_units=config.n_units,
        k_sub=config.k_sub,
        temperature=config.temperature
    )
    physics.load_state_dict(checkpoint['physics_state'])
    physics.eval()
    
    # Reconstruct loss function
    loss_class = LOSS_REGISTRY[config.loss_type]
    if config.loss_type == "exact":
        loss_fn = loss_class(entropy_type=config.entropy)
    elif config.loss_type == "proxy":
        loss_fn = loss_class(cov_weight=config.cov_weight)
    elif config.loss_type == "family":
        loss_fn = loss_class(entropy_type=config.entropy)
    elif config.loss_type == "conc":
        loss_fn = loss_class(entropy_type=config.entropy)
    else:
        loss_fn = loss_class()
    loss_fn.eval()
    
    # Get receptor indices
    receptor_indices = checkpoint['receptor_indices']
    if isinstance(receptor_indices, torch.Tensor):
        receptor_indices = receptor_indices.cpu()
    
    return env, physics, loss_fn, receptor_indices, stats_df, config


# %%

# Base paths for different experiment types
base_paths = {
    'homomers': Path("../data/homomers_w/"),
    'heteromers': Path("../data/heteromers/")
}

experiments = {}

for exp_type, base_data_path in base_paths.items():
    if not base_data_path.exists():
        continue
    
    # Find all sweep directories
    sweep_dirs = [d for d in base_data_path.iterdir() if d.is_dir() and 'sweep_' in d.name]
    
    for sweep_dir in sweep_dirs:
        # Load sweep config to get metadata
        try:
            sweep_loader = SweepLoader(str(sweep_dir))
            sweep_config = sweep_loader.config
        except Exception as e:
            print(f"Error loading sweep config from {sweep_dir}: {e}")
            continue
        
        # Traverse the sweep directory structure: dim_X/sample_Y/units_Z/
        for dim_dir in sweep_dir.glob("dim_*"):
            latent_dim = int(dim_dir.name.split("_")[1])
            
            for sample_dir in dim_dir.glob("sample_*"):
                sample_id = int(sample_dir.name.split("_")[1])
                
                for units_dir in sample_dir.glob("units_*"):
                    n_units = int(units_dir.name.split("_")[1])
                    
                    # For heteromers, try to extract n_receptors from directory
                    # In new system, n_receptors = n_units * k_sub
                    # But we need to load config first to get k_sub
                    try:
                        loader = SingleRunLoader(str(units_dir))
                        config = loader.load_config()
                        n_receptors = config.n_units * config.k_sub
                    except Exception as e:
                        print(f"Error loading config from {units_dir}: {e}")
                        continue
                    
                    config_key = (exp_type, sweep_config.n_families, latent_dim, n_units, n_receptors)
                    
                    if config_key not in experiments:
                        experiments[config_key] = {'samples': [], 'test_entropies': []}
                    
                    # Load objects from this run
                    try:
                        env, physics, loss_fn, receptor_indices, stats_df, config = load_run_objects(str(units_dir))
                        
                        # Load test results if available
                        test_json_path = units_dir / "test_results.json"
                        metrics = {
                            "full_array_entropy": [],
                            "mutual_information_family": None,
                            "mutual_information_concentration": None
                        }
                        
                        if test_json_path.exists():
                            with open(test_json_path, "r") as f:
                                data = json.load(f)
                                metrics["full_array_entropy"] = data.get("full_array_entropy", [])
                                metrics["mutual_information_family"] = data.get("mutual_information_family")
                                metrics["mutual_information_concentration"] = data.get("mutual_information_concentration")
                        
                        # Append to experiments dictionary
                        experiments[config_key]['samples'].append({
                            'env': env, 
                            'physics': physics, 
                            'receptor_indices': receptor_indices,
                            'stats_df': stats_df, 
                            'config': config,
                            'test_entropies': metrics["full_array_entropy"],
                            'mi_family': metrics["mutual_information_family"],
                            'mi_concentration': metrics["mutual_information_concentration"]
                        })
                        
                        experiments[config_key]['test_entropies'].extend(metrics["full_array_entropy"])
                        
                    except Exception as e:
                        print(f"Error loading {units_dir}: {e}")
                        pass

print(f"Loaded {len(experiments)} unique configurations.")
# %%
 
# Define dimensions to plot
dims_to_plot = [3,7,10]
families_to_plot = [5,10,30]
het_units_to_plot = [2,3,5,10]

# Define the metrics
metrics_info = [
    {'key': 'test_entropies', 'label': 'Test Entropy', 'title': 'Full Array Entropy'},
    {'key': 'mi_family', 'label': 'MI Family', 'title': 'Mutual Information (Family)'},
    {'key': 'mi_concentration', 'label': 'MI Concentration', 'title': 'Mutual Information (Concentration)'}
]

# Create a grid: Rows = Dimensions, Cols = Metrics
n_dims = len(dims_to_plot)
fig, axes = plt.subplots(nrows=n_dims, ncols=3, figsize=(3*4, 3 * n_dims), squeeze=False)

def get_plot_data(exp_dict, exp_type, fam, dim, metric_key, n_units=None):
    """Filters and sorts data for plotting."""
    x, y_means, y_stds = [], [], []
    for config_key, exp_data in exp_dict.items():
        etype, n_fam, l_dim, u, n_rec = config_key
        if etype == exp_type and n_fam == fam and l_dim == dim and (u == n_units if n_units else True):
            sample_vals = [s[metric_key] for s in exp_data['samples'] if s.get(metric_key) is not None]
            # Process values
            processed_vals = []
            for v in sample_vals:
                if isinstance(v, (list, np.ndarray)):
                    if len(v) > 0:
                        processed_vals.append(np.mean(v))
                else:
                    processed_vals.append(v)
            if processed_vals:
                x.append(n_rec)
                y_means.append(np.mean(processed_vals))
                y_stds.append(np.std(processed_vals))
    if not x: return None
    idx = np.argsort(x)
    return np.array(x)[idx], np.array(y_means)[idx], np.array(y_stds)[idx]

# 1. Define color mapping ONCE outside all loops
color_configs = []
for fam in families_to_plot:
    color_configs.append(('homomers', fam, None))
    for u in het_units_to_plot:
        color_configs.append(('heteromers', fam, u))

colors = plt.cm.tab20(np.linspace(0, 1, len(color_configs)))
color_map = {config: colors[i] for i, config in enumerate(color_configs)}

# --- Main Loop ---
for row_idx, target_dim in enumerate(dims_to_plot):
    for col_idx, m_info in enumerate(metrics_info):
        ax = axes[row_idx, col_idx]
        m_key = m_info['key']
        
        # 2. PLOT FIRST
        for fam in families_to_plot:
            # Plot Homomers
            c_key = ('homomers', fam, None)
            homo_data = get_plot_data(experiments, 'homomers', fam, target_dim, m_key)
            if homo_data:
                x, m, s = homo_data
                ax.errorbar(x[:-1], m[:-1], yerr=s[:-1], marker='o', capsize=3, 
                            color=color_map[c_key], label=f'Homo, Fam={fam}', 
                            linestyle='-', linewidth=2)
            
            # Plot Heteromers
            for u in het_units_to_plot:
                c_key = ('heteromers', fam, u)
                het_data = get_plot_data(experiments, 'heteromers', fam, target_dim, m_key, n_units=u)
                if het_data:
                    x, m, s = het_data
                    ax.errorbar(x[:-1], m[:-1], yerr=s[:-1], marker='x', capsize=3,
                                color=color_map[c_key], label=f'Het (u={u}), Fam={fam}', 
                                linestyle='--', alpha=0.7)
        
        # 3. FORMAT SECOND (Legend now has data to read!)
        if row_idx == 0:
            ax.set_title(f"{m_info['title']}", fontsize=14, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel(f"Dim {target_dim}\n\nBits", fontsize=12, fontweight='bold')
        if row_idx == n_dims - 1:
            ax.set_xlabel('Number of coding genes')
        
        #ax.set_xlim(0, 32)
        
        # Legend call works now because errorbars exist on the axis
        if row_idx == 0 and col_idx == 2:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

plt.tight_layout()
#plt.show()
plt.savefig('all_entropy.svg',bbox_inches='tight')
