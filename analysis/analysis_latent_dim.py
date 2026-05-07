# %%
"""
Analysis of latent dimension sweep results.
Plots how the entropy of the array evolves with different latent dimensions.
"""
import sys
sys.path.append('../')
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.IO import SweepLoader, SingleRunLoader

# %% Configuration
sweep_root = Path("../data/latent_dim_sweep")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Find the actual sweep directory (with sweep_config.json)
def find_sweep_directory(root):
    """Find the subdirectory containing sweep_config.json"""
    if (root / "sweep_config.json").exists():
        return root
    for subdir in root.iterdir():
        if subdir.is_dir() and (subdir / "sweep_config.json").exists():
            return subdir
    return root  # fallback

# Expected sweep parameters
n_families = 1
n_units_list = [1, 2, 3, 5, 7, 10, 20, 30]
latent_dim_list = [1, 2, 3, 5, 7, 10, 15, 20]
n_samples = 5

# %%
def load_sweep_data(sweep_path):
    """Load all run data from a sweep directory."""
    sweep_loader = SweepLoader(str(sweep_path))
    sweep_config = sweep_loader.config
    
    data = {
        'latent_dims': [],
        'n_units': [],
        'samples': [],
        'final_entropies': [],
        'converged': [],
        'full_trajectories': []
    }
    
    for meta, trajectory in sweep_config.generate_trajectories():
        latent_dim = meta['latent_dim']
        sample_id = meta['sample_id']
        
        for run_cfg in trajectory:
            n_units = run_cfg.n_units
            
            # Build expected run directory path - actual structure is sweep_path/dim_X/sample_Y/units_Z/
            run_dir = sweep_path / f"dim_{latent_dim}" / f"sample_{sample_id}" / f"units_{n_units}"
            
            if not run_dir.exists():
                print(f"Warning: {run_dir} not found, skipping")
                continue
            
            try:
                loader = SingleRunLoader(str(run_dir))
                stats_df = loader.load_history()
                
                if stats_df is not None and not stats_df.empty:
                    # Get full_array_entropy values
                    if 'full_array_entropy' in stats_df.columns:
                        entropy_vals = stats_df['full_array_entropy'].dropna().values
                        final_entropy = entropy_vals[-1] if len(entropy_vals) > 0 else None
                        
                        data['latent_dims'].append(latent_dim)
                        data['n_units'].append(n_units)
                        data['samples'].append(sample_id)
                        data['final_entropies'].append(final_entropy)
                        data['converged'].append(final_entropy is not None)
                        data['full_trajectories'].append(entropy_vals.tolist())
                    else:
                        print(f"Warning: full_array_entropy not in columns for {run_dir}")
            except Exception as e:
                print(f"Error loading {run_dir}: {e}")
    
    return data

# %%
print(f"Loading data from {sweep_root}...")
if sweep_root.exists():
    actual_sweep_dir = find_sweep_directory(sweep_root)
    sweep_data = load_sweep_data(actual_sweep_dir)
    print(f"Loaded {len(sweep_data['final_entropies'])} runs")
else:
    print(f"Sweep directory {sweep_root} not found. Using placeholder data for template.")
    sweep_data = {
        'latent_dims': [],
        'n_units': [],
        'samples': [],
        'final_entropies': [],
        'converged': [],
        'full_trajectories': []
    }

# %%
# 2x2 grid: mean and std of test entropy vs units and vs dimension
latent_dims = np.array(sweep_data['latent_dims'])
n_units_arr = np.array(sweep_data['n_units'])
final_entropies = np.array(sweep_data['final_entropies'])

unique_latent_dims = sorted(set(latent_dims))
unique_n_units = sorted(set(n_units_arr))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# === Plot (0,0): Mean test entropy vs n_units, one curve per dimension ===
ax = axes[0, 0]
colors_ld = plt.cm.viridis(np.linspace(0, 1, len(unique_latent_dims)))

for ld, color in zip(unique_latent_dims, colors_ld):
    mean_entropies = []
    n_units_values = []
    
    for nu in unique_n_units:
        mask = (latent_dims == ld) & (n_units_arr == nu)
        entropies = final_entropies[mask]
        entropies = entropies[~np.isnan(entropies)]
        
        if len(entropies) > 0:
            n_units_values.append(nu)
            mean_entropies.append(np.mean(entropies))
    
    if len(n_units_values) > 0:
        ax.plot(n_units_values, mean_entropies, marker='o', color=color, label=f'latent_dim={ld}', linewidth=2)

ax.set_xlabel('Number of Units')
ax.set_ylabel('Mean Test Entropy (bits)')
ax.set_title('Mean Test Entropy vs Number of Units')
ax.legend(fontsize='small')
ax.grid(True, alpha=0.3)

# === Plot (0,1): Mean test entropy vs dimension, one curve per n_units ===
ax = axes[0, 1]
colors_nu = plt.cm.viridis(np.linspace(0, 1, len(unique_n_units)))

for nu, color in zip(unique_n_units, colors_nu):
    mean_entropies = []
    dim_values = []
    
    for ld in unique_latent_dims:
        mask = (latent_dims == ld) & (n_units_arr == nu)
        entropies = final_entropies[mask]
        entropies = entropies[~np.isnan(entropies)]
        
        if len(entropies) > 0:
            dim_values.append(ld)
            mean_entropies.append(np.mean(entropies))
    
    if len(dim_values) > 0:
        ax.plot(dim_values, mean_entropies, marker='o', color=color, label=f'n_units={nu}', linewidth=2)

ax.set_xlabel('Latent Dimension')
ax.set_ylabel('Mean Test Entropy (bits)')
ax.set_title('Mean Test Entropy vs Latent Dimension')
ax.legend(fontsize='small')
ax.grid(True, alpha=0.3)

# === Plot (1,0): Std of test entropy vs n_units, one curve per dimension ===
ax = axes[1, 0]

for ld, color in zip(unique_latent_dims, colors_ld):
    std_entropies = []
    n_units_values = []
    
    for nu in unique_n_units:
        mask = (latent_dims == ld) & (n_units_arr == nu)
        entropies = final_entropies[mask]
        entropies = entropies[~np.isnan(entropies)]
        
        if len(entropies) > 0:
            n_units_values.append(nu)
            std_entropies.append(np.std(entropies))
    
    if len(n_units_values) > 0:
        ax.plot(n_units_values, std_entropies, marker='o', color=color, label=f'latent_dim={ld}', linewidth=2)

ax.set_xlabel('Number of Units')
ax.set_ylabel('Std Test Entropy (bits)')
ax.set_title('Std Test Entropy vs Number of Units')
ax.legend(fontsize='small')
ax.grid(True, alpha=0.3)

# === Plot (1,1): Std of test entropy vs dimension, one curve per n_units ===
ax = axes[1, 1]

for nu, color in zip(unique_n_units, colors_nu):
    std_entropies = []
    dim_values = []
    
    for ld in unique_latent_dims:
        mask = (latent_dims == ld) & (n_units_arr == nu)
        entropies = final_entropies[mask]
        entropies = entropies[~np.isnan(entropies)]
        
        if len(entropies) > 0:
            dim_values.append(ld)
            std_entropies.append(np.std(entropies))
    
    if len(dim_values) > 0:
        ax.plot(dim_values, std_entropies, marker='o', color=color, label=f'n_units={nu}', linewidth=2)

ax.set_xlabel('Latent Dimension')
ax.set_ylabel('Std Test Entropy (bits)')
ax.set_title('Std Test Entropy vs Latent Dimension')
ax.legend(fontsize='small')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Plot entropy vs epoch with subplots per dimension
# For each latent_dim, plot one curve per n_units with error bars from samples

full_trajectories = sweep_data['full_trajectories']

# Find max number of epochs across all trajectories
max_epochs = max(len(traj) for traj in full_trajectories if traj)

# Create subplots: one per latent dimension
n_dims = len(unique_latent_dims)
fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4 * n_dims))

if n_dims == 1:
    axes = [axes]  # Ensure axes is iterable

colors = plt.cm.viridis(np.linspace(0, 1, len(unique_n_units)))

for dim_idx, ld in enumerate(unique_latent_dims):
    ax = axes[dim_idx]
    
    # For each n_units, compute mean and std trajectory across samples for this dimension
    for nu, color in zip(unique_n_units, colors):
        # Filter trajectories for this (latent_dim, n_units) combination
        trajectories = []
        for ld_val, nu_val, traj in zip(latent_dims, n_units_arr, full_trajectories):
            if ld_val == ld and nu_val == nu and traj:
                trajectories.append(traj)
        
        if not trajectories:
            continue
        
        # Align trajectories to same length
        aligned = np.array([traj + [np.nan] * (max_epochs - len(traj)) for traj in trajectories])
        mean_traj = np.nanmean(aligned, axis=0)
        std_traj = np.nanstd(aligned, axis=0)
        
        epochs = np.arange(len(mean_traj))
        valid_mask = ~np.isnan(mean_traj)
        
        ax.plot(epochs[valid_mask], mean_traj[valid_mask], 
                color=color, label=f'n_units={nu}', linewidth=2)
        ax.fill_between(epochs[valid_mask], 
                        mean_traj[valid_mask] - std_traj[valid_mask],
                        mean_traj[valid_mask] + std_traj[valid_mask],
                        color=color, alpha=0.2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Entropy (bits)')
    ax.set_title(f'Entropy vs Epoch (latent_dim={ld})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
