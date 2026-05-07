# %%
"""
Analysis of batch size sweep results.
Plots how the entropy of the array evolves with different batch sizes.
"""
import sys
sys.path.append('../')
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.IO import SweepLoader, SingleRunLoader

# %% Configuration
sweep_root = Path("../data/batch_size_sweep")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Expected sweep parameters
n_families = 5
n_units = 10
latent_dim = 3
batch_size_list = [2**i for i in range(5, 16)]  # 32 to 32768
n_samples = 5

# %%
def load_sweep_data(sweep_path):
    """Load all run data from a sweep directory."""
    data = {
        'batch_sizes': [],
        'samples': [],
        'final_entropies': [],
        'converged': [],
        'full_trajectories': []
    }
    
    # The actual sweep directory structure is:
    # batch_size_sweep/batch_size_XXX_YYYYMMDD_HHMMSS/dim_3/sample_Y/units_10/
    if not sweep_path.exists():
        return data
    
    for batch_size_dir in sorted(sweep_path.iterdir()):
        if not batch_size_dir.is_dir() or not batch_size_dir.name.startswith("batch_size_"):
            continue
        
        # Extract batch size from directory name like "batch_size_32_20260504_151250"
        try:
            batch_size = int(batch_size_dir.name.split("_")[2])
        except (IndexError, ValueError):
            continue
        
        # Navigate to dim_3 directory
        dim_dir = batch_size_dir / "dim_3"
        if not dim_dir.exists():
            continue
        
        for sample_dir in sorted(dim_dir.glob("sample_*")):
            sample_id = int(sample_dir.name.split("_")[1])
            
            # Navigate to units_10 directory (the actual run directory)
            units_dir = sample_dir / "units_10"
            if not units_dir.exists():
                continue
            
            try:
                loader = SingleRunLoader(str(units_dir))
                stats_df = loader.load_history()
                
                if stats_df is not None and not stats_df.empty:
                    if 'full_array_entropy' in stats_df.columns:
                        entropy_vals = stats_df['full_array_entropy'].dropna().values
                        final_entropy = entropy_vals[-1] if len(entropy_vals) > 0 else None
                        
                        data['batch_sizes'].append(batch_size)
                        data['samples'].append(sample_id)
                        data['final_entropies'].append(final_entropy)
                        data['converged'].append(final_entropy is not None)
                        data['full_trajectories'].append(entropy_vals.tolist())
            except Exception as e:
                print(f"Error loading {units_dir}: {e}")
    
    return data

# %%
print(f"Loading data from {sweep_root}...")
if sweep_root.exists():
    sweep_data = load_sweep_data(sweep_root)
    print(f"Loaded {len(sweep_data['final_entropies'])} runs")
else:
    print(f"Sweep directory {sweep_root} not found. Using placeholder data for template.")
    sweep_data = {
        'batch_sizes': [],
        'samples': [],
        'final_entropies': [],
        'converged': [],
        'full_trajectories': []
    }

# %%
# Plot entropy as a function of epoch for each batch size with different colors and error bars
batch_sizes = np.array(sweep_data['batch_sizes'])
final_entropies = np.array(sweep_data['final_entropies'])
full_trajectories = sweep_data['full_trajectories']

unique_batch_sizes = sorted(set(batch_sizes))

# Find max number of epochs across all trajectories
max_epochs = max(len(traj) for traj in full_trajectories if traj)

# Create arrays for mean and std entropy at each epoch for each batch size
mean_trajectories = {}
std_trajectories = {}

for bs in unique_batch_sizes:
    # Get all trajectories for this batch size
    bs_trajectories = [traj for bs_val, traj in zip(batch_sizes, full_trajectories) if bs_val == bs and traj]
    
    if not bs_trajectories:
        continue
    
    # Align trajectories to same length by padding with NaN
    aligned = np.array([traj + [np.nan] * (max_epochs - len(traj)) for traj in bs_trajectories])
    
    mean_trajectories[bs] = np.nanmean(aligned, axis=0)
    std_trajectories[bs] = np.nanstd(aligned, axis=0)

# Plot
fig, ax = plt.subplots(figsize=(12, 7))

colors = plt.cm.viridis(np.linspace(0, 1, len(unique_batch_sizes)))

for bs, color in zip(unique_batch_sizes, colors):
    if bs in mean_trajectories:
        epochs = np.arange(len(mean_trajectories[bs]))
        mean_traj = mean_trajectories[bs]
        std_traj = std_trajectories[bs]
        
        # Only plot non-NaN values
        valid_mask = ~np.isnan(mean_traj)
        ax.plot(epochs[valid_mask], mean_traj[valid_mask], 
                color=color, label=f'BS={bs}', linewidth=2)
        ax.fill_between(epochs[valid_mask], 
                        mean_traj[valid_mask] - std_traj[valid_mask],
                        mean_traj[valid_mask] + std_traj[valid_mask],
                        color=color, alpha=0.2)

ax.set_xlabel('Epoch')
ax.set_ylabel('Entropy (bits)')
ax.set_title('Entropy vs Epoch for Each Batch Size')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Histogram of full entropy trajectory values for each batch size on the same graph with empty bars
fig, ax = plt.subplots(figsize=(12, 7))

for bs, color in zip(unique_batch_sizes, colors):
    # Get all entropy values across all epochs for this batch size
    bs_all_entropies = []
    for bs_val, traj in zip(batch_sizes, full_trajectories):
        if bs_val == bs and traj:
            bs_all_entropies.extend(traj)
    
    bs_all_entropies = np.array(bs_all_entropies)
    bs_all_entropies = bs_all_entropies[~np.isnan(bs_all_entropies)]
    
    if len(bs_all_entropies) > 0:
        # Use step histogram (empty bars) with density normalization
        n, bins, patches = ax.hist(bs_all_entropies, bins=20, 
                                   histtype='step', color=color, 
                                   label=f'BS={bs}', linewidth=2,
                                   density=True)

ax.set_xlabel('Entropy (bits)')
ax.set_ylabel('Density')
ax.set_title('Full Entropy Trajectory Distribution for Each Batch Size')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Scatter plot: test entropy vs batch size (one point per sample)
fig, ax = plt.subplots(figsize=(12, 7))

samples = np.array(sweep_data['samples'])
unique_samples = sorted(set(samples))
sample_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_samples)))

for sample_id, color in zip(unique_samples, sample_colors):
    mask = samples == sample_id
    bs_values = batch_sizes[mask]
    entropy_values = final_entropies[mask]
    
    # Filter out NaN values
    valid_mask = ~np.isnan(entropy_values)
    ax.scatter(bs_values[valid_mask], entropy_values[valid_mask], 
               color=color, label=f'Sample {sample_id}', alpha=0.7, s=80)

ax.set_xscale('log', base=2)
ax.set_xlabel('Batch Size (log scale, base 2)')
ax.set_ylabel('Test Entropy (bits)')
ax.set_title('Test Entropy vs Batch Size (one point per sample)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.show()

# %%
