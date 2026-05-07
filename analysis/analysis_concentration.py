# %%
"""
Analysis of family distance sweep results.
Plots final test entropy as a function of family distance.
One curve per shape_sigma, with error bars via samples.
Grid: rows = n_families, columns = n_units.
"""
import sys
sys.path.append('../')
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re

from src.IO import SingleRunLoader

# %% Configuration
sweep_root = Path("../data/fam_distance_sweep")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# %%
def parse_sweep_dirname(dirname):
    """Parse family distance sweep directory name.
    Expected format: fam_dist_nf{NF}_ss{SS}_dist{DIST}_{TIMESTAMP}
    """
    pattern = r'fam_dist_nf(\d+)_ss([\d.]+)_dist([\d.]+)_(\d{8}_\d{6})'
    match = re.match(pattern, dirname)
    if match:
        n_families = int(match.group(1))
        shape_sigma = float(match.group(2))
        family_distance = float(match.group(3))
        return n_families, shape_sigma, family_distance
    return None, None, None

def load_sweep_data(sweep_path):
    """Load all run data from the family distance sweep directory."""
    data = {
        'n_families': [],
        'shape_sigmas': [],
        'family_distances': [],
        'n_units': [],
        'samples': [],
        'final_entropies': [],
        'full_trajectories': []
    }
    
    if not sweep_path.exists():
        return data
    
    for sweep_dir in sorted(sweep_path.iterdir()):
        if not sweep_dir.is_dir() or not sweep_dir.name.startswith("fam_dist_"):
            continue
        
        n_families, shape_sigma, family_distance = parse_sweep_dirname(sweep_dir.name)
        if n_families is None:
            print(f"Warning: Could not parse directory name {sweep_dir.name}")
            continue
        
        dim_dir = sweep_dir / "dim_3"
        if not dim_dir.exists():
            continue
        
        for sample_dir in sorted(dim_dir.glob("sample_*")):
            sample_id = int(sample_dir.name.split("_")[1])
            
            for units_dir in sorted(sample_dir.glob("units_*")):
                n_units = int(units_dir.name.split("_")[1])
                
                try:
                    loader = SingleRunLoader(str(units_dir))
                    stats_df = loader.load_history()
                    
                    if stats_df is not None and not stats_df.empty:
                        if 'full_array_entropy' in stats_df.columns:
                            entropy_vals = stats_df['full_array_entropy'].dropna().values
                            final_entropy = entropy_vals[-1] if len(entropy_vals) > 0 else None
                            
                            data['n_families'].append(n_families)
                            data['shape_sigmas'].append(shape_sigma)
                            data['family_distances'].append(family_distance)
                            data['n_units'].append(n_units)
                            data['samples'].append(sample_id)
                            data['final_entropies'].append(final_entropy)
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
        'n_families': [],
        'shape_sigmas': [],
        'family_distances': [],
        'n_units': [],
        'samples': [],
        'final_entropies': [],
        'full_trajectories': []
    }

# %%
# Grid plot: rows = n_families, columns = n_units
# Each cell: test entropy vs family distance, one curve per shape_sigma

family_distances = np.array(sweep_data['family_distances'])
shape_sigmas = np.array(sweep_data['shape_sigmas'])
n_families_arr = np.array(sweep_data['n_families'])
n_units_arr = np.array(sweep_data['n_units'])
final_entropies = np.array(sweep_data['final_entropies'])

unique_n_families = sorted(set(n_families_arr))
unique_n_units = sorted(set(n_units_arr))
unique_shape_sigmas = sorted(set(shape_sigmas))

n_rows = len(unique_n_families)
n_cols = len(unique_n_units)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

if n_rows == 1 and n_cols == 1:
    axes = np.array([[axes]])
elif n_rows == 1:
    axes = np.array([axes])
elif n_cols == 1:
    axes = np.array([[ax] for ax in axes])

colors = plt.cm.viridis(np.linspace(0, 1, len(unique_shape_sigmas)))

for row_idx, nf in enumerate(unique_n_families):
    for col_idx, nu in enumerate(unique_n_units):
        ax = axes[row_idx, col_idx] if (n_rows > 1 and n_cols > 1) else axes[row_idx][col_idx]
        
        for ss, color in zip(unique_shape_sigmas, colors):
            # Filter data for this (n_families, n_units, shape_sigma) combination
            mask = (n_families_arr == nf) & (n_units_arr == nu) & (shape_sigmas == ss)
            fd_values = family_distances[mask]
            entropies = final_entropies[mask]
            
            valid_mask = ~np.isnan(entropies)
            fd_values = fd_values[valid_mask]
            entropies = entropies[valid_mask]
            
            if len(fd_values) == 0:
                continue
            
            # For each unique family_distance, compute mean and std of entropies
            unique_fds = sorted(set(fd_values))
            mean_entropies = []
            std_entropies = []
            fd_list = []
            
            for fd in unique_fds:
                fd_mask = fd_values == fd
                fd_entropies = entropies[fd_mask]
                if len(fd_entropies) > 0:
                    fd_list.append(fd)
                    mean_entropies.append(np.mean(fd_entropies))
                    std_entropies.append(np.std(fd_entropies))
            
            if len(fd_list) > 0:
                ax.errorbar(fd_list, mean_entropies, yerr=std_entropies,
                            marker='o', capsize=5, color=color, label=f'ss={ss}',
                            linewidth=2, capthick=2)
        
        ax.set_xlabel('Family Distance')
        ax.set_ylabel('Test Entropy (bits)')
        ax.set_title(f'n_fam={nf}, n_units={nu}')
        if col_idx == n_cols - 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
