# %%

import sys
sys.path.append('/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp')
import numpy as np
import matplotlib.pyplot as plt
from analysis.plotlib import load_runs, load_epochs, plot_metric, latest_sweep

GOAL = 'mix_size'
METRIC = "full_array_entropy_blocked_mean"
METRIC_LO = "full_array_entropy_mean"
MIX_SIZE = np.arange(1,11,1)

df = load_runs(GOAL)

# %%
print(set(df['sweep_folder']))
homo = latest_sweep(df[df['receptor_type'] == 'homomer'])
hete = df[df['sweep_folder'] == 'het_ng15_20260623_232423']
print(homo['sweep_folder'])
# %%
ax = plot_metric(homo,y=METRIC,x='mu_ligands_per_source')
plot_metric(homo,y=METRIC_LO,x='mu_ligands_per_source',ax=ax,ls='--')
plot_metric(hete,y=METRIC,x='mu_ligands_per_source',group="n_genes",ax=ax)
plot_metric(hete,y=METRIC_LO,x='mu_ligands_per_source',group="n_genes",ax=ax,ls='--')
# %%
ax = plot_metric(df,y=METRIC_LO,x='mu_ligands_per_source',group="n_genes")
# %%
ep = load_epochs(df)
ax = plot_metric(ep, y="full_array_entropy_blocked", x="epoch",
                 group="mu_ligands_per_source", cmap="viridis")
# %%
ep = load_epochs(homo)
ax = plot_metric(ep, y="full_array_entropy_blocked", x="epoch",
                 group="mu_ligands_per_source", cmap="viridis", err=None)
# %%
