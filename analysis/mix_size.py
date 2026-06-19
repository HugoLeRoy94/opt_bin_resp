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
df['mu_ligands_per_source']
# %%
ax = plot_metric(df,y=METRIC,x='mu_ligands_per_source',group="n_genes")
# %%
ax = plot_metric(df,y=METRIC_LO,x='mu_ligands_per_source',group="n_genes")
# %%
ep = load_epochs(df)
ax = plot_metric(ep, y="full_array_entropy_blocked", x="epoch",
                 group="mu_ligands_per_source", cmap="viridis")
# %%
