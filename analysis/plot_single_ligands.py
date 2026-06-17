# %%
"""Fig 1 plots — entropy/MI of receptor arrays, built on analysis.plotlib.

Edit METRIC to switch what is plotted; everything else is generic.
"""
import sys
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")  # exec dir

import numpy as np
import matplotlib.pyplot as plt

from analysis.plotlib import load_runs, load_epochs, plot_metric, latest_sweep

GOAL      = "fig1_single_ligand"
METRIC    = "full_array_entropy_blocked_mean"   # upper bound on MI
METRIC_LO = "full_array_entropy_mean"           # lower bound on MI
GENES     = [3]

homo = load_runs(GOAL, receptor_type="homomer",   entropy="shannon")
hete = load_runs(GOAL, receptor_type="heteromer", entropy="shannon")

r_ref = np.arange(1, 25)
# %%
hete = latest_sweep(hete)
homo = latest_sweep(homo)
print(set(hete['sweep_folder']))
print(set(homo['sweep_folder']))

# %%
ax = plot_metric(homo,y = METRIC,x = 'R')
plot_metric(homo,y = METRIC_LO,x = 'R',ax=ax)
plot_metric(hete,y = METRIC,x = 'R',ax=ax)
plot_metric(hete,y = METRIC_LO,x = 'R',ax=ax)
ax.plot(r_ref, r_ref, "k--", lw=.8)
# %%

ep = load_epochs(hete)
ax = plot_metric(ep, y="full_array_entropy_blocked", x="epoch",
                 group="R", cmap="viridis", err=None)
# %%
ep = load_epochs(homo)
ax = plot_metric(ep, y="full_array_entropy_blocked", x="epoch",
                 group="R", cmap="viridis", err=None)