# %%
import sys, sqlite3, os
from pathlib import Path

exec_dir = "/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp"
sys.path.append(exec_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.IO import run_files

METRIC = "full_array_entropy_blocked_mean"

dfs = []
for sweep in ["sweepD5", "sweepD10"]:
    db_path = Path(exec_dir) / "data" / sweep / "runs.db"
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM runs WHERE status='complete'", conn)
    df["sweep"] = sweep
    dfs.append(df)
df_db = pd.concat(dfs, ignore_index=True)

# %%
for sweep, group in df_db.groupby("sweep"):
    g = group.sort_values('mu_ligands_per_source')
    plt.plot(g['mu_ligands_per_source'], g[METRIC], label=sweep)
plt.legend()
# %%
