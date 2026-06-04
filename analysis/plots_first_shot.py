# %%
import sys, sqlite3, os
from pathlib import Path

exec_dir = "/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp"
sys.path.append(exec_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.IO import run_files

base_dir = Path(exec_dir + "/data/test")
db_path  = base_dir / "runs.db"
METRIC   = "full_array_entropy_mean"

# ── Step 1: load DB ───────────────────────────────────────────────────────────
# %%
with sqlite3.connect(db_path) as conn:
    df_db = pd.read_sql_query("SELECT * FROM runs WHERE status='complete'", conn)

# %%
# ── Step 2: split homomers / heteromers, keep only latest sweep each ──────────

def latest_per_R(df: pd.DataFrame, receptor_type: str, r_col: str) -> pd.DataFrame:
    sub = df[df["receptor_type"] == receptor_type].copy()
    if sub.empty:
        return pd.DataFrame()
    # for each R, keep only runs from the latest sweep that has that R
    latest = sub.groupby(r_col)["sweep_date"].transform("max")
    result = sub[sub["sweep_date"] == latest].copy()
    result["R"] = result[r_col]
    return result

homo_df = latest_per_R(df_db, "homomer",   "n_genes")
hete_df = latest_per_R(df_db, "heteromer", "n_receptors")

print(f"Homomers  : {len(homo_df)} runs  R ∈ {sorted(homo_df['R'].unique()) if not homo_df.empty else []}")
print(f"Heteromers: {len(hete_df)} runs  R ∈ {sorted(hete_df['R'].unique()) if not hete_df.empty else []}")

ARMS = [
    (homo_df, "n_genes",     "Homomers",   "#222222", "-"),
    (hete_df, "n_receptors", "Heteromers", "#d62728", "-"),
]
# %%

homo_df.columns

# ── Plot 1 — H(A) vs R ───────────────────────────────────────────────────────
# %%
fig, ax = plt.subplots(figsize=(7, 5))

for df, r_col, label, color, ls in ARMS:
    if df.empty or METRIC not in df.columns:
        print("no metric found")
        continue
    grp    = df.groupby("R")[METRIC]
    r_vals = sorted(df["R"].unique())
    med = [grp.get_group(r).median()       for r in r_vals]
    p10 = [grp.get_group(r).quantile(0.10) for r in r_vals]
    p90 = [grp.get_group(r).quantile(0.90) for r in r_vals]
    ax.plot(r_vals, med, color=color, ls=ls, lw=2, label=label)
    ax.fill_between(r_vals, p10, p90, color=color, alpha=0.2)

ax.plot(np.arange(1, 20, 1), np.arange(1, 20, 1),color='black',linestyle='--',label='perfect array',linewidth=1)

ax.set_xlabel("R  (receptors)")
ax.set_ylabel("H(A)  [bits]")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ── Plot 2 — H(A) vs epochs ──────────────────────────────────────────────────
# Load stats.csv directly per run using run_files — no sweep crawl needed.
# %%
data_root = str(base_dir)

def load_histories(df: pd.DataFrame, r_col: str) -> pd.DataFrame:
    parts = []
    for _, row in df.iterrows():
        p = run_files(row["path"], data_root)["stats"]
        if not os.path.exists(p):
            continue
        h = pd.read_csv(p)
        h["R"] = row[r_col]
        parts.append(h)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for ax, (df, r_col, label, color, ls) in zip(axes, ARMS):
    if df.empty:
        ax.set_title(f"{label}\n(no data)")
        continue

    hist = load_histories(df, r_col)
    # stats.csv uses bare metric names (no _mean suffix)
    loss_col = next((c for c in ["loss", "full_array_entropy"] if c in hist.columns), None)
    if hist.empty or loss_col is None:
        ax.set_title(f"{label}\n(no history)")
        continue

    r_vals   = sorted(hist["R"].unique())
    colors_r = plt.cm.viridis(np.linspace(0, 1, len(r_vals)))
    for r_val, c_r in zip(r_vals, colors_r):
        sub    = hist[hist["R"] == r_val].sort_values("epoch")
        grp    = sub.groupby("epoch")[loss_col]
        epochs = sorted(sub["epoch"].unique())
        mean_v = grp.mean().values
        std_v  = grp.std().fillna(0).values
        ax.plot(epochs, mean_v, color=c_r, lw=1.5)
        ax.fill_between(epochs, mean_v - std_v, mean_v + std_v, color=c_r, alpha=0.15)

    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(min(r_vals), max(r_vals)))
    plt.colorbar(sm, ax=ax, label="R")
    ax.set_title(label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(loss_col)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# %%
