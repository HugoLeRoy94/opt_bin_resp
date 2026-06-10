# %%
import sys, sqlite3, os
from pathlib import Path

exec_dir = "/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp"
sys.path.append(exec_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.IO import run_files

base_dir  = Path(exec_dir + "/data/fig1")
db_path   = base_dir / "runs.db"
data_root = str(base_dir)

METRIC   = "full_array_entropy_blocked_mean"   # DB column (final value per run)
LOSS_COL = "full_array_entropy_blocked"        # stats.csv column (per epoch)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — load all complete runs (no latest-sweep filter)
# ─────────────────────────────────────────────────────────────────────────────
# %%
with sqlite3.connect(db_path) as conn:
    df_db = pd.read_sql_query("SELECT * FROM runs WHERE status='complete'", conn)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — env_id key + receptor axis R
# env_id encodes the four environmental parameters that vary across runs:
#   n_families, latent_dim, family_spread, average_family_distance
# ─────────────────────────────────────────────────────────────────────────────
# %%
def make_env_id(row) -> str:
    return (f"nf{row.n_families}_ld{row.latent_dim}"
            f"_sp{round(row.family_spread, 4):.4f}"
            f"_dist{round(row.average_family_distance, 4):.4f}")

df_db["env_id"] = df_db.apply(make_env_id, axis=1)
# fallback: derive sweep_folder from path when the DB column is not yet backfilled
if "sweep_folder" not in df_db.columns or df_db["sweep_folder"].isna().all():
    df_db["sweep_folder"] = df_db["path"].str.split("/").str[0]
else:
    df_db["sweep_folder"] = df_db["sweep_folder"].fillna(
        df_db["path"].str.split("/").str[0])


# homomers: R = n_genes (number of distinct receptor types in the array)
homo_df = df_db[df_db["receptor_type"] == "homomer"].copy()
homo_df["R"] = homo_df["n_genes"]
print(f"homo_df: {len(homo_df)} rows, {homo_df['env_id'].nunique()} env_ids")
print(homo_df.groupby(["env_id", "R"]).size().rename("count").reset_index().to_string())


# heteromers: R = n_receptors; arms split by number of gene variants
hete_frames: dict[int, pd.DataFrame] = {}
for g in [3, 5, 10]:
    sub = df_db[(df_db["receptor_type"] == "heteromer") & (df_db["n_genes"] == g)].copy()
    sub["R"] = sub["n_receptors"]
    hete_frames[g] = sub

# viridis color per arm, keyed by n_genes (homomers treated as 1)
_ARM_NG  = [1, 3, 5, 10]
_vir     = plt.colormaps["viridis"]
_arm_clr = [_vir(ng / max(_ARM_NG)) for ng in _ARM_NG]

ARMS: dict[str, tuple[pd.DataFrame, tuple]] = {
    "Homomers":              (homo_df,         _arm_clr[0]),
    "Heteromers (3 genes)":  (hete_frames[3],  _arm_clr[1]),
    "Heteromers (5 genes)":  (hete_frames[5],  _arm_clr[2]),
    "Heteromers (10 genes)": (hete_frames[10], _arm_clr[3]),
}

for name, (df, _) in ARMS.items():
    print(f"{name:25s}: {len(df):4d} runs, "
          f"{df['env_id'].nunique():3d} env conditions, "
          f"R ∈ [{df['R'].min()}, {df['R'].max()}]")

# %%
print( set(ARMS['Heteromers (10 genes)'][0]['sweep_folder']).__len__())
# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Individual H vs R, one panel per arm
# Each line = one environmental condition
# ─────────────────────────────────────────────────────────────────────────────
# %%
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

r_ref = np.arange(1, 50)

for ax, (name, (df, color)) in zip(axes, ARMS.items()):
    if df.empty or METRIC not in df.columns:
        ax.set_title(f"{name}\n(no data)")
        continue

    for env_id in sorted(df["env_id"].unique()):
        sub = (df[df["env_id"] == env_id]
               .groupby("R")[METRIC].mean()
               .reset_index()
               .sort_values("R"))
        ax.plot(sub["R"].values, sub[METRIC].values, color=color, lw=1, alpha=0.6)

    ax.plot(r_ref, r_ref, "k--", lw=0.8, label="perfect array")
    ax.set_title(name)
    ax.set_xlabel("R  (receptors)")
    ax.set_ylabel("H(A)  [bits]")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Summary: all arms, mean ± std over env conditions per R
# ─────────────────────────────────────────────────────────────────────────────
# %%
fig, ax = plt.subplots(figsize=(4, 3))

for name, (df, color) in ARMS.items():
    if df.empty or METRIC not in df.columns:
        continue
    grp    = df.groupby("R")[METRIC]
    r_vals = sorted(df["R"].unique())
    mean_v = np.array([grp.get_group(r).mean() for r in r_vals])
    std_v  = np.array([grp.get_group(r).std()  for r in r_vals])
    ax.plot(r_vals, mean_v, color=color, lw=2, label=name)
    ax.fill_between(r_vals, mean_v - std_v, mean_v + std_v,
                    color=color, alpha=0.2)

ax.plot(r_ref, r_ref, "k--", lw=1, label="perfect array")
ax.set_xlabel("R  (receptors)",fontsize=9)
ax.set_ylabel("H(A)  [bits]",fontsize=9)
ax.legend(fontsize=9)
#ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('MI_R50.svg')
#plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Convergence: H(A) vs epoch
# One PNG per (arm × env_condition), saved to data/fig1/convergence/{arm}/
# Within each figure: one curve per R value, viridis colormap
# ─────────────────────────────────────────────────────────────────────────────
# %%
def load_histories(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, row in df.iterrows():
        p = run_files(row["path"], data_root)["stats"]
        if not os.path.exists(p):
            continue
        h = pd.read_csv(p)
        h["R"] = row["R"]
        parts.append(h)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

ARM_SLUGS = {
    "Homomers":              "homomers",
    "Heteromers (3 genes)":  "hete_ng3",
    "Heteromers (5 genes)":  "hete_ng5",
    "Heteromers (10 genes)": "hete_ng10",
}

conv_root = base_dir / "convergence"

for name, (df, _) in ARMS.items():
    if df.empty:
        continue

    out_dir = conv_root / ARM_SLUGS[name]
    out_dir.mkdir(parents=True, exist_ok=True)

    env_ids = sorted(df["env_id"].unique())
    print(f"{name}: saving {len(env_ids)} figures → {out_dir}")

    for env_id in env_ids:
        sub  = df[df["env_id"] == env_id]
        hist = load_histories(sub)

        loss_col = next(
            (c for c in [LOSS_COL, "full_array_entropy", "loss"] if c in hist.columns),
            None)
        if hist.empty or loss_col is None:
            continue

        r_vals   = sorted(hist["R"].unique())
        colors_r = plt.cm.viridis(np.linspace(0, 1, max(len(r_vals), 2)))

        fig, ax = plt.subplots(figsize=(6, 4))
        for r_val, c_r in zip(r_vals, colors_r):
            h_r    = hist[hist["R"] == r_val].sort_values("epoch")
            grp    = h_r.groupby("epoch")[loss_col]
            epochs = sorted(h_r["epoch"].unique())
            mean_v = grp.mean().values
            std_v  = grp.std().fillna(0).values
            ax.plot(epochs, mean_v, color=c_r, lw=1.5)
            ax.fill_between(epochs, mean_v - std_v, mean_v + std_v,
                            color=c_r, alpha=0.15)

        sm = plt.cm.ScalarMappable(cmap="viridis",
                                   norm=plt.Normalize(min(r_vals), max(r_vals)))
        plt.colorbar(sm, ax=ax, label="R")
        ax.set_title(f"{name}  |  {env_id}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("H(A)  [bits]")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{env_id}.png", dpi=100)
        plt.close(fig)

print("Done.")
# %%
