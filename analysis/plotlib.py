"""
plotlib.py — thin loading + plotting layer over the per-goal runs.db index.

Data layout on disk:

    data/<goal>/<n_genes>_<date>/<env conditions>.../<receptor sweep>/run_*/

`data/<goal>/runs.db` is a flat table with one row per run: every scalar config
field is a column, plus metadata (`sweep_folder` = the `<n_genes>_<date>` dir,
`sweep_name`, `sweep_date`, `git_hash`, `receptor_type`, `status`) and one
`<metric>_mean` column per metric.  Everything below is just filtering that
table and aggregating it.

Typical use
-----------
    from analysis.plotlib import load_runs, load_run, load_epochs, plot_metric

    M = "full_array_entropy_blocked_mean"

    # all heteromers in goal "fig1", Renyi entropy, one curve per gene count
    df = load_runs("fig1", receptor_type="heteromer", entropy="renyi")
    plot_metric(df, y=M, x="R", group="n_genes", cmap="viridis")

    # one specific sweep (goal + n_genes + date)
    df = load_runs("fig1", n_genes=20, date="20260612")

    # metric vs epoch, one curve per R
    ep = load_epochs(df)
    plot_metric(ep, y="full_array_entropy_blocked", x="epoch",
                group="R", cmap="viridis")

    # a single run
    cfg, hist = load_run("fig1", n_genes=5, n_receptors=10)
"""
import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_EXEC_DIR = Path(__file__).resolve().parents[1]          # opt_bin_resp/
if str(_EXEC_DIR) not in sys.path:
    sys.path.append(str(_EXEC_DIR))
from src.IO import SingleRunLoader, run_files            # noqa: E402

DATA_ROOT = _EXEC_DIR / "data"


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_runs(goal: str, n_genes=None, date=None, complete: bool = True,
              **filters) -> pd.DataFrame:
    """Load aggregated runs from ``data/<goal>/runs.db`` into a DataFrame.

    goal     : sub-folder of data/ that holds runs.db (e.g. "fig1").
    n_genes  : keep only this gene count(s) — int or list. ([goal + n_genes])
    date     : keep sweep folders whose name contains this string, e.g.
               "20260612" or a full "ng20_20260612_165139". ([... + date])
    complete : keep only finished runs (status == 'complete').
    filters  : extra equality filters on ANY column; scalar or list, e.g.
               entropy="renyi", receptor_type="heteromer", git_hash="a1885319".

    All receptor counts and environmental conditions matching the filter are
    returned together (one row per run).  A convenience column ``R`` is added:
    n_receptors for heteromers, n_genes for homomers.  Metric columns keep
    their stored ``…_mean`` name.  ``df.attrs['goal']`` records the goal so
    load_epochs / load_run can resolve paths without repeating it.
    """
    db = DATA_ROOT / goal / "runs.db"
    if not db.exists():
        raise FileNotFoundError(f"No runs.db for goal {goal!r}: {db}")
    with sqlite3.connect(db) as conn:
        df = pd.read_sql_query("SELECT * FROM runs", conn)

    if complete and "status" in df:
        df = df[df["status"] == "complete"]
    if n_genes is not None:
        df = df[df["n_genes"].isin(np.atleast_1d(n_genes))]
    if date is not None:
        df = df[df["sweep_folder"].str.contains(str(date), na=False)]
    for col, val in filters.items():
        df = df[df[col].isin(np.atleast_1d(val))]

    df = df.copy()
    df["R"] = np.where(df["receptor_type"] == "heteromer",
                       df["n_receptors"], df["n_genes"])
    df = df.reset_index(drop=True)
    df.attrs["goal"] = goal
    return df


def load_run(goal: str = None, run_dir: str = None, **filters):
    """Return ``(config, history_df)`` for a single run.

    Either pass an explicit ``run_dir``, or a ``goal`` plus equality filters
    that identify a run (first match wins, e.g. n_genes=5, n_receptors=10).
    """
    if run_dir is None:
        if goal is None:
            raise ValueError("pass run_dir, or goal + filters")
        df = load_runs(goal, complete=False, **filters)
        if df.empty:
            raise FileNotFoundError(f"No run matches {filters} in goal {goal!r}")
        run_dir = str(DATA_ROOT / goal / df.iloc[0]["path"])
    loader = SingleRunLoader(run_dir)
    return loader.load_config(), loader.load_history()


def latest_sweep(df: pd.DataFrame, group: str = "n_genes") -> pd.DataFrame:
    """Keep only runs from the latest sweep_folder per ``group`` value.

    ``sweep_folder`` embeds the date (``ngXX_YYYYMMDD_HHMMSS``), so the
    lexicographic max gives the newest sweep.  Use this instead of comparing
    ``sweep_date``, which is the per-run timestamp and would select a single run.
    """
    best = df.groupby(group)["sweep_folder"].transform("max")
    return df[df["sweep_folder"] == best]


def load_epochs(df: pd.DataFrame, goal: str = None) -> pd.DataFrame:
    """Concatenate every run's per-epoch stats.csv into one long DataFrame.

    ``df`` must come from :func:`load_runs` (needs the ``path`` column).  Each
    run's config columns (incl. ``R`` and env params) are carried onto its
    epoch rows, so the result feeds straight into :func:`plot_metric` with
    ``x="epoch"``.  ``goal`` defaults to ``df.attrs['goal']``.
    """
    goal = goal or df.attrs.get("goal")
    if goal is None:
        raise ValueError("goal unknown; pass goal= or use a df from load_runs")
    root = str(DATA_ROOT / goal)
    parts = []
    for _, row in df.iterrows():
        stats = run_files(row["path"], root)["stats"]
        if not Path(stats).exists():
            continue
        h = pd.read_csv(stats)
        for c in df.columns:
            if c not in h.columns:
                h[c] = row[c]
        parts.append(h)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.attrs["goal"] = goal
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_metric(df: pd.DataFrame, y: str, x: str = "R", group: str = None,
                err: str = "std", ax=None, cmap: str = None, **kw):
    """Mean line + error band of ``y`` vs ``x``, aggregating everything else.

    df    : runs DataFrame (from load_runs) or epoch DataFrame (load_epochs).
    y     : metric column to plot (change this to switch metric).
    x     : x-axis column (default "R"; use "epoch" for convergence).
    group : column → one curve per value (e.g. "n_genes", "sweep_folder",
            "git_hash", "entropy"). None = a single aggregated curve.
    err   : "std" | "sem" | None — the shaded band, i.e. the spread over all
            *other* varying parameters (the environmental conditions / seeds).
    ax    : axis to draw on (created if None).
    cmap  : if set and grouping, colour curves along this colormap by sorted
            group value and add a colorbar (else a legend is drawn).
    kw    : forwarded to ax.plot (lw, ls, color, label, ...).

    Returns the axis, so calls compose: pass ``ax=`` to overlay arms.
    """
    if ax is None:
        _, ax = plt.subplots()
    user_label = kw.pop("label", None)

    keys = [None] if group is None else sorted(df[group].dropna().unique())
    norm = cm = None
    if cmap is not None and group is not None and keys:
        norm = plt.Normalize(min(keys), max(keys))
        cm = plt.colormaps[cmap]

    for key in keys:
        sub = df if key is None else df[df[group] == key]
        stat = sub.groupby(x)[y].agg(["mean", "std", "sem"]).sort_index()
        if stat.empty:
            continue
        plot_kw = dict(kw)
        if cm is not None:
            plot_kw["color"] = cm(norm(key))
        label = user_label if key is None else str(key)
        line, = ax.plot(stat.index.values, stat["mean"].values,
                        label=label, **plot_kw)
        band = stat[err] if err in ("std", "sem") else None
        if band is not None:
            ax.fill_between(stat.index.values,
                            (stat["mean"] - band).values,
                            (stat["mean"] + band).values,
                            color=line.get_color(), alpha=0.2)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if cm is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        ax.figure.colorbar(sm, ax=ax, label=group)
    elif group is not None:
        ax.legend(title=group, fontsize=8)
    return ax
