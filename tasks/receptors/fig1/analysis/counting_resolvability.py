# %%
"""Can sampled counting resolve H(Y) for the fig1 arrays, and at what cost?

Counting measures H(Y) from the frequencies of the sampled binary responses, so it needs
to SEE each code several times: about 46·2^H(Y) sniffs for a bias under 0.1 bit, with
H(Y) = MI + H(Y|X) the quantity being measured. Below that the plug-in entropy reports
the budget, not the array — at B sniffs it cannot exceed log2(B).

Three panels, one per question:
  1  GPU memory and host RAM over the course of the run (from the monitor trace).
  2  computation time against batch size B.
  3  H(Y) against 1/B in log-log, one curve per R, to read off whether H still moves
     with B and to see the fitted power law H(B) = H∞ − a·B^−b on the same axes.

Measured on the ng10 data (October 2026), the extrapolation of panel 3 does NOT hold:
H∞ drifts downwards as larger batches are added and never settles. R=30 is the control,
since its curve reached a missing mass of 0.018 at B = 4.3e9 and so is essentially
measured at H = 22.52 bits — fitted on the 4 smallest batches the same law predicted
25.09, on 6 of them 23.70, on all 9 of them 22.94, still 0.4 bit high while quoting a fit
error of 0.11. R=40 and R=50 drift the same way from above. The summary table prints that
drift next to each fit, because it is the part the fit error does not contain.

Reads <data>/fig1/ng*/test_counting.csv (scripts/test_final.py, scripts/test_scaling.py)
and <data>/fig1/gpu_mem_*.csv (the run monitor). Measurements whose run died before the
CSV was written are recovered from <data>/fig1/run_*.log, which prints every batch.
"""
import glob
import os
import re
import sys
from pathlib import Path

sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")  # exec dir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from src.plotlib import DATA_ROOT

GOAL = "fig1"
GENES = None            # e.g. [10] to restrict; None = every n_genes present
SESSION = None          # gpu_mem_*.csv stem to plot; None = the longest trace present
MIN_POINTS = 4          # fewer batch sizes than this cannot constrain a 3-parameter fit
CONVERGED = 0.02        # missing mass below which a measured value needs no extrapolation
ROOT = DATA_ROOT / GOAL

# %%
# ── data ────────────────────────────────────────────────────────────────────
POINT = re.compile(
    r"test=\s*(?P<test_size>\d+)\s+H_MM=\s*(?P<response_entropy_mm>[-\d.]+)\s+"
    r"MI_MM=\s*(?P<mutual_information_counting_mm>[-\d.]+)\s+ceiling=\s*\S+\s+"
    r"missing_mass=(?P<response_counting_missing_mass>[\d.]+)"
    r"(?:\s+codes=(?P<response_counting_K_hat>\S+))?"
    r"(?:\s+(?P<seconds>[\d.]+)s\s+ram=(?P<peak_ram_gb>[\d.]+)GB)?")
HEAD = re.compile(r"^(?P<sweep_folder>\S+) \| G(?P<n_genes>\d+) R(?P<n_receptors>\d+) "
                  r"train=(?P<train_batch>\d+)")


def from_csv(root):
    files = sorted(glob.glob(str(root / "ng*" / "test_counting.csv")))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def from_logs(root):
    """Measurements printed by a run whose CSV was never written (crash, kill, OOM).

    The printed line carries the same information as the CSV row, minus the run
    directory, so recovered rows are keyed by (sweep, n_genes, R) — unambiguous for
    the --per_condition surveys these curves come from. H(Y|X) is recovered as
    H_MM - MI_MM, which is exact.
    """
    rows, head = [], None
    for path in sorted(glob.glob(str(root / "run_*.log"))):
        for line in open(path, errors="ignore"):
            match = HEAD.match(line)
            if match:
                head = {k: (v if k == "sweep_folder" else int(v))
                        for k, v in match.groupdict().items()}
                continue
            match = POINT.search(line)
            if match and head:
                row = {k: v for k, v in match.groupdict().items() if v is not None}
                rows.append(dict(head, source=Path(path).stem,
                                 **{k: float(v) for k, v in row.items()}))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["test_size"] = out["test_size"].astype(int)
    out["conditional_entropy_response"] = (out["response_entropy_mm"]
                                           - out["mutual_information_counting_mm"])
    return out


KEY = ["sweep_folder", "n_genes", "n_receptors", "test_size"]
csv, logs = from_csv(ROOT), from_logs(ROOT)
df = pd.concat([d for d in (csv, logs) if not d.empty], ignore_index=True)
if df.empty:
    raise SystemExit(f"no counting data under {ROOT}; run scripts/test_final.py and sync")
df = df.drop_duplicates(KEY, keep="first").sort_values(KEY)   # CSV wins over the log
recovered = int(df["source"].notna().sum()) if "source" in df else 0
if GENES:
    df = df[df.n_genes.isin(GENES)]
print(f"{len(df)} measurements over "
      f"{df.groupby(['n_genes','n_receptors']).ngroups} conditions"
      f"{f', {recovered} recovered from run logs' if recovered else ''}")


# %%
# ── power-law fit:  H(B) = H_inf - a·B^-b  ──────────────────────────────────
def law(b_size, h_inf, a, b):
    return h_inf - a * b_size ** -b


def fit(sizes, h):
    """(H_inf, standard error, exponent b) or (nan, nan, nan) if it will not fit.

    H_inf is bounded below by the largest measured H: the plug-in entropy only grows
    with B, so no asymptote below a measurement is admissible.
    """
    if len(sizes) < MIN_POINTS:
        return np.nan, np.nan, np.nan
    try:
        p, cov = curve_fit(law, np.asarray(sizes, float), np.asarray(h, float),
                           p0=[max(h) + 1, 10.0, 0.2],
                           bounds=([max(h), 1e-9, 1e-3], [max(h) + 40, 1e12, 3.0]),
                           maxfev=100000)
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan
    return p[0], float(np.sqrt(np.diag(cov))[0]), p[2]


conditions = sorted(df.groupby(["n_genes", "n_receptors"]).groups)
summary = []
for (g, r) in conditions:
    sub = df[(df.n_genes == g) & (df.n_receptors == r)].sort_values("test_size")
    sizes, h = sub.test_size.values, sub.response_entropy_mm.values
    h_inf, err, b = fit(sizes, h)
    # Refit without the largest batch: how much the answer moves when the last
    # measurement is added is the only honest test of the extrapolation, and it is
    # exactly what the quoted fit error leaves out.
    drift = h_inf - fit(sizes[:-1], h[:-1])[0]
    last = sub.iloc[-1]
    # The oldest rows predate the missing-mass column, so take the coverage from the
    # largest batch that actually recorded one rather than reporting nothing.
    covered = sub.dropna(subset=["response_counting_missing_mass"])
    summary.append(dict(
        n_genes=g, n_receptors=r, points=len(sub), B_max=int(sizes[-1]),
        H=last.response_entropy_mm, MI=last.mutual_information_counting_mm,
        missing=(covered.iloc[-1].response_counting_missing_mass
                 if len(covered) else np.nan),
        missing_at=int(covered.iloc[-1].test_size) if len(covered) else 0,
        H_inf=h_inf, H_inf_err=err, exponent=b, drift=drift,
        MI_inf=h_inf - last.conditional_entropy_response))
summary = pd.DataFrame(summary)
print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# %%
# ── figure ──────────────────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
colors = plt.colormaps["viridis"](np.linspace(0, .9, max(len(conditions), 2)))
label = {c: f"G{c[0]} R{c[1]}" for c in conditions}

# 1. memory over time, from the monitor trace
ax = axs[0]
traces = sorted(glob.glob(str(ROOT / "gpu_mem_*.csv")), key=os.path.getsize)
trace = next((t for t in traces if SESSION and SESSION in t), traces[-1] if traces else None)
if trace:
    m = pd.read_csv(trace, skipinitialspace=True)
    t = pd.to_datetime(m.timestamp, format="%Y/%m/%d %H:%M:%S.%f")
    hours = (t - t.iloc[0]).dt.total_seconds() / 3600
    ax.plot(hours, m.mem_used_MiB / 1024, color="tab:red", lw=1.2, label="GPU used")
    ax.plot(hours, m.mem_total_MiB / 1024, color="tab:red", ls=":", lw=.8, label="GPU total")
    if "ram_used_MiB" in m:      # only traces taken after the monitor started logging RAM
        ax.plot(hours, m.ram_used_MiB / 1024, color="tab:blue", lw=1.2, label="host RAM used")
        ax.plot(hours, m.ram_total_MiB / 1024, color="tab:blue", ls=":", lw=.8,
                label="host RAM total")
    else:
        ax.text(.5, .08, "this trace has no RAM columns\n(monitor_gpu.sh logs them "
                "from October 2026 on)", transform=ax.transAxes, ha="center",
                fontsize=7, color="0.4")
    ax.set_title(f"memory during the run\n{Path(trace).stem}", fontsize=8)
    ax.set_xlabel("time since the run started [hours]"), ax.set_ylabel("memory [GB]")
    ax.legend(fontsize=6)
else:
    ax.set_axis_off()

# 2. computation time against batch size
ax = axs[1]
for color, c in zip(colors, conditions):
    sub = df[(df.n_genes == c[0]) & (df.n_receptors == c[1])].sort_values("test_size")
    if "seconds" not in sub or sub.seconds.isna().all():
        continue
    ax.plot(sub.test_size, sub.seconds, "-o", ms=3, color=color, label=label[c])
if ax.lines:
    span = np.array([df.test_size.min(), df.test_size.max()], float)
    ax.plot(span, df.seconds.min() * span / span[0], "k:", lw=.8, label="linear in B")
    ax.legend(fontsize=6)
else:
    ax.text(.5, .5, "no `seconds` column in these rows yet\nrerun test_final.py to "
            "record timing", transform=ax.transAxes, ha="center", fontsize=8, color="0.4")
ax.set_xscale("log", base=2), ax.set_yscale("log")
ax.set_xlabel("batch size B [sniffs]")
ax.set_ylabel("time to grow the batch to B [s]")
ax.set_title("computation time vs batch size", fontsize=8)

# 3. H(Y) against 1/B, log-log, one curve per R
ax = axs[2]
for color, c in zip(colors, conditions):
    sub = df[(df.n_genes == c[0]) & (df.n_receptors == c[1])].sort_values("test_size")
    sizes = sub.test_size.values.astype(float)
    h = sub.response_entropy_mm.values
    ax.plot(1 / sizes, h, "o", ms=4, color=color, label=label[c])
    row = summary[(summary.n_genes == c[0]) & (summary.n_receptors == c[1])].iloc[0]
    if np.isfinite(row.H_inf):      # the fitted law over the measured range
        grid = np.geomspace(1 / sizes.max(), 1 / sizes.min(), 200)
        ax.plot(grid, law(1 / grid, row.H_inf, *curve_fit(
            law, sizes, h, p0=[row.H_inf, 10, row.exponent],
            bounds=([max(h), 1e-9, 1e-3], [max(h) + 40, 1e12, 3.0]),
            maxfev=100000)[0][1:]), "-", lw=1, color=color, alpha=.7)
ax.set_xscale("log"), ax.set_yscale("log")
ax.set_xlabel("1 / B"), ax.set_ylabel("H(Y) Miller-Madow [bits]")
ax.set_title("entropy vs inverse batch size (line = fitted power law)", fontsize=8)
ax.legend(fontsize=6)

fig.suptitle("Counting resolvability for fig1: cost of a batch, and whether H(Y) has "
             "stopped depending on it", y=1.0, fontsize=11)
fig.tight_layout()
plt.show()

# %%
# ── verdict per condition ───────────────────────────────────────────────────
for _, s in summary.iterrows():
    # The missing mass only falls as B grows, so a small one recorded at a smaller
    # batch (older rows predate the column) still certifies the largest batch.
    if s.missing < CONVERGED:
        where = "" if s.missing_at == s.B_max else f" (measured at B={int(s.missing_at):,})"
        verdict = (f"MEASURED: missing mass {s.missing:.3f}{where}, "
                   f"H = {s.H:.2f}, MI = {s.MI:.2f}")
    elif s.points < MIN_POINTS:
        verdict = (f"only {int(s.points)} batch size(s): rerun the sweep "
                   "(scripts/test_final.py) before judging this condition")
    else:
        drift = ("" if not np.isfinite(s.drift) else
                 f", still moving {s.drift:+.2f} bit when the last batch was added")
        verdict = (f"NOT CONVERGED: missing mass {s.missing:.3f}. Quote H >= {s.H:.2f} "
                   f"(MI >= {s.MI:.2f}); the fit says {s.H_inf:.2f}{drift}, so it is an "
                   "over-estimate, not a value")
    print(f"G{int(s.n_genes):>3} R{int(s.n_receptors):>3}  B={int(s.B_max):>13,}  {verdict}")
# %%
