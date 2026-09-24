"""Load task manifests and summarize independent optimizations, not test repeats."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.IO import SweepLoader, find_latest_sweep
from tasks.cells.gene_expression._experiments import run_key


# One design point. `mi_estimator` and `training_entropy` are part of the identity:
# pooling sweeps that used different methods is allowed, but their means are never
# merged into one number.
METHOD = ["mi_estimator", "training_entropy"]
POINT = ["coverage", "n_genes", "n_cells", "genes_per_cell", "profile", *METHOD]


def _manifests(data_root, experiment, sweep_dirs):
    """Sweep roots carrying an experiment.json for this task, newest first."""
    if sweep_dirs is None:
        try:
            roots = find_latest_sweep(str(data_root), prefix=f"{experiment}_")
        except FileNotFoundError:
            roots = []
    else:
        roots = [str(path) for path in sweep_dirs]
    found = []
    for root in roots:
        path = Path(root) / "experiment.json"
        if not path.exists():
            continue
        metadata = json.loads(path.read_text())
        if metadata["experiment"] != experiment:
            raise ValueError(f"Wrong experiment in {path}")
        found.append((Path(root), metadata))
    return found


def _gene_sets(value):
    """Normalise gene sets to tuples so a JSON list compares equal to a config."""
    return tuple(tuple(int(g) for g in genes) for genes in value)


# Protocol settings the loaded frame represents as its own column, so runs that
# differ in them land on distinct rows instead of being averaged together.
_SEPARATED_SETTINGS = {
    "entropy":               "training_entropy",
    "measurement_fns":       "mi_estimator",
    "final_measurement_fns": "mi_estimator",
    "n_genes":               "n_genes",
    "n_cells":               "n_cells",
    "cell_max_genes":        "genes_per_cell",
    "cell_sampling_seed":    "cell_sampling_seed",
}


def _warn_if_incompatible(manifests):
    """Report protocol differences without blocking the comparison.

    Differences listed in _SEPARATED_SETTINGS survive into the frame, so summarize()
    keeps them as distinct design points. Anything else (epochs, learning rate,
    batch budgets, the environment behind a profile name) has no column, so those
    runs ARE averaged together. Pooling them is the caller's call; this only makes
    sure it is a visible one.
    """
    if len({m['protocol_id'] for _, m in manifests}) == 1:
        return
    settings = [m['protocol']['settings'] for _, m in manifests]
    keys = sorted(set().union(*(s.keys() for s in settings)))
    varying = {k: [s.get(k) for s in settings] for k in keys
               if len({json.dumps(s.get(k), sort_keys=True) for s in settings}) > 1}
    print("\nWARNING: the selected sweeps do not share one protocol.")
    for root, metadata in manifests:
        print(f"  {root.name}: protocol {metadata['protocol_id']}")
    pooled = []
    for key, values in varying.items():
        column = _SEPARATED_SETTINGS.get(key)
        shown = ", ".join(str(v) for v in values)
        if column is None:
            pooled.append(key)
            print(f"    {key}: {shown}   <-- POOLED")
        else:
            print(f"    {key}: {shown}   (separate rows, via `{column}`)")
    if len({json.dumps(m['protocol']['points'], sort_keys=True) for _, m in manifests}) > 1:
        print("    the parameter grids differ: points covered by only one sweep have a smaller n.")
    if pooled:
        print(f"  Runs differing only in {', '.join(pooled)} are averaged into the same mean.")
    else:
        print("  Every difference is represented by a column; nothing is silently averaged.")
    print("  estimator_comparison.py compares methods panel by panel instead.\n")


def load_study(data_root, experiment, sweep_dirs=None):
    """Latest sweep per condition/coverage by default; explicit folders can pool seeds.

    Incompatible physics/training grids are rejected. Re-running an identical
    seeded sweep does not create another independent replicate.

    Run discovery goes through src.IO.SweepLoader, so a partial or unreadable run
    is skipped exactly as everywhere else, and configs arrive as SingleRunConfig
    with the same backward-compatibility fixes the rest of the codebase gets.
    """
    manifests = _manifests(data_root, experiment, sweep_dirs)
    if sweep_dirs is None:
        latest = {}
        for sweep_root, metadata in manifests:   # newest first
            latest.setdefault((metadata["condition"], metadata["coverage"]),
                              (sweep_root, metadata))
        manifests = list(latest.values())
    if not manifests:
        print(f"No {experiment} experiments found in {data_root}. Run scripts/{experiment}.py first.")
        return pd.DataFrame()
    _warn_if_incompatible(manifests)
    records = []
    for sweep_root, metadata in manifests:
        planned = {run_key(r): r for r in metadata["rows"]}
        completed = set()
        for single_cfg, run_dir in SweepLoader(str(sweep_root)).iter_run_dirs():
            run_dir = Path(run_dir)
            if not (run_dir / "test_results.json").is_file() or not (run_dir / "best_model.pt").is_file():
                continue
            cfg = asdict(single_cfg)
            key = run_key(cfg)
            if key not in planned:
                raise ValueError(f"Unplanned run in experiment folder: {run_dir}")
            point = planned[key]
            if _gene_sets(cfg["cell_gene_sets"]) != _gene_sets(point["cell_gene_sets"]):
                raise ValueError(f"Gene sets differ from the recorded design: {run_dir}")
            test = json.loads((run_dir / "test_results.json").read_text())
            counting = metadata.get("arguments", {}).get("evaluation", "exact") == "counting"
            mi_key = "mutual_information_grouped_counting_plugin" if counting else "mutual_information_grouped"
            noise_key = "conditional_entropy_response_grouped_counting" if counting else "conditional_entropy_response_grouped"
            count_key = "grouped_count_entropy_counting_plugin" if counting else "grouped_count_entropy"
            response_key = "response_entropy_grouped_counting_plugin" if counting else "response_entropy_grouped"
            mean = lambda name: float(np.mean(test[name]))
            genes = np.asarray(cfg["cell_gene_sets"], dtype=int)
            records.append(dict(
                condition=metadata["condition"], coverage=metadata["coverage"],
                replicate=point["replicate"], cell_sampling_seed=cfg["cell_sampling_seed"],
                world_seed=metadata["world_seed"], profile=point["profile"],
                n_genes=cfg["n_genes"], n_cells=cfg["n_cells"], genes_per_cell=genes.shape[1],
                fraction_expressed=genes.shape[1] / cfg["n_genes"],
                mi_estimator="grouped counting (plug-in)" if counting else "exact grouped",
                training_entropy=cfg["entropy"],
                mi=mean(mi_key), response_noise=mean(noise_key),
                count_entropy=mean(count_key), full_entropy=mean(response_key),
                counting_unique_fraction=mean("grouped_counting_unique_fraction") if counting else np.nan,
                hard_entropy=mean("codeword_entropy_plugin") if "codeword_entropy_plugin" in test else np.nan,
                count_states=mean("grouped_n_states"),
                count_ceiling=mean("grouped_count_entropy_upper"),
                input_sample_ceiling=np.log2(mean("response_evaluation_samples")),
                genes_represented=np.unique(genes).size, receptor_pool=len(cfg["receptor_indices"]),
                run_dir=str(run_dir), sweep=str(sweep_root),
            ))
            completed.add(key)
        print(f"{sweep_root.name}: {len(completed)}/{len(planned)} completed runs")
        if len(completed) < len(planned):
            print("  Incomplete sweep: per-point n below counts only completed independent optimizations.")
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records).sort_values(["sweep", "run_dir"])
    identity = ["condition", *POINT, "world_seed", "cell_sampling_seed"]
    duplicated = frame.duplicated(identity, keep="last")
    if duplicated.any():
        print(f"Excluded {duplicated.sum()} repeated seeded runs; they are not additional replicates.")
    return frame.loc[~duplicated].reset_index(drop=True)


def summarize(runs):
    if runs.empty:
        return pd.DataFrame()
    summary = runs.groupby(["condition", *POINT], as_index=False).agg(
        n=("mi", "size"), mi_mean=("mi", "mean"), mi_sd=("mi", "std"),
        noise_mean=("response_noise", "mean"), count_entropy_mean=("count_entropy", "mean"),
        full_entropy_mean=("full_entropy", "mean"), hard_entropy_mean=("hard_entropy", "mean"),
        represented_mean=("genes_represented", "mean"), represented_min=("genes_represented", "min"),
        counting_unique_fraction_mean=("counting_unique_fraction", "mean"),
        states_max=("count_states", "max"), pool_mean=("receptor_pool", "mean"),
        input_sample_ceiling=("input_sample_ceiling", "min"))
    summary["mi_sem"] = summary["mi_sd"] / np.sqrt(summary["n"])
    baseline_keys = ["condition", "coverage", "n_genes", "n_cells", "profile", *METHOD]
    baseline = summary[summary.genes_per_cell == 1][baseline_keys + ["mi_mean", "n"]].rename(
        columns={"mi_mean": "baseline_mi", "n": "baseline_n"})
    summary = summary.merge(baseline, on=baseline_keys, how="left", validate="many_to_one")
    # Ratio of environmental means, not the mean of noisy per-run ratios.
    summary["retained"] = summary.mi_mean / summary.baseline_mi.where(summary.baseline_mi > 0)
    summary["fraction_expressed"] = summary.genes_per_cell / summary.n_genes
    return summary


def effects(summary):
    """Difference of environmental means; unpaired SEM, no matched-world claim."""
    if summary.empty:
        return pd.DataFrame()
    columns = POINT + ["mi_mean", "mi_sem", "n"]
    joined = summary[summary.condition == "heteromers"][columns].merge(
        summary[summary.condition == "homomers"][columns], on=POINT, suffixes=("_het", "_hom"))
    joined["advantage_bits"] = joined.mi_mean_het - joined.mi_mean_hom
    joined["advantage_sem"] = np.sqrt(joined.mi_sem_het**2 + joined.mi_sem_hom**2)
    return joined


def plot_curves(ax, frame, x, y, group=("condition", "coverage"), error=None):
    """One line per group, plus one line per method that varies within the frame.

    Without the method split, a frame pooling two estimators would put two y values
    at the same x on a single line and draw a zigzag that reads as noise.
    """
    if frame.empty:
        return
    group = list(group) + [column for column in METHOD
                           if column in frame and column not in group
                           and frame[column].nunique(dropna=False) > 1]
    for labels, part in frame.groupby(group, sort=False):
        labels = labels if isinstance(labels, tuple) else (labels,)
        label = ", ".join(f"G={v}" if k == "n_genes" else str(v) for k, v in zip(group, labels))
        if ('mi_estimator' not in group and 'mi_estimator' in part
                and (part.mi_estimator == 'grouped counting (plug-in)').all()):
            label += ", counting (plug-in)"
        part = part.sort_values(x)
        ax.plot(part[x], part[y], "o-", label=label)
        if error is not None:
            valid = part[error].notna()
            ax.errorbar(part.loc[valid, x], part.loc[valid, y], yerr=part.loc[valid, error],
                        fmt="none", color=ax.lines[-1].get_color(), capsize=3)
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=8)


def report(summary):
    if summary.empty:
        return
    print(summary.to_string(index=False))
    print("MI mean ± SEM across independent optimizations; n does not count repeated test batches.")
    if (summary.mi_estimator == "grouped counting (plug-in)").any():
        print("Sampled grouped counting: plug-in MI can be biased downward or negative; "
              "check sample-budget convergence separately from world-to-world SEM.")
    print("Retention is the ratio of means to each strategy's own one-gene baseline; no error bars propagated.")
    if summary.baseline_mi.isna().any():
        print("Missing one-gene baselines: corresponding retention values remain NaN.")
    if (summary.baseline_mi <= 0).any():
        print("Nonpositive one-gene baselines: retention is undefined and left as NaN.")
    near = summary.mi_mean >= summary.input_sample_ceiling - 1
    if near.any():
        print("Some MI values are within one bit of log2(evaluation samples). Run evaluation_budget.py.")
    effect = effects(summary)
    if not effect.empty:
        print("\nHeteromer advantage (difference of means; unpaired SEM):\n" + effect.to_string(index=False))
