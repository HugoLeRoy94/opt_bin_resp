"""Load task manifests and summarize independent optimizations, not test repeats."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tasks.cells.gene_expression._experiments import run_key


POINT = ["coverage", "n_genes", "n_cells", "genes_per_cell", "profile"]


def load_study(data_root, experiment, sweep_dirs=None):
    """Latest sweep per condition/coverage by default; explicit folders can pool seeds.

    Incompatible physics/training grids are rejected. Re-running an identical
    seeded sweep does not create another independent replicate.
    """
    manifests = []
    files = ([Path(p) / "experiment.json" for p in sweep_dirs] if sweep_dirs is not None
             else sorted(Path(data_root).glob(f"{experiment}_*/experiment.json")))
    for path in files:
        metadata = json.loads(path.read_text())
        if metadata["experiment"] != experiment:
            raise ValueError(f"Wrong experiment in {path}")
        manifests.append((path, metadata))
    if sweep_dirs is None:
        latest = {}
        for path, metadata in manifests:
            latest[(metadata["condition"], metadata["coverage"])] = (path, metadata)
        manifests = list(latest.values())
    if not manifests:
        print(f"No {experiment} experiments found in {data_root}. Run scripts/{experiment}.py first.")
        return pd.DataFrame()
    if len({m['protocol_id'] for _, m in manifests}) != 1:
        raise ValueError("Selected sweeps have different parameter grids or training/evaluation budgets. "
                         "Set SWEEPS to compatible experiment folders before comparing means.")
    records = []
    for manifest_path, metadata in manifests:
        planned = {run_key(r): r for r in metadata["rows"]}
        completed = set()
        for path in sorted(manifest_path.parent.rglob("test_results.json")):
            cfg_path = path.parent / "config.json"
            if not cfg_path.exists() or not (path.parent / "best_model.pt").exists():
                continue
            cfg = json.loads(cfg_path.read_text())
            key = run_key(cfg)
            if key not in planned:
                raise ValueError(f"Unplanned run in experiment folder: {path.parent}")
            point = planned[key]
            if cfg["cell_gene_sets"] != [list(g) for g in point["cell_gene_sets"]]:
                raise ValueError(f"Gene sets differ from the recorded design: {path.parent}")
            test = json.loads(path.read_text())
            mean = lambda name: float(np.mean(test[name]))
            genes = np.asarray(cfg["cell_gene_sets"], dtype=int)
            records.append(dict(
                condition=metadata["condition"], coverage=metadata["coverage"],
                replicate=point["replicate"], cell_sampling_seed=cfg["cell_sampling_seed"],
                world_seed=metadata["world_seed"], profile=point["profile"],
                n_genes=cfg["n_genes"], n_cells=cfg["n_cells"], genes_per_cell=genes.shape[1],
                fraction_expressed=genes.shape[1] / cfg["n_genes"],
                mi=mean("mutual_information_grouped"),
                response_noise=mean("conditional_entropy_response_grouped"),
                count_entropy=mean("grouped_count_entropy"),
                full_entropy=mean("response_entropy_grouped"),
                hard_entropy=mean("codeword_entropy_plugin") if "codeword_entropy_plugin" in test else np.nan,
                count_states=mean("grouped_n_states"),
                count_ceiling=mean("grouped_count_entropy_upper"),
                input_sample_ceiling=np.log2(mean("response_evaluation_samples")),
                genes_represented=np.unique(genes).size, receptor_pool=len(cfg["receptor_indices"]),
                run_dir=str(path.parent), sweep=str(manifest_path.parent),
            ))
            completed.add(key)
        print(f"{manifest_path.parent.name}: {len(completed)}/{len(planned)} completed runs")
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
        states_max=("count_states", "max"), pool_mean=("receptor_pool", "mean"),
        input_sample_ceiling=("input_sample_ceiling", "min"))
    summary["mi_sem"] = summary["mi_sd"] / np.sqrt(summary["n"])
    baseline_keys = ["condition", "coverage", "n_genes", "n_cells", "profile"]
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
    if frame.empty:
        return
    for labels, part in frame.groupby(list(group), sort=False):
        labels = labels if isinstance(labels, tuple) else (labels,)
        label = ", ".join(f"G={v}" if k == "n_genes" else str(v) for k, v in zip(group, labels))
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
