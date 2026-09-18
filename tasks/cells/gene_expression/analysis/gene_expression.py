# %%
"""Compare final information and training curves across a gene-expression sweep.

Reads saved JSON/CSV directly; no GPU, torch, or rebuilt runs.db is required.
The newest sweep is selected even if partial; incomplete runs are reported/skipped.
Use --sweep PATH to select an older sweep explicitly.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import warnings

TASK_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(__file__).resolve().parents[4] / "data" / "gene_expression"
METRICS = (
    "identity_channel", "mutual_information_kt", "mutual_information_kt_upper",
    "mutual_information_counting_mm", "conditional_entropy_response",
    "codeword_entropy_plugin", "codeword_entropy_K_hat",
)


def metric_mean(values):
    if values is None:
        return math.nan
    if isinstance(values, list):
        return sum(values) / len(values) if values else math.nan
    return float(values)


def load_sweep(sweep):
    """Use persisted PMFs and resolved gene sets, never the current launch config."""
    records = []
    for path in sorted(sweep.rglob("config.json")):
        run = path.parent
        if not (run / "test_results.json").is_file() or not (run / "stats.csv").is_file():
            warnings.warn(f"Skipping incomplete run: {run}")
            continue
        cfg = json.loads(path.read_text())
        results = json.loads((run / "test_results.json").read_text())
        pmf = cfg["cell_size_pmf"]
        genes = cfg["cell_gene_sets"]
        target = sum((i + 1) * p for i, p in enumerate(pmf)) / sum(pmf)
        with (run / "stats.csv").open(newline="") as stream:
            history = list(csv.DictReader(stream))
        if not history:
            warnings.warn(f"Skipping empty history: {run}")
            continue
        summary = dict(
            mean_genes_target=target,
            mean_genes_realized=sum(map(len, genes)) / len(genes),
            unique_gene_sets=len({tuple(sorted(g)) for g in genes}),
            n_cells=len(genes), receptor_pool=len(cfg["receptor_indices"]),
            cell_sampling_seed=cfg["cell_sampling_seed"],
            cell_readout=cfg["cell_readout"],
            identity_reference=min(math.log2(cfg["n_ligands"]), len(genes)),
            run_dir=str(run),
        )
        summary.update({key: metric_mean(results.get(key)) for key in METRICS})
        records.append((summary, cfg, history))
    if not records:
        raise ValueError(f"No completed runs with test results and history in {sweep}")
    return sorted(records, key=lambda item: item[0]["mean_genes_target"])


def optimization_steps(history, cfg):
    """Recover optimization updates from the runner's legacy logging indices."""
    steps = [int(float(row["epoch"])) for row in history]
    if steps == list(range(len(steps))):
        interval = max(1, cfg["epochs"] // 100)
        steps = [step * interval for step in steps]
    return steps


def analyze(sweep, output_dir, show=True):
    import matplotlib.pyplot as plt

    records = load_sweep(sweep)
    summaries = [record[0] for record in records]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Sweep: {sweep}")
    print("mean (target/realized)  unique cells  pool     identity MI   KT MI lower/upper")
    for row in summaries:
        print(f"{row['mean_genes_target']:6.2f}/{row['mean_genes_realized']:<6.2f}"
              f"          {row['unique_gene_sets']:2}/{row['n_cells']:<2}"
              f"      {row['receptor_pool']:5}    {row['identity_channel']:8.3f}"
              f"      {row['mutual_information_kt']:.3f}/{row['mutual_information_kt_upper']:.3f}")

    x = [row["mean_genes_target"] for row in summaries]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for key, label, style in (
        ("identity_channel", "Identity MI", "o-"),
        ("mutual_information_kt", "KT MI lower", "s-"),
        ("mutual_information_kt_upper", "KT MI upper", "s--"),
        ("mutual_information_counting_mm", "Counting MI (MM)", "^:"),
    ):
        axes[0].plot(x, [row[key] for row in summaries], style, label=label)
    axes[0].plot(x, [row["identity_reference"] for row in summaries], "k:",
                 label="Singleton identity reference")
    axes[0].set_ylabel("Information [bits]")
    axes[0].legend(fontsize=8)
    for key, label in (
        ("conditional_entropy_response", "H(response | sniff)"),
        ("codeword_entropy_plugin", "H(hard code), diagnostic"),
    ):
        axes[1].plot(x, [row[key] for row in summaries], "o-", label=label)
    axes[1].set_ylabel("Entropy [bits]")
    axes[1].legend(fontsize=8)
    axes[2].plot(x, [row["receptor_pool"] for row in summaries], "o-")
    axes[2].set_ylabel("Distinct receptors in pool")
    for ax in axes:
        ax.set_xlabel("Target mean expressed genes / cell")
        ax.grid(alpha=.2)
    fig.suptitle("Mean cell readout — gene-expression sweep")
    fig.savefig(output_dir / "information_vs_genes.png", dpi=180)

    curves, curve_axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                     layout="constrained")
    for row, cfg, history in records:
        steps = optimization_steps(history, cfg)
        label = f"mean genes = {row['mean_genes_target']:g}"
        for ax, key in zip(curve_axes, ("mutual_information_kt", "conditional_entropy_response")):
            ax.plot(steps, [float(point.get(key, "nan")) for point in history], label=label)
    curve_axes[0].set_ylabel("KT MI lower [bits]")
    curve_axes[1].set_ylabel("H(response | sniff) [bits]")
    curve_axes[1].set_xlabel("Optimization update")
    for ax in curve_axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    curves.suptitle("Evaluation at final receptor temperature")
    curves.savefig(output_dir / "training_curves.png", dpi=180)
    print(f"Saved summary and figures to {output_dir}")
    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(curves)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, help="Sweep directory; defaults to newest")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening windows")
    args = parser.parse_args()
    sweep = args.sweep
    if sweep is None:
        candidates = sorted(p for p in args.data_root.glob("cell_gene_expression_*") if p.is_dir())
        if not candidates:
            parser.error(f"No sweeps in {args.data_root}; run and sync gene_expression first.")
        sweep = candidates[-1]
    if not sweep.is_dir():
        parser.error(f"Sweep directory does not exist: {sweep}")
    # Keep figures from different sweeps separate.
    output_dir = args.output_dir or TASK_ROOT / "figures" / sweep.name
    analyze(sweep, output_dir, show=not args.no_show)


if __name__ == "__main__":
    main()
