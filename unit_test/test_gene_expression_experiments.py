"""Experiment designs, replicate accounting, and small end-to-end execution."""
import contextlib
import importlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tasks.cells.gene_expression._experiments import expression_sets
from tasks.cells.gene_expression.analysis._shared import load_study, summarize, effects


@pytest.mark.parametrize("coverage", ["random", "complete"])
def test_expression_design_is_nested_and_does_not_touch_world_rng(coverage):
    before = torch.get_rng_state()
    single = torch.tensor(expression_sets(6, 18, 1, 123, coverage))
    two = torch.tensor(expression_sets(6, 18, 2, 123, coverage))
    full = torch.tensor(expression_sets(6, 18, 6, 123, coverage))
    assert (two == single).any(1).all()
    torch.testing.assert_close(full, torch.arange(6).expand(18, -1))
    torch.testing.assert_close(before, torch.get_rng_state())
    if coverage == "complete":
        torch.testing.assert_close(torch.bincount(single.flatten()), torch.full((6,), 3))
    with pytest.raises(ValueError, match="n_cells"):
        expression_sets(6, 3, 1, 123, "complete")


@pytest.mark.parametrize("script,extra,expected", [
    ("replicates", ["--n_genes", "3", "--n_cells", "10"], 6),
    ("scaling", ["--n_genes", "3", "4"], 14),
    ("environment", ["--n_genes", "3", "--profiles", "base", "dimension"], 12),
    ("environment", ["--n_genes", "3", "--profiles", "base", "dimension", "--baseline_only"], 4),
])
def test_zipped_configs_retain_world_and_architecture_design(monkeypatch, script, extra, expected):
    module = importlib.import_module(f"tasks.cells.gene_expression.scripts.{script}")
    monkeypatch.setattr(module, "launch", lambda config, args, rows, experiment: (config, args, rows))
    het, _, planned = module.main(extra + ["--replicates", "2", "--coverage", "complete"])
    hom, _, _ = module.main(extra + ["--replicates", "2", "--coverage", "complete", "--condition", "homomers"])
    assert len(planned) == expected
    hsteps, msteps = next(het.generate_trajectories()), next(hom.generate_trajectories())
    assert len(hsteps) == len(msteps) == expected
    for h, m, row in zip(hsteps, msteps, planned):
        assert h.cell_gene_sets == m.cell_gene_sets
        assert h.n_cells > h.n_genes
        assert len(h.conc_mean) == len(h.conc_std) == h.n_ligands
        assert h.cell_max_genes == row["genes_per_cell"]
        assert h.entropy == m.entropy == "grouped_mi"
        assert h.cell_readout == m.cell_readout == "mean"
        assert set(np.asarray(h.cell_gene_sets).flatten()) == set(range(h.n_genes))
        assert all(len(set(r)) == 1 for r in m.receptor_indices)
    assert not het.warm_start and not hom.warm_start


def test_default_scaling_fits_declared_guard_and_dry_run_writes_nothing(tmp_path, capsys):
    from tasks.cells.gene_expression.scripts.scaling import main
    manifest = main(["--dry_run", "--base_folder", str(tmp_path)])
    assert len(manifest["rows"]) == 5 * (3 + 4 + 5 + 6)
    assert max(r["count_states"] for r in manifest["rows"]) <= manifest["arguments"]["max_states"]
    assert list(tmp_path.iterdir()) == []
    assert "Largest count alphabet" in capsys.readouterr().out


def test_repeated_sweeps_analysis_and_budget_evaluation(tmp_path):
    from tasks.cells.gene_expression.scripts.replicates import main
    from tasks.cells.gene_expression.scripts.evaluation_budget import main as budget_main
    # Real training and checkpoint loading, kept tiny for a CPU regression.
    args = ["--replicates", "2", "--epochs", "1", "--batch_size", "8", "--test_batch_size", "8",
            "--final_batch_size", "17", "--eval_chunk_size", "5", "--n_genes", "2", "--n_cells", "6",
            "--coverage", "complete", "--base_folder", str(tmp_path)]
    h_manifest = main(args)
    m_manifest = main(args + ["--condition", "homomers"])
    assert h_manifest["world_seed"] != m_manifest["world_seed"]
    runs = load_study(tmp_path, "replicates")
    assert len(runs) == 8  # not 80: the ten test repeats are not optimization repeats
    summary = summarize(runs)
    assert (summary.n == 2).all()
    valid_baseline = (summary.genes_per_cell == 1) & (summary.baseline_mi > 0)
    assert (summary[valid_baseline].retained == 1).all()
    assert summary[summary.baseline_mi <= 0].retained.isna().all()
    assert (summary.represented_min == 2).all()
    assert len(effects(summary)) == 2
    # Re-listing a seeded sweep must not double its sample size.
    folders = sorted(tmp_path.glob("replicates_*"))
    assert len(load_study(tmp_path, "replicates", folders + folders)) == 8
    known = runs.copy()
    known["mi"] = np.where(known.genes_per_cell == 1, 2., 1.) + 2 * known.replicate
    expected = summarize(known)
    assert (expected.mi_sem == 1).all()
    np.testing.assert_allclose(expected[expected.genes_per_cell == 2].retained, 2 / 3)

    # Missing baselines cannot produce a bogus retention curve.
    no_baseline = summarize(runs[runs.genes_per_cell > 1])
    assert no_baseline.retained.isna().all()

    run_dir = Path(runs.iloc[0].run_dir)
    original = (run_dir / "test_results.json").read_bytes()
    budget_main(["--run_dirs", str(run_dir), "--budgets", "7", "19", "--repeats", "2",
                 "--chunk_size", "5", "--device", "cpu"])
    assert (run_dir / "test_results.json").read_bytes() == original
    report = json.loads(next(run_dir.glob("grouped_evaluation_budget_*.json")).read_text())
    assert len(report["records"]) == 4
    assert [r["samples"] for r in report["records"]] == [7, 19, 7, 19]
    assert all(np.isfinite(r["mutual_information_grouped"]) for r in report["records"])

    # Actual incomplete sweeps report reduced n instead of inventing results.
    (run_dir / "test_results.json").rename(run_dir / "incomplete_results.json")
    assert len(load_study(tmp_path, "replicates")) == 7

    # Budget/protocol mismatches are reported, not silently accepted. Loading still
    # returns the runs: whether the comparison is worth making is the caller's call.
    manifest_path = next(tmp_path.glob("replicates_homomers_*/experiment.json"))
    metadata = json.loads(manifest_path.read_text())
    metadata["protocol_id"] = "different"
    manifest_path.write_text(json.dumps(metadata))
    warning = io.StringIO()
    with contextlib.redirect_stdout(warning):
        mixed = load_study(tmp_path, "replicates")
    assert "do not share one protocol" in warning.getvalue()
    assert len(mixed) == 7

    # Pooling two methods must not average them into one mean.
    both = pd.concat([runs, runs.assign(training_entropy="grouped_kt_mi")], ignore_index=True)
    pooled = summarize(both)
    assert set(pooled.training_entropy) == {"grouped_mi", "grouped_kt_mi"}
    assert (pooled.n == 2).all()
    assert len(pooled) == 2 * len(summarize(runs))
