"""Post-hoc test-size scaling orchestration.

Reload each run in a set of sweeps (best_model.pt) and re-measure the KT bracket at
chosen test sizes — the cheap way to check the test-entropy-vs-samples curve plateaus,
no retraining. The per-run MEASUREMENT lives in analysis_helper (kt_bracket); THIS module
only orchestrates: walk sweeps, pick runs, choose sizes, write the CSV. Task scripts
(tasks/*/scripts/test_scaling.py) are thin wrappers over build_parser + sizes_from_args + run.

Size strategies (sizes_from_args, in priority order):
  --test_sizes A B ...  explicit absolute sizes
  --mult m1 m2 ...      per-run multiples of that run's train batch (B, 2B, 4B, ...)
  (neither)             auto geometric ladder of --n_test points (×4), up to the mem cap
Every size is clamped to [EVAL_TILE, eval_batch_cap(free)]; sizes may exceed 2^R (more
samples of the same state space just cut variance — that IS the plateau check).
"""
import argparse
import glob
import os

import pandas as pd
import torch

from src.IO import SingleRunLoader
from src.plotlib import load_model
from src.analysis_helper import kt_bracket, eval_batch_cap, EVAL_TILE


def build_parser(description, data_default, sweep_default, mult_default=None):
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=data_default, help="data root holding the sweeps")
    p.add_argument("--sweep_glob", default=sweep_default,
                   help="sweep-folder glob (e.g. 'ng2_*' targets one n_genes → parallelise per GPU)")
    p.add_argument("--test_sizes", type=int, nargs="+", default=None,
                   help="explicit absolute test batch sizes (highest priority)")
    p.add_argument("--mult", type=float, nargs="+", default=mult_default,
                   help="per-run multiples of the train batch B (e.g. 1 2 4 8 16 → B,2B,4B,8B,16B)")
    p.add_argument("--n_test", type=int, default=5, help="auto ladder: number of sizes (×4 apart)")
    p.add_argument("--n_receptors", type=int, nargs="+", default=None,
                   help="only measure runs with these R (default: all)")
    p.add_argument("--per_condition", action="store_true",
                   help="measure only the FIRST run per (n_genes, R) — skip repeat envs (faster)")
    return p


def sizes_from_args(args):
    """Return a sizes_for(cfg, mem_free) -> list[int] callable per the chosen strategy."""
    def sizes_for(cfg, mem_free):
        if args.test_sizes:
            return list(args.test_sizes)
        if args.mult:
            b = cfg.batch_size if isinstance(cfg.batch_size, int) else 0
            return [int(round(m * b)) for m in args.mult]
        top = min(1 << cfg.n_receptors, eval_batch_cap(mem_free))       # auto ladder
        return [int(top // (4 ** k)) for k in range(args.n_test)]
    return sizes_for


def select_runs(sweep_root, n_receptors=None, per_condition=False):
    """(run_dir, cfg) for every checkpoint in the sweep, filtered by R / per-condition."""
    out, seen = [], set()
    for p in sorted(glob.glob(os.path.join(sweep_root, "**", "best_model.pt"), recursive=True)):
        rd = os.path.dirname(p)
        cfg = SingleRunLoader(rd).load_config()
        if n_receptors and cfg.n_receptors not in n_receptors:
            continue
        key = (cfg.n_genes, cfg.n_receptors)
        if per_condition and key in seen:
            continue
        seen.add(key)
        out.append((rd, cfg))
    return out


def _clamp(sizes, mem_free):
    cap = eval_batch_cap(mem_free)
    keep = [s for s in sorted(set(sizes)) if EVAL_TILE <= s <= cap]
    return keep, [s for s in sizes if s not in keep], cap


def run(data_root, sweep_glob, sizes_for, *, n_receptors=None, per_condition=False, device=None):
    """Measure the KT bracket at sizes_for(cfg, mem_free) for every selected run in every
    matching sweep; write/merge <sweep_root>/test_scaling.csv (dedup on run_dir+test_size)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sweeps = sorted(glob.glob(os.path.join(data_root, sweep_glob)))
    if not sweeps:
        print(f"no sweeps match {sweep_glob!r} under {data_root}")
        return

    for sweep_root in sweeps:
        entries = select_runs(sweep_root, n_receptors, per_condition)
        if not entries:
            continue
        rows = []
        for run_dir, cfg in entries:
            env, physics, ri = load_model(run_dir=run_dir, device=device)
            train_batch = cfg.batch_size if isinstance(cfg.batch_size, int) else None
            free = torch.cuda.mem_get_info()[0] if device == "cuda" else 8 * (1 << 30)
            sizes, dropped, cap = _clamp(sizes_for(cfg, free), free)
            if dropped:
                print(f"    (dropped {dropped}: outside [{EVAL_TILE}, {cap}])")
            print(f"{os.path.basename(sweep_root)} | G{cfg.n_genes} R{cfg.n_receptors} "
                  f"train={train_batch} | sizes {sizes}")
            for ts in sizes:
                lo, up = kt_bracket(env, physics, ri, ts, tile=EVAL_TILE)
                rows.append(dict(sweep_folder=os.path.basename(sweep_root),
                                 run_dir=os.path.relpath(run_dir, sweep_root),
                                 n_genes=cfg.n_genes, n_receptors=cfg.n_receptors,
                                 train_batch=train_batch, test_size=ts,
                                 kt_lower=lo, kt_upper=up))
                print(f"    test={ts:>9d}  kt_lower={lo:6.3f}  kt_upper={up:6.3f}")
            del env, physics
            if device == "cuda":
                torch.cuda.empty_cache()

        out = os.path.join(sweep_root, "test_scaling.csv")
        new = pd.DataFrame(rows)
        if os.path.exists(out):
            new = pd.concat([pd.read_csv(out), new], ignore_index=True)
        new = (new.drop_duplicates(["run_dir", "test_size"], keep="last")
                  .sort_values(["run_dir", "test_size"]))
        new.to_csv(out, index=False)
        print(f"wrote {out}  ({len(new)} rows total)\n")
