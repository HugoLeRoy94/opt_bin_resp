#!/usr/bin/env python3
"""Post-hoc test-set scaling — re-measure KT entropy at growing TEST sizes.

Reads the optimized environments saved by sample_limit.py (best_model.pt) and,
WITHOUT any retraining, re-evaluates the KT lower + upper entropy on increasingly
large test batches. This is the cheap way to sweep the test size: reloading the
frozen env and only re-sampling costs a forward pass, not a full optimization.

For every run in the matched sweeps it writes <sweep_root>/test_scaling.csv with
one row per (run, test_size): kt_lower, kt_upper, plus n_genes/n_receptors and the
resolved train batch (so the plotter can draw the log2(train) ceiling). The test
ladder is n_test geometric points (×4 apart) ending at the memory/2^R maximum, so
only ONE point sits at the slow high end (KT is O(B²)).

Run on the cluster (GPU), same launcher as the training scripts:
  ../run_remote.sh optimizer test_scaling.py 0
  ../run_remote.sh optimizer test_scaling.py 0 -- --n_test 6 --sweep_glob 'sample_limit_*'
"""
import argparse
import glob
import os
import sys
sys.path.append('/app')

import numpy as np
import pandas as pd
import torch

from src.IO import SingleRunLoader
from src.plotlib import load_model
from src.bin_loss import compute_kt_entropy, compute_kt_upper_entropy

EVAL_TILE = 2048        # KT internal tile; also the memory-cap denominator
FWD_CHUNK = 16384       # forward sub-batch (no_grad) — bounds sampling memory


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="/app/data/optimizer", help="data root holding the sweeps")
    p.add_argument("--sweep_glob", default="sample_limit_*", help="sweep-folder glob")
    p.add_argument("--n_test", type=int, default=5, help="number of test sizes (×4 apart)")
    return p.parse_args()


def _test_ladder(n_receptors: int, n_test: int, mem_free: int) -> list:
    """n_test geometric sizes (×4) ending at min(2^R, memory-cap)."""
    eval_cap = max(EVAL_TILE, int(mem_free * 0.8) // (EVAL_TILE * 4 * 4))
    top = min(1 << n_receptors, eval_cap)
    sizes = sorted({int(top // (4 ** k)) for k in range(n_test)})
    return [s for s in sizes if s >= 1024]


@torch.no_grad()
def _measure_kt(env, physics, ri, test_size: int, device: str) -> tuple:
    """KT lower + upper (bits) on `test_size` fresh samples, forward in FWD_CHUNK
    sub-batches so the sampling forward pass stays bounded; KT tiles internally."""
    ri_fwd = ri if env.use_interface_model else None
    acts = []
    n = 0
    while n < test_size:
        b = min(FWD_CHUNK, test_size - n)
        E, concs, _ = env.sample_batch(b, receptor_indices=ri_fwd)
        acts.append(physics(E, concs, ri, pre_gathered=env.use_interface_model))
        n += b
    activity = torch.cat(acts, dim=0)                       # (test_size, R)
    soft = torch.stack([1.0 - activity, activity], dim=-1)  # (test_size, R, 2)
    lo = compute_kt_entropy(soft, chunk_size=EVAL_TILE).item()
    up = compute_kt_upper_entropy(soft, chunk_size=EVAL_TILE).item()
    return lo, up


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sweeps = sorted(glob.glob(os.path.join(args.data, args.sweep_glob)))
    if not sweeps:
        print(f"no sweeps match {args.sweep_glob!r} under {args.data}")
        return

    for sweep_root in sweeps:
        run_dirs = sorted(os.path.dirname(p) for p in
                          glob.glob(os.path.join(sweep_root, "**", "best_model.pt"),
                                    recursive=True))
        if not run_dirs:
            continue
        rows = []
        for run_dir in run_dirs:
            cfg = SingleRunLoader(run_dir).load_config()
            env, physics, ri = load_model(run_dir=run_dir, device=device)
            train_batch = cfg.batch_size if isinstance(cfg.batch_size, int) else None
            free = torch.cuda.mem_get_info()[0] if device == "cuda" else 8 * (1 << 30)
            ladder = _test_ladder(cfg.n_receptors, args.n_test, free)
            print(f"{os.path.basename(sweep_root)} | G{cfg.n_genes} R{cfg.n_receptors} "
                  f"train={train_batch} | test sizes {ladder}")
            for ts in ladder:
                lo, up = _measure_kt(env, physics, ri, ts, device)
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
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"wrote {out}  ({len(rows)} rows)\n")


if __name__ == "__main__":
    main()
