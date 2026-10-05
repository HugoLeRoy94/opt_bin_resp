"""Post-hoc test-size scaling orchestration.

Reload each run in a set of sweeps (best_model.pt) and re-measure either the KT bracket
or stochastic response-counting entropy/MI at chosen test sizes, without retraining.
The per-run measurements live in analysis_helper; THIS module only orchestrates: walk
sweeps, pick runs, choose sizes, write the CSV. Task scripts
(tasks/*/scripts/test_scaling.py) are thin wrappers over build_parser + sizes_from_args + run.

Size strategies (sizes_from_args, in priority order):
  --test_sizes A B ...  explicit absolute sizes
  --largest f           single size = f × that run's train batch B
  --mult m1 m2 ...      per-run multiples of that run's train batch (B, 2B, 4B, ...)
  (none)                auto geometric ladder of --n_test points (×4), downwards from
                        the KT memory cap (kt), or from the counting budget
                        min(--max_samples, 64·2^R) (counting)
KT sizes are clamped to [EVAL_TILE, eval_batch_cap(free)] because its (tile, B) buffer
and its quadratic pairwise work both live on the GPU. Counting sizes only need be
positive: the forward is chunked and each sampled response is packed into one or two
int64 keys merged into a CPU frequency table, so a counting size is bounded by time
and by the log2(n) ceiling on a measurable entropy, not by GPU memory. Its default
budget is 64× the binary alphabet 2^R, where the bias measured against exact
enumeration (≈3.34·(n/2^H)^-0.92 bits, doc/theory/07) is under 0.1 bit — capped at
--max_samples and floored at 2^18 sniffs, so small R stays cheap.
"""
import argparse
import glob
import os

import pandas as pd
import torch

from src.IO import SingleRunLoader
from src.plotlib import load_model
from src.analysis_helper import (kt_bracket, count_sampled_responses,
                                 eval_batch_cap, EVAL_TILE, FWD_CHUNK)


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
    p.add_argument("--largest", type=float, default=None,
                   help="measure a SINGLE size = min(largest×B, memory cap) per run — the "
                        "largest feasible test batch, for the final all-env figure values")
    p.add_argument("--measurement", choices=("kt", "counting"), default="kt",
                   help="KT entropy bracket (default), or streamed stochastic response "
                        "counting that reports response entropy and mutual information")
    p.add_argument("--n_test", type=int, default=5, help="auto ladder: number of sizes (×4 apart)")
    p.add_argument("--max_samples", type=int, default=1 << 26,
                   help="counting auto ladder: cap on the per-run budget min(cap, 64*2**R) "
                        "(default 2**26 = 67.1M sniffs → entropies up to 26 bits measurable, "
                        "~9 min/run of forward pass and ~9 GB of CPU counting memory at R=75)")
    p.add_argument("--fwd_chunk", type=int, default=FWD_CHUNK,
                   help="forward sub-batch; bounds GPU memory, independent of the test size")
    p.add_argument("--seed", type=int, default=None,
                   help="seed the input and response sampling (reproducible measurement)")
    p.add_argument("--n_receptors", type=int, nargs="+", default=None,
                   help="only measure runs with these R (default: all)")
    p.add_argument("--per_condition", action="store_true",
                   help="measure only the FIRST run per (n_genes, R) — skip repeat envs (faster)")
    return p


def sizes_from_args(args):
    """Return a sizes_for(cfg, mem_free) -> list[int] callable per the chosen strategy."""
    def sizes_for(cfg, mem_free):
        b = cfg.batch_size if isinstance(cfg.batch_size, int) else 0
        if args.test_sizes:
            return list(args.test_sizes)
        if getattr(args, "largest", None) is not None:                  # single largest feasible
            size = int(round(args.largest * b))
            return [min(size, eval_batch_cap(mem_free)) if args.measurement == "kt" else size]
        if args.mult:
            return [int(round(m * b)) for m in args.mult]
        if args.measurement == "counting":                              # auto ladder
            top = min(args.max_samples,
                      max(1 << 18, 1 << min(cfg.n_receptors + 6, 62)))
        else:
            top = min(1 << cfg.n_receptors, eval_batch_cap(mem_free))
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


def _clamp(sizes, mem_free, measurement):
    if measurement == "counting":
        keep = [s for s in sorted(set(sizes)) if s > 0]
        return keep, [s for s in sizes if s not in keep], None
    cap = eval_batch_cap(mem_free)
    keep = [s for s in sorted(set(sizes)) if EVAL_TILE <= s <= cap]
    return keep, [s for s in sizes if s not in keep], cap


def run(data_root, sweep_glob, sizes_for, *, n_receptors=None, per_condition=False,
        measurement="kt", device=None, fwd_chunk=FWD_CHUNK, seed=None):
    """Measure one post-hoc estimator at sizes_for(cfg, mem_free) for each selected run.

    ``measurement='kt'`` writes ``test_scaling.csv`` with lower/upper entropy bounds.
    ``measurement='counting'`` writes ``test_counting.csv`` with response entropy and
    mutual-information estimates.  Separate files prevent incomparable estimators
    from being silently combined by existing KT plotting scripts.

    ``seed`` seeds the input stream (global RNG) and, on a separate generator, the
    Bernoulli response draws, so a counting measurement is reproducible.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator = None
    if seed is not None:
        torch.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed + 104729)
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
            sizes, dropped, cap = _clamp(sizes_for(cfg, free), free, measurement)
            if dropped:
                limits = "positive integers" if cap is None else f"[{EVAL_TILE}, {cap}]"
                print(f"    (dropped {dropped}: outside {limits})")
            print(f"{os.path.basename(sweep_root)} | G{cfg.n_genes} R{cfg.n_receptors} "
                  f"train={train_batch} | sizes {sizes}")
            for ts in sizes:
                row = dict(sweep_folder=os.path.basename(sweep_root),
                           run_dir=os.path.relpath(run_dir, sweep_root),
                           n_genes=cfg.n_genes, n_receptors=cfg.n_receptors,
                           train_batch=train_batch, test_size=ts)
                if measurement == "kt":
                    lo, up = kt_bracket(env, physics, ri, ts, tile=EVAL_TILE)
                    row.update(kt_lower=lo, kt_upper=up)
                    print(f"    test={ts:>9d}  kt_lower={lo:6.3f}  kt_upper={up:6.3f}")
                else:
                    row.update(count_sampled_responses(env, physics, ri, ts,
                                                       fwd_chunk=fwd_chunk,
                                                       response_generator=generator))
                    print(f"    test={ts:>9d}  H_MM={row['response_entropy_mm']:6.3f}  "
                          f"MI_MM={row['mutual_information_counting_mm']:6.3f}  "
                          f"ceiling={row['response_counting_log2B']:6.3f}  "
                          f"missing_mass={row['response_counting_missing_mass']:.3f}")
                rows.append(row)
            del env, physics
            if device == "cuda":
                torch.cuda.empty_cache()

        out = os.path.join(sweep_root, "test_scaling.csv" if measurement == "kt"
                           else "test_counting.csv")
        new = pd.DataFrame(rows)
        if os.path.exists(out):
            new = pd.concat([pd.read_csv(out), new], ignore_index=True)
        new = (new.drop_duplicates(["run_dir", "test_size"], keep="last")
                  .sort_values(["run_dir", "test_size"]))
        new.to_csv(out, index=False)
        print(f"wrote {out}  ({len(new)} rows total)\n")
