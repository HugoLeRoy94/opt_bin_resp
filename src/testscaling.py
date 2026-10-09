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
  (none)                auto ladder: ×4 batches from --start_samples (counting) or ×4
                        down from the KT memory cap (kt)
KT sizes are clamped to [EVAL_TILE, eval_batch_cap(free)] because its (tile, B) buffer
and its quadratic pairwise work both live on the GPU, and each size is an independent
measurement.

Counting sizes only need be positive, and they are NESTED batches of one growing stream:
the forward is chunked and each sampled response is packed into one or two int64 keys
merged into a CPU frequency table, so the batch of 4n reuses the n sniffs before it and
a ladder costs its largest batch. A counting budget is bounded by time and by the
log2(n) ceiling on a measurable entropy, not by GPU memory, and the needed n is
~46·2^H for an entropy H nobody knows in advance. Hence `convergence_stop`: grow until
the missing mass is small, or a ×4 batch stops paying, or the projected counting table
(one entry per DISTINCT code) would exceed --max_memory_gb. Cheap conditions quit in
seconds, expensive ones keep the full budget.
"""
import argparse
import glob
import os
import resource
import time

import pandas as pd
import torch

from src.IO import SingleRunLoader
from src.plotlib import load_model
from src.analysis_helper import (kt_bracket, count_sampled_responses_ladder,
                                 eval_batch_cap, EVAL_TILE, FWD_CHUNK)


MISSING_TARGET = 0.02    # Good-Turing missing mass below which a batch is converged
DELTA_TARGET = 0.02      # bits gained over a ×4 batch below which growing is pointless
MAX_MEMORY_GB = 32.0     # projected CPU counting memory the next batch may not exceed
BYTES_PER_CODE = (100, 140)   # measured peak CPU bytes per DISTINCT code, by int64 word


def convergence_stop(missing_target=MISSING_TARGET, delta_target=DELTA_TARGET,
                     memory_gb=MAX_MEMORY_GB, announce=print):
    """Build the per-run stop rule for the counting ladder in :func:`run`.

    Returns make(cfg) -> stop(metrics, next_size), the signature
    count_sampled_responses_ladder expects. It is built per run because the memory
    projection needs that run's R.

    Growing the stream stops for one of three stated reasons, so an expensive
    condition keeps its budget while a cheap one quits early instead of being given
    the same one:

    converged  the missing mass (the probability never sampled) is under target, or
               a ×4 bigger batch bought less than `delta_target` bits. Both say the
               frequencies have stopped being about the sample size.
    memory     the counting table holds one entry per DISTINCT code, so its size
               follows the growth of K_hat, not of n. The next batch's K_hat is
               projected from the growth just observed, priced at 100 bytes per code
               (140 above R=62, where a response needs two int64 words) and refused
               if it would exceed the budget.

    There is deliberately no criterion on n itself: the whole point of adapting is
    that the needed n (about 46·2^H) depends on the H we are trying to measure.
    """
    def make(cfg):
        per_code = BYTES_PER_CODE[1 if cfg.n_receptors > 62 else 0]
        state = {}

        def stop(metrics, next_size):
            previous_h, previous_k = state.get("h"), state.get("k")
            h, k = metrics["response_entropy_mm"], metrics["response_counting_K_hat"]
            state["h"], state["k"] = h, k
            if metrics["response_counting_missing_mass"] < missing_target:
                reason = f"missing mass < {missing_target:g}"
            elif previous_h is not None and h - previous_h < delta_target:
                reason = f"last batch bought {h - previous_h:.3f} < {delta_target:g} bit"
            elif next_size is None:
                reason = None
            else:
                growth = (k / previous_k if previous_k else
                          next_size / metrics["response_counting_samples"])
                projected = k * growth * per_code / (1 << 30)
                reason = None if projected <= memory_gb else (
                    f"next batch projects {projected:.3g} GB of counting memory "
                    f"(cap {memory_gb:g}, raise --max_memory_gb if the node has it)")
            if reason and next_size is not None:
                announce(f"    stopping here: {reason}")
            return bool(reason)
        return stop
    return make


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
    p.add_argument("--start_samples", type=int, default=1 << 20,
                   help="counting ladder: first batch (default 2**20); batches grow ×4 and are "
                        "NESTED, so the ladder costs its last batch, not the sum")
    p.add_argument("--missing_target", type=float, default=MISSING_TARGET,
                   help="counting: stop growing once the Good-Turing missing mass is below "
                        f"this (default {MISSING_TARGET})")
    p.add_argument("--max_memory_gb", type=float, default=MAX_MEMORY_GB,
                   help="counting: stop growing when the NEXT batch is projected to need more "
                        f"than this much CPU counting memory (default {MAX_MEMORY_GB}; the "
                        "table holds one entry per distinct code, check the node with free -g)")
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
                      max(args.start_samples, 1 << min(cfg.n_receptors + 6, 62)))
            batches, n = [], args.start_samples
            while n < top:
                batches.append(n)
                n *= 4
            return batches + [top]
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
        measurement="kt", device=None, fwd_chunk=FWD_CHUNK, seed=None, stop=None):
    """Measure one post-hoc estimator at sizes_for(cfg, mem_free) for each selected run.

    ``measurement='kt'`` writes ``test_scaling.csv`` with lower/upper entropy bounds.
    ``measurement='counting'`` writes ``test_counting.csv`` with response entropy and
    mutual-information estimates.  Separate files prevent incomparable estimators
    from being silently combined by existing KT plotting scripts.

    ``seed`` seeds the input stream (global RNG) and, on a separate generator, the
    Bernoulli response draws, so a counting measurement is reproducible.

    For counting the sizes are NESTED batches of one growing stream (see
    count_sampled_responses_ladder), so a ladder costs its largest batch. ``stop`` is
    a factory make(cfg) -> stop(metrics, next_size) (e.g. :func:`convergence_stop`)
    that ends that growth per run, which is how an expensive R gets a big budget and
    a cheap one does not. Pass stop=None to measure every requested size.
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
        out = os.path.join(sweep_root, "test_scaling.csv" if measurement == "kt"
                           else "test_counting.csv")
        for run_dir, cfg in entries:
            rows = []
            env, physics, ri = load_model(run_dir=run_dir, device=device)
            train_batch = cfg.batch_size if isinstance(cfg.batch_size, int) else None
            free = torch.cuda.mem_get_info()[0] if device == "cuda" else 8 * (1 << 30)
            sizes, dropped, cap = _clamp(sizes_for(cfg, free), free, measurement)
            if dropped:
                limits = "positive integers" if cap is None else f"[{EVAL_TILE}, {cap}]"
                print(f"    (dropped {dropped}: outside {limits})")
            print(f"{os.path.basename(sweep_root)} | G{cfg.n_genes} R{cfg.n_receptors} "
                  f"train={train_batch} | sizes {sizes}")
            base = dict(sweep_folder=os.path.basename(sweep_root),
                        run_dir=os.path.relpath(run_dir, sweep_root),
                        n_genes=cfg.n_genes, n_receptors=cfg.n_receptors,
                        train_batch=train_batch)
            if measurement == "kt":
                for ts in sizes:
                    lo, up = kt_bracket(env, physics, ri, ts, tile=EVAL_TILE)
                    rows.append(dict(base, test_size=ts, kt_lower=lo, kt_upper=up))
                    print(f"    test={ts:>9d}  kt_lower={lo:6.3f}  kt_upper={up:6.3f}")
            else:
                timing, mark = [], [time.perf_counter()]

                def show(m):
                    now = time.perf_counter()
                    # Rungs are nested, so `seconds` is the time to grow from the
                    # previous batch and `seconds_total` the cost of reaching this one.
                    # ru_maxrss is the high-water mark of the whole process's
                    # resident memory, i.e. the counting table plus interpreter+model.
                    timing.append(dict(seconds=now - mark[-1], seconds_total=now - mark[0],
                                       peak_ram_gb=resource.getrusage(
                                           resource.RUSAGE_SELF).ru_maxrss / 1e6))
                    mark.append(now)
                    print(f"    test={m['response_counting_samples']:>11d}  "
                          f"H_MM={m['response_entropy_mm']:6.3f}  "
                          f"MI_MM={m['mutual_information_counting_mm']:6.3f}  "
                          f"ceiling={m['response_counting_log2B']:6.3f}  "
                          f"missing_mass={m['response_counting_missing_mass']:.3f}  "
                          f"codes={m['response_counting_K_hat']:.4g}  "
                          f"{timing[-1]['seconds']:.0f}s  "
                          f"ram={timing[-1]['peak_ram_gb']:.1f}GB", flush=True)

                # The train batch was auto-sized to fit WITH gradients and the KT
                # buffer, so 4× it is a safe no-grad forward; a plain --fwd_chunk is
                # not, because the (B, L, R·k_sub) energy tensor grows with R (a
                # 65536 chunk that fits at R=40 is an 80 GB OOM at R=50).
                chunk = min(fwd_chunk, 4 * train_batch) if train_batch else fwd_chunk
                if chunk != fwd_chunk:
                    print(f"    (fwd_chunk {fwd_chunk} -> {chunk}: 4× train batch)")
                batches = count_sampled_responses_ladder(
                    env, physics, ri, sizes, fwd_chunk=chunk,
                    response_generator=generator, report=show,
                    stop=None if stop is None else stop(cfg))
                for metrics, clock in zip(batches, timing):   # show() filled timing
                    rows.append(dict(base, **metrics, fwd_chunk=chunk, **clock,
                                     test_size=metrics["response_counting_samples"]))
            del env, physics
            if device == "cuda":
                torch.cuda.empty_cache()
            # Flush after EVERY run: one 4e9-sniff batch is hours of GPU, and an OOM
            # or a walltime kill three runs later must not throw away the ones that
            # already finished.
            _merge_csv(out, rows)

        print()


def _merge_csv(out, rows):
    """Append rows to the sweep CSV, newest measurement of a (run, size) winning."""
    if not rows:
        return
    new = pd.DataFrame(rows)
    if os.path.exists(out):
        new = pd.concat([pd.read_csv(out), new], ignore_index=True)
    new = (new.drop_duplicates(["run_dir", "test_size"], keep="last")
              .sort_values(["run_dir", "test_size"]))
    new.to_csv(out, index=False)
    print(f"    wrote {out}  ({len(new)} rows total)", flush=True)
