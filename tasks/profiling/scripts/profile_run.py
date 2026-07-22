#!/usr/bin/env python3
"""Profiling task — PROFILING-SPECIFIC, not part of the science pipeline.

Runs a TINY kt sweep (1 run, few epochs) under a profiler to locate hotspots, then
writes the results to /app/data/profiling so sync.sh pulls them back.

  --mode torch      torch.profiler, run ASYNC (normal CUDA). This is the correct way to
                    time the GPU: the profiler uses CUDA events, so it measures real
                    kernel time INCLUDING async overlap. Dumps an op table + a Chrome/
                    Perfetto trace. Use this for "where is GPU time spent".

  --mode cprofile   cProfile of the Python side. Run ASYNC (default) for the ORCHESTRATION
                    / CPU audit: it then measures the real Python overhead (call counts,
                    sweep loop, sampling, logging) while the GPU runs in the background —
                    which is exactly where orchestration design issues show up. Because
                    CUDA is async, GPU compute is NOT charged to its launching line (it
                    lands at the next sync, e.g. .item()); that is fine — use --mode torch
                    for GPU time. Pass --blocking to set CUDA_LAUNCH_BLOCKING=1 (folds GPU
                    time into the launching Python line, serialised & slower — only to see
                    where GPU cost sits in the call tree, NOT for representative CPU timing).

The record_function("prof:...") labels in src/run.py group the ops (sample+physics_fwd /
loss_fwd / backward / eval_kt) in the torch trace; they are inert when no profiler runs.

Open the artifacts:
  data/profiling/trace.json    -> perfetto.dev  or  chrome://tracing
  data/profiling/cprofile.prof -> snakeviz cprofile.prof
                               -> pyprof2calltree -i cprofile.prof -o out.callgrind; kcachegrind out.callgrind

--target separates the two phases (a short lumped run is dominated by the O(B²) final
KT test, NOT the training loop): use --target train for the training breakdown (no KT
eval → training dominates), --target measure to profile the final KT test.

Run on the cluster:
  ../run_remote.sh profiling profile_run.py 0 -- --mode torch --target train
  ../run_remote.sh profiling profile_run.py 0 -- --mode torch --target measure
  ../run_remote.sh profiling profile_run.py 0 -- --mode cprofile --target train
  ../run_remote.sh profiling profile_run.py 0 -- --mode cprofile --blocking
"""
import os
import sys
import argparse

# CUDA_LAUNCH_BLOCKING must be set BEFORE torch initialises CUDA, so peek at argv here
# (before importing torch) rather than after argparse.  [profiling-specific]
if "--blocking" in sys.argv:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

sys.path.append('/app')
from pathlib import Path
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity

from src.config import RunConfig
from src.run import SweepRunner

OUT = Path("/app/data/profiling")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["torch", "cprofile"], default="torch")
    p.add_argument("--target", choices=["train", "measure", "both"], default="train",
                   help="train: no KT eval, so the training loop dominates (representative "
                        "of a real multi-epoch run). measure: profile the O(B²) final KT "
                        "test. both: lumps them (the final test will dominate a short run).")
    p.add_argument("--blocking", action="store_true",
                   help="cprofile only: CUDA_LAUNCH_BLOCKING=1 (see module docstring).")
    p.add_argument("--stack", action="store_true",
                   help="torch: with_stack=True (Python-line attribution). WARNING: bloats "
                        "the trace to GBs and can make trace.json invalid JSON for perfetto — "
                        "use cProfile/kcachegrind for Python attribution instead.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the KT tile kernel (fuse the launch-heavy loop). "
                        "A/B against the eager default to measure the launch-count drop.")
    p.add_argument("--n_genes", type=int, default=10)
    p.add_argument("--n_receptors", type=int, default=20)
    p.add_argument("--n_ligands", type=int, default=200)
    p.add_argument("--epochs", type=int, default=25, help="tiny — keeps the trace small.")
    p.add_argument("--batch_size", default="auto", help="'auto' (realistic) or an int.")
    return p.parse_args()


def build_config(a):
    """A single small kt run exercising the real interface-model + KT path.

    target="train" drops the KT eval (measurement_fns=()) so the final test is cheap and
    the training loop (physics_fwd / loss_fwd / backward) dominates — representative of a
    real run. target="measure"/"both" keep the KT bracket so the O(B²) final test shows.
    """
    bs = a.batch_size if a.batch_size == "auto" else int(a.batch_size)
    meas = () if a.target == "train" else ("entropy_kt", "entropy_kt_upper")
    return RunConfig(
        n_families=6, n_ligands=a.n_ligands, latent_dim=8,
        family_spread=0.3, average_family_distance=1.0,
        environment_geometry="asymmetric", distribution_type="gaussian",
        observation_noise_sigma=0.01,
        n_presence_blocks=1, mu_sources=1, mu_ligands_per_source=25,
        block_shared_conc_mean=False,
        use_interface_model=True, conc_model_type="lognormal",
        conc_mean=tuple(np.random.uniform(-8.0, -3.0, a.n_ligands)),
        conc_std=(1.0,) * a.n_ligands,
        k_sub=5, temperature=0.1, affinity_kernel="gaussian", kernel_params=(1.0,),
        entropy="kt", epochs=a.epochs, lr=0.05, use_scheduler=False,
        batch_size=bs, test_batch_size="auto", per_epoch_measure=False,
        compile_kt=a.compile,
        measurement_fns=meas,
        n_genes=a.n_genes, n_receptors=a.n_receptors,
        receptor_sampling_strategy="cascading", receptor_sampling_seed=0,
        sweep_name="profile", base_folder=str(OUT), warm_start=False,
    )


def run_torch(cfg, tag, with_stack=False):
    """ASYNC torch.profiler — real GPU kernel timing (CUDA events, incl. overlap).

    with_stack defaults OFF: with_stack=True embeds Python source lines into event names,
    which bloats trace.json to GBs AND can make it invalid JSON for perfetto's strict
    parser. Op names + the record_function("prof:*") labels are enough for the GPU
    breakdown; use cProfile/kcachegrind for Python-line attribution.
    """
    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts, record_shapes=True, with_stack=with_stack,
                 profile_memory=True) as prof:
        SweepRunner(cfg).execute()
    sort = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    kw = dict(group_by_stack_n=5) if with_stack else {}
    table = prof.key_averages(**kw).table(sort_by=sort, row_limit=30)
    print(table)
    tbl_path, trace_path = OUT / f"torch_table_{tag}.txt", OUT / f"trace_{tag}.json"
    tbl_path.write_text(table)
    prof.export_chrome_trace(str(trace_path))
    print(f"\n[profiling] wrote {tbl_path} and {trace_path} "
          f"(open the trace in perfetto.dev / chrome://tracing)")


def run_cprofile(cfg, tag, blocking):
    """cProfile — Python/orchestration audit (call counts + cumulative time)."""
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    SweepRunner(cfg).execute()
    pr.disable()
    prof_path = OUT / f"cprofile_{tag}.prof"
    pr.dump_stats(str(prof_path))
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
    print(s.getvalue())
    mode = "SYNC (CUDA_LAUNCH_BLOCKING=1)" if blocking else "ASYNC"
    print(f"[profiling] {mode} cProfile → {prof_path}\n"
          f"  snakeviz {prof_path}\n"
          f"  pyprof2calltree -i {prof_path} -o out.callgrind && kcachegrind out.callgrind")


def main():
    a = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = build_config(a)
    # tag distinguishes A/B runs so their outputs don't overwrite each other
    tag = f"{a.target}_{'compiled' if a.compile else 'eager'}"
    if a.blocking:
        tag += "_blocking"
    print(f"[profiling] mode={a.mode} target={a.target} compile={a.compile} "
          f"blocking={a.blocking} G={a.n_genes} R={a.n_receptors} "
          f"epochs={a.epochs} batch={a.batch_size}  tag={tag}")
    if a.mode == "torch":
        run_torch(cfg, tag, with_stack=a.stack)
    else:
        run_cprofile(cfg, tag, a.blocking)


if __name__ == "__main__":
    main()
