#!/usr/bin/env python3
"""
convergence.py — does the cell pipeline actually converge to the RIGHT answer?

Not a benchmark: a correctness test with a ground truth we can compute by hand.

  Run:
    python3 tasks/cells/convergence/scripts/convergence.py
    python3 tasks/cells/convergence/scripts/convergence.py --n_ligands 16 --n_cells 8
    ../../run_remote.sh cells/convergence convergence.py 0

## The ground truth

The environment is deliberately crippled so its entropy is a number we know exactly:

  * exactly ONE ligand is present per sniff   (mu_ligands_per_source -> 0, so the
    zero-truncated Poisson puts essentially all its mass on 1)
  * that ligand is drawn UNIFORMLY from the L in the pool
  * its concentration is FIXED               (conc_std ~ 0)

So a sniff carries exactly one thing: which of L ligands arrived.

    H(environment) = log2(L)      bits, exactly.

An array of C cells emits a C-bit word, so it can carry at most C bits. The
information the array can possibly extract is therefore

    ceiling = min(log2(L), C)

and with C > log2(L) the ceiling is log2(L) — set by the WORLD, not by the array.
A converged optimizer should land on it: every ligand mapped to its own distinct
codeword. That is the whole point of this script — the target is not "high", it is
a specific number, and overshooting it is as much a failure as undershooting.

## Why overshooting is the failure that matters

The entropy estimators read each cell activity as the probability that a coin lands
heads. A cell parked at activity 0.5 contributes a full bit of COIN entropy while
telling you nothing about the sniff. An array of such cells reports C bits — the
maximum — and carries zero information. This is a real failure mode that has bitten
this pipeline before (doc/theory/07 §3b.6-3b.7), and it looks like a triumph on the
entropy curve. So the checks below are two-sided:

  1. entropy converged to log2(L)          -- and did not sail past it
  2. no activity sits near 0.5             -- the cells are decided, not coins
  3. distinct codewords ~ L                -- the code really does separate ligands
  4. mutual information ~ log2(L)          -- H minus the coin term is the real signal

A run that passes 1 but fails 2 is the degenerate solution, not a success.

## Measured behaviour (8 ligands, 6 cells, batch 2048, CPU)

Convergence is SLOW and the shortfall is an epoch budget, not a wall:

    epochs |  information  | codewords | firing | coins
      400  |  1.06 / 3.00  |   3 / 8   |  0.06  |  0%
     3000  |  2.75 / 3.00  |   7 / 8   |  0.44  |  0%

Still climbing at 3000, with the firing rate walking up toward the 0.5 the median
threshold targets and the codewords merging apart one at a time. The coin term stayed
at 1e-4 throughout, so every bit reported here is real. Budget accordingly, and read a
FAIL on check 1 together with the codeword count before concluding anything is wrong:
7/8 codewords at 2.75 bits is an under-trained run, not a broken one.
"""
import argparse
import math
import sys

sys.path.append('/app')
sys.path.append('.')

import torch

from src.config import RunConfig
from src.run import SweepRunner


def build_config(args) -> RunConfig:
    L = args.n_ligands
    return RunConfig(
        # --- Environment: one ligand per sniff, fixed concentration ------------
        n_families=args.n_families,
        n_ligands=L,
        latent_dim=args.latent_dim,
        family_spread=0.3,
        average_family_distance=3.0,      # families far apart => ligands separable
        environment_geometry="asymmetric",
        distribution_type="gaussian",
        observation_noise_sigma=0.0,      # no observation noise: purely deterministic
        n_presence_blocks=1,
        mu_sources=1.0,
        mu_ligands_per_source=1e-6,       # ZTP -> P(exactly 1 ligand) ~ 1
        block_shared_conc_mean=False,
        conc_model_type="lognormal",
        conc_mean=(0.0,) * L,
        # 1e-4 and not 0: torch's Normal requires a strictly positive scale. At this
        # width the concentration contributes ~0 bits, so the world entropy is still
        # log2(L) to well inside the tolerance.
        conc_std=(1e-4,) * L,

        # --- Physics ---
        n_genes=args.n_genes,
        k_sub=5,
        temperature=0.05,
        affinity_kernel="gaussian",
        kernel_params=(1.0,),
        use_interface_model=args.interface,

        # --- Cells ---
        n_cells=args.n_cells,
        cell_sampling_strategy="size_pmf",
        # every cell expresses exactly `genes_per_cell` genes
        cell_size_pmf=tuple([0.0] * (args.genes_per_cell - 1) + [1.0]),
        cell_sampling_seed=args.seed,
        cell_stoichiometry=args.stoichiometry,
        cell_readout="threshold",

        # --- Training ---
        batch_size=args.batch_size,
        entropy="kt",
        epochs=args.epochs,
        lr=1e-2,
        use_scheduler=False,
        test_batch_size=args.batch_size,
        measurement_fns=("full_array_entropy",),

        sweep_name=args.name,
        base_folder=args.out,
        warm_start=False,
    )


@torch.no_grad()
def score(run_dir: str, n_samples: int = 8192):
    """Reload the trained run and measure what it actually learned."""
    from src.config import SingleRunConfig
    from src.environment import LigandEnvironment
    from src.physics import BinaryReceptor
    from src.cells import CellArray, CellReadout, cell_activity
    from src.bin_loss import compute_kt_entropy, compute_kt_upper_entropy
    from src import LogNormalConcentration
    import json, os

    cj = json.load(open(os.path.join(run_dir, "config.json")))
    ck = torch.load(os.path.join(run_dir, "best_model.pt"), weights_only=False)
    c = SingleRunConfig(**{k: v for k, v in cj.items()
                           if k in SingleRunConfig.__dataclass_fields__})

    conc = LogNormalConcentration(n_ligands=c.n_ligands, init_mean=c.conc_mean,
                                  init_scale=c.conc_std)
    env = LigandEnvironment(
        c.n_genes, c.n_families, conc_model=conc, n_ligands=c.n_ligands,
        mu_sources=c.mu_sources, mu_ligands_per_source=c.mu_ligands_per_source,
        observation_noise_sigma=c.observation_noise_sigma, latent_dim=c.latent_dim,
        family_spread=c.family_spread, avg_family_distance=c.average_family_distance,
        n_presence_blocks=c.n_presence_blocks, affinity_kernel=c.affinity_kernel,
        kernel_params=c.kernel_params, distribution_type=c.distribution_type,
        use_interface_model=c.use_interface_model,
        block_shared_conc_mean=c.block_shared_conc_mean)
    env.load_state_dict(ck["env_state"]); env.eval()

    phys = BinaryReceptor(c.n_genes, c.k_sub, temperature=c.temperature)
    phys.load_state_dict(ck["physics_state"])

    ca = CellArray(c.cell_gene_sets, c.k_sub, stoichiometry=c.cell_stoichiometry,
                   use_interface_model=c.use_interface_model)
    ro = CellReadout(ca.W, mode=c.cell_readout, temperature=ck["readout_temperature"],
                     k_sub=c.k_sub, learnable_threshold=False)
    ro.load_state_dict(ck["readout_state"], strict=False)

    ri = torch.tensor(c.receptor_indices, dtype=torch.long)
    comp = env.bind_receptors(ri) if c.use_composition else None

    E, cc, masks = env.sample_batch(n_samples, receptor_indices=(ri if c.use_interface_model and comp is None else None))
    A = cell_activity(phys, ro, E, cc, ri, pre_gathered=(c.use_interface_model and comp is None),
                      composition=comp).clamp(1e-6, 1 - 1e-6)

    soft = torch.stack([1 - A, A], dim=-1)
    kt_lo = compute_kt_entropy(soft, chunk_size=2048).item()
    kt_hi = compute_kt_upper_entropy(soft, chunk_size=2048).item()
    # coin term: H(response | sniff). Real information is what is left over.
    noise = (-(A * A.log2() + (1 - A) * (1 - A).log2())).sum(1).mean().item()

    codes = (A > 0.5).to(torch.int64)
    n_distinct = len(torch.unique(codes, dim=0))
    coin_frac = ((A - 0.5).abs() < 1e-3).float().mean().item()
    # which ligand arrived on each sniff (exactly one is present by construction)
    ligand = masks.float().argmax(dim=1)
    return dict(kt_lo=kt_lo, kt_hi=kt_hi, noise=noise, info=kt_lo - noise,
                n_distinct=n_distinct, coin_frac=coin_frac,
                n_cells=A.shape[1], mean_firing=(A > 0.5).float().mean().item(),
                n_ligands_seen=len(torch.unique(ligand)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_ligands", type=int, default=8,
                    help="world entropy is exactly log2(this)")
    ap.add_argument("--n_cells", type=int, default=6)
    ap.add_argument("--n_genes", type=int, default=10)
    ap.add_argument("--genes_per_cell", type=int, default=3)
    ap.add_argument("--n_families", type=int, default=4)
    ap.add_argument("--latent_dim", type=int, default=3)
    ap.add_argument("--stoichiometry", default="multinomial",
                    choices=["multinomial", "uniform"])
    ap.add_argument("--interface", action="store_true",
                    help="use the per-interface (pocket) biophysics model")
    ap.add_argument("--epochs", type=int, default=5000,
                    help="3000 reached 2.75/3.00 bits and was still climbing")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.15,
                    help="allowed shortfall in bits before the run is called FAIL")
    ap.add_argument("--name", default="cell_convergence")
    ap.add_argument("--out", default="/app/data/convergence")
    args = ap.parse_args()

    ceiling = min(math.log2(args.n_ligands), args.n_cells)
    print("=" * 72)
    print(f"  {args.n_ligands} ligands, one per sniff, fixed concentration")
    print(f"  world entropy      = log2({args.n_ligands}) = {math.log2(args.n_ligands):.4f} bits")
    print(f"  array capacity     = {args.n_cells} cells  = {args.n_cells} bits")
    print(f"  => TARGET          = {ceiling:.4f} bits, {args.n_ligands} distinct codewords")
    print("=" * 72)

    cfg = build_config(args)
    SweepRunner(cfg).execute()

    # The sweep writes one run under a timestamped root; take the newest checkpoint.
    import glob, os
    ckpts = glob.glob(os.path.join(args.out, "**", "best_model.pt"), recursive=True)
    if not ckpts:
        print(f"no checkpoint written under {args.out}")
        return 1
    r = score(os.path.dirname(max(ckpts, key=os.path.getmtime)))

    print("\n" + "=" * 72)
    print(f"  ligands actually seen : {r['n_ligands_seen']} / {args.n_ligands}")
    print(f"  KT bracket            : [{r['kt_lo']:.4f}, {r['kt_hi']:.4f}] bits")
    print(f"  coin term (noise)     : {r['noise']:.4f} bits")
    print(f"  information           : {r['info']:.4f} bits   (target {ceiling:.4f})")
    print(f"  distinct codewords    : {r['n_distinct']}       (target {args.n_ligands})")
    print(f"  activities near 0.5   : {r['coin_frac']:.2%}      (target 0%)")
    print(f"  mean firing rate      : {r['mean_firing']:.3f}")
    print("=" * 72)

    checks = [
        ("information reached the world's entropy",
         r["info"] >= ceiling - args.tol),
        ("information did not exceed it (no free bits)",
         r["info"] <= ceiling + args.tol),
        ("cells are decided, not coins",
         r["coin_frac"] < 0.01),
        ("codewords separate the ligands",
         r["n_distinct"] >= min(args.n_ligands, 2 ** args.n_cells) - 1),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    ok = all(c[1] for c in checks)
    print("\n" + ("CONVERGED to the expected result." if ok else
                  "DID NOT converge to the expected result."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
