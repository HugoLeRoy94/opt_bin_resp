#!/usr/bin/env python3
"""analytic_baseline: homomer array vs the closed form, sweeping R at fixed D.

The setting is the one the analytics assumes (analytics/homomer_array_mi.tex):
single Gaussian family, gaussian (saturating) kernel, exactly one ligand per
sniff, homomers only, deterministic-ish threshold readout.

Two axes.

  R = n_genes      homomers give one receptor per gene, so sweeping n_genes
                   sweeps R. Crossing R = D moves the array from the
                   exchangeable regime (MI ~ R*eta) into geometric saturation
                   (MI ~ D log2 R). THIS AXIS DOES NOT MOVE nu.

  conc_std         nu is independent of R. It is set by the geometry ratio
                   rho = sigma sqrt(D)/lambda and the dynamic range
                   Gamma = Emax/sigma_c. lambda has a floor (nu_min is
                   non-monotone in it), so the clean knob for nu is the
                   concentration spread. Each conc_std value is one nu.

Run (dry first, it costs nothing and prints what the theory expects):
    python3 tasks/receptors/analytic_baseline/scripts/sweep_R.py --dry-run
    docker compose -f .../docker-compose.yaml run --rm gpu-runner \
        python3 /app/tasks/receptors/analytic_baseline/scripts/sweep_R.py
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

_TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TASK / "_lib"))
sys.path.append("/app")
sys.path.append(str(_TASK.parent.parent.parent))   # repo root, for native runs

LN10 = math.log(10.0)

# --- defaults -------------------------------------------------------------- #
D_DEFAULT = 8
SIGMA = 1.0            # family_spread, per-coordinate ligand cloud spread
LAMBDA = 4.0           # kernel range; rho = sigma sqrt(D)/lambda = 0.707 at D=8
EMAX_INIT = 10.0       # softplus init in environment.py, natural-log units
CONC_MEAN = 0.0        # log10; only sets Ebase init, which is learnable anyway

# conc_std in log10. sigma_c = conc_std * ln10, so these are sigma_c = 4,2,1,0.5
# giving Gamma = 2.5, 5, 10, 20 and nu ~ 3.15, 1.68, 0.99, 0.67 at D=8, lam=4.
CONC_STD_DEFAULT = [1.7372, 0.8686, 0.4343, 0.2171]
R_DEFAULT = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16]

TEMPERATURE = 0.05     # see the preflight: needs T << 0.1 * Delta_priv ~ 0.11
N_LIGANDS = 50000      # one ligand per sniff, so s_upper = 1: a big pool is cheap
TEST_BATCH = 2 ** 18   # log2 B = 18, the ceiling on any counting MI estimate


def theory_table(D, lam, conc_stds, Rs):
    """Print what the closed form expects, before spending any GPU time."""
    import theory as T
    print(f"\nclosed form at D={D}, sigma={SIGMA}, lambda={lam}, "
          f"Emax={EMAX_INIT} (ln units)")
    p0 = T.Params(D=D, sigma=SIGMA, lam=lam, Emax=EMAX_INIT, sigma_c=1.0)
    print(f"  rho = sigma sqrt(D) / lambda = {p0.rho:.3f}"
          f"   (A1 wants this below about 1)")
    print(f"\n  {'conc_std':>8} {'sigma_c':>8} {'Gamma':>7} {'nu':>7} {'eta':>7} "
          f"{'U*':>7} {'zbar':>7} {'Ebase*':>8} {'selfcons':>9} {'T<=':>7}")
    nus = []
    for cs in conc_stds:
        sc = cs * LN10
        p = T.Params(D=D, sigma=SIGMA, lam=lam, Emax=EMAX_INIT, sigma_c=sc,
                     m=CONC_MEAN * LN10)
        o = T.lambert_optimum(p)
        nus.append(o["nu_min"])
        d_priv = 2 * o["U_star"] * SIGMA * EMAX_INIT * o["zbar"] / lam ** 2
        print(f"  {cs:8.4f} {sc:8.3f} {p.Gamma:7.1f} {o['nu_min']:7.3f} "
              f"{T.eta(o['nu_min']):7.3f} {o['U_star']:7.3f} {o['zbar']:7.3f} "
              f"{o['Ebase']:8.2f} {o['selfcons']:9.2f} {0.1*d_priv:7.4f}")
    print(f"\n  selfcons (Eq. 41) below ~2 means the delta route is not "
          f"quantitative there:\n  use theory.exact_optimum for those rows.")
    print(f"\n  predicted MI, and the counting ceiling log2 B = "
          f"{math.log2(TEST_BATCH):.0f}")
    hdr = "  " + "R".rjust(4) + "".join(f"  nu={n:<5.2f}" for n in nus) + "   worst margin"
    print(hdr)
    for R in Rs:
        mis = [T.mi(n, R)["MI"] for n in nus]
        margin = math.log2(TEST_BATCH) - max(mis)
        flag = "" if margin >= 3 else "   <-- TOO CLOSE TO THE CEILING"
        print("  " + f"{R:>4}" + "".join(f"  {m:8.2f}" for m in mis)
              + f"   {margin:6.1f}{flag}")
    cap = math.log2(N_LIGANDS * (max(Rs) + 1))
    print(f"\n  fixed ligand pool caps the alphabet at log2(L*(R+1)) = {cap:.1f} bits")
    print(f"  temperature = {TEMPERATURE}; it must stay under the T<= column above")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the theory table and the planned grid, run nothing")
    ap.add_argument("-D", "--latent-dim", type=int, default=D_DEFAULT)
    ap.add_argument("--lam", type=float, default=LAMBDA)
    ap.add_argument("--R", type=int, nargs="+", default=R_DEFAULT)
    ap.add_argument("--conc-std", type=float, nargs="+", default=CONC_STD_DEFAULT)
    ap.add_argument("--replicates", type=int, default=3,
                    help="independent environments per grid node")
    ap.add_argument("--epochs-per-R", type=int, default=170)
    ap.add_argument("--base-folder", default="/app/data/analytic_baseline")
    args = ap.parse_args()

    D, lam, Rs, stds = args.latent_dim, args.lam, args.R, args.conc_std
    n_nodes = len(Rs) * len(stds) * args.replicates
    print(f"grid: {len(Rs)} R values x {len(stds)} conc_std x "
          f"{args.replicates} replicates = {n_nodes} runs")
    print(f"R crosses D = {D} at R = {D}: "
          f"{sum(r <= D for r in Rs)} runs below, {sum(r > D for r in Rs)} above")

    try:
        theory_table(D, lam, stds, Rs)
    except Exception as exc:                                   # pragma: no cover
        print(f"  (theory table unavailable: {exc})")

    if args.dry_run:
        print("\ndry run, nothing launched")
        return

    from src.config import RunConfig                            # noqa: E402
    from src.run import SweepRunner                             # noqa: E402

    # Cartesian block: R inner, conc_std outer. Replicated so each node gets
    # `replicates` independent environments (the global RNG advances run to run);
    # no seed is shared between conditions, per AGENTS.md.
    blk_R = Rs * len(stds)
    blk_cs = np.repeat(stds, len(Rs)).tolist()
    blk_ep = [int(args.epochs_per_R * r + 500) for r in Rs] * len(stds)

    config = RunConfig(
        # --- environment: one family, continuous-ish ligand pool -------------
        n_families              = 1,
        n_ligands               = N_LIGANDS,
        latent_dim              = D,
        family_spread           = SIGMA,
        average_family_distance = 1.0,
        environment_geometry    = "asymmetric",
        distribution_type       = "gaussian",
        observation_noise_sigma = 0.0,     # A2: the sniff is the ligand itself
        initial_temperature     = "auto",

        # --- presence: exactly one ligand per sniff (A2) ---------------------
        n_presence_blocks      = 1,
        mu_sources             = 1,
        mu_ligands_per_source  = 1,
        block_shared_conc_mean = False,

        use_interface_model = False,
        # A6 of the analytics: ONE learnable (E_base, E_max) pair for the whole
        # array. Without this the simulator optimises n_genes of each, which is a
        # different model from the closed form, and the comparison is only valid
        # where the optimiser happens to make them equal.
        shared_unit_energies = True,

        # --- concentration: lognormal, identical for every ligand ------------
        conc_model_type = "lognormal",
        conc_mean       = CONC_MEAN,
        conc_std        = blk_cs * args.replicates,

        # --- physics: gaussian kernel, near-hard threshold (A1, A3) ----------
        k_sub           = 5,
        temperature     = TEMPERATURE,
        affinity_kernel = "gaussian",
        kernel_params   = (lam,),

        # --- loss / training -------------------------------------------------
        entropy       = "annealed",
        epochs        = blk_ep * args.replicates,
        lr            = 0.01,
        use_scheduler = False,
        batch_size      = "auto",
        test_batch_size = "auto",
        block_size      = 10,
        measurement_fns = ("full_array_entropy", "codeword_entropy"),
        final_measurement_fns = ("full_array_entropy", "codeword_entropy",
                                 "entropy_collision", "mutual_information_ligand",
                                 "mutual_information_concentration"),
        final_test_batch_size = TEST_BATCH,
        eval_chunk_size       = 16384,

        # --- sweep: homomers, R = n_genes ------------------------------------
        n_genes     = blk_R * args.replicates,
        sweep_name  = "homomer_R",
        base_folder = args.base_folder,
        warm_start  = False,
    )

    print(config)
    t0 = time.time()
    SweepRunner(config).execute()
    h, rem = divmod(time.time() - t0, 3600)
    m, s = divmod(rem, 60)
    print(f"\nanalytic_baseline sweep complete  {int(h)}h {int(m)}m {s:.0f}s")


if __name__ == "__main__":
    main()
