#!/usr/bin/env python3
"""two_families: homomer arrays facing two well-separated ligand families.

Prediction (README): each receptor serves one family, and the array splits into two
independent single-family arrays,

    MI_2(R) = H(F) + max_{R_1} [ pi G(R_1) + (1-pi) G(R-R_1) ] - H(F|A)

with G the single-family MI, pi the probability of family 1 and H(F|A) the
ambiguity of the patterns both families produce. G is NOT simulated here: it is
read from the large_R sweep at D = 2, conc_std = 1.7372, whose parameters this
script copies. Only the two-family arm runs, one replicate per R.

    python3 tasks/receptors/two_families/scripts/sweep_two_families.py --dry-run
    ./tasks/run_remote.sh receptors/two_families sweep_two_families.py 0
"""
import argparse
import math
import sys
import time
from pathlib import Path

_TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TASK / "_lib"))
sys.path.append("/app")
sys.path.append(str(_TASK.parent.parent.parent))   # repo root, for native runs

import two_fam as TF                                           # noqa: E402

T, LR = TF.T, TF.LR

# Same as large_R at D = 2, so its single-family runs give G(n).
D = 2
SIGMA = 1.0
EMAX = 10.0
CONC_MEAN = 0.0
CONC_STD = 1.7372
TEMPERATURE = 0.05
N_LIGANDS = 10000
TEST_BATCH = 2 ** 20

PI_DEFAULT = 0.8
FAMILY_DIST_DEFAULT = 8.0     # target average distance between the two centres
R_DEFAULT = [2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64]
EPOCHS_PER_SEC = 17.0         # measured on the 2026-10-19 analytic_baseline sweep


def theory_table(pi, lam, Rs, dist):
    m, sc = T.from_log10(CONC_MEAN, CONC_STD)
    p = T.Params(D=D, sigma=SIGMA, lam=lam, Emax=EMAX, sigma_c=sc, m=m)
    o = T.lambert_optimum(p)
    rad = math.sqrt(D * SIGMA ** 2 + o["U_star"] ** 2)
    print(f"\nD={D}  lambda={lam:.3f}  rho={p.rho:.3f}  Emax={EMAX}  "
          f"conc_std={CONC_STD} (sigma_c={sc:.3f})  pi={pi}")
    print(f"  nu_min={o['nu_min']:.2f}  U*={o['U_star']:.2f}  field radius={rad:.2f}  "
          f"P(s > ceiling)={o['sat_prob']:.3f}  (closed form, approximate here)")
    print(f"  centres: random in a disc, average distance {dist:g}, "
          f"so up to ~{dist / 0.905:.1f} from the origin where the receptors start")

    try:
        df = TF.RU.load(LR.GOAL)
        s = df[(df["latent_dim"] == D) & (abs(df["conc_std"] - CONC_STD) < 1e-6)]
        g = s.groupby("n_genes")["codeword_entropy_mm_mean"].mean()
    except Exception as exc:                                   # pragma: no cover
        print(f"  (large_R data unavailable here: {exc})")
        return
    G = TF.interp_G(g.index.values, g.values)
    print(f"  G(n) from large_R: " + ", ".join(f"{n}:{v:.2f}" for n, v in g.items()))
    print(f"\n  {'R':>4} {'R_1*':>5} {'H(F) + split':>13} {'G(R), 1 family':>15}")
    for R in Rs:
        r1, v = TF.best_split(G, R, pi)
        print(f"  {R:>4} {r1:>5} {TF.h2(pi) + v:>13.2f} {float(G(R)):>15.2f}")
    print("  (before subtracting H(F|A), which the analysis measures)")
    print(f"\n  counting ceiling log2 B = {math.log2(TEST_BATCH):.0f} bits, "
          f"ligand pool cap log2(L(R+1)) = {math.log2(N_LIGANDS * (max(Rs) + 1)):.1f} bits")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pi", type=float, default=PI_DEFAULT,
                    help="probability that a sniff comes from family 1")
    ap.add_argument("--family-dist", type=float, default=FAMILY_DIST_DEFAULT)
    ap.add_argument("--R", type=int, nargs="+", default=R_DEFAULT)
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--epochs-per-R", type=int, default=600)
    ap.add_argument("--epochs-base", type=int, default=2000)
    ap.add_argument("--base-folder", default="/app/data/two_families")
    args = ap.parse_args()

    lam = LR.lam_for(D, SIGMA)
    Rs, rep = args.R, args.replicates
    blk_ep = [int(args.epochs_per_R * r + args.epochs_base) for r in Rs]
    total = sum(blk_ep) * rep
    print(f"grid: {len(Rs)} R x {rep} replicate(s) = {len(Rs) * rep} runs, {total} epochs "
          f"(~{total / EPOCHS_PER_SEC / 3600:.1f} h at {EPOCHS_PER_SEC:.0f} epochs/s, "
          f"rough: large R is slower per epoch)")

    theory_table(args.pi, lam, Rs, args.family_dist)

    if args.dry_run:
        print("\ndry run, nothing launched")
        return

    from src.config import RunConfig                            # noqa: E402
    from src.run import SweepRunner                             # noqa: E402

    config = RunConfig(
        n_families              = 2,
        n_ligands               = N_LIGANDS,
        latent_dim              = D,
        family_spread           = SIGMA,
        average_family_distance = args.family_dist,
        environment_geometry    = "asymmetric",
        distribution_type       = "gaussian",
        observation_noise_sigma = 0.0,
        initial_temperature     = "auto",
        family_weights          = (args.pi, round(1.0 - args.pi, 12)),

        n_presence_blocks      = 1,
        mu_sources             = 1,
        mu_ligands_per_source  = 1e-6,
        block_shared_conc_mean = False,

        use_interface_model  = False,
        shared_unit_energies = True,
        fixed_max_energy     = EMAX,

        conc_model_type = "lognormal",
        conc_mean       = CONC_MEAN,
        conc_std        = CONC_STD,

        k_sub           = 5,
        temperature     = TEMPERATURE,
        affinity_kernel = "gaussian",
        kernel_params   = (lam,),

        entropy       = "annealed",
        epochs        = blk_ep * rep,
        lr            = 0.01,
        use_scheduler = False,
        batch_size      = "auto",
        test_batch_size = "auto",
        block_size      = 10,
        measurement_fns = ("full_array_entropy", "codeword_entropy"),
        # no mutual_information_*: they enumerate all 2^R patterns (see large_R)
        final_measurement_fns = ("full_array_entropy", "codeword_entropy"),
        final_test_batch_size = TEST_BATCH,
        eval_chunk_size = 4096,

        n_genes     = Rs * rep,
        sweep_name  = "fam_R",
        base_folder = args.base_folder,
        warm_start  = False,
    )

    print(config)
    t0 = time.time()
    SweepRunner(config).execute()
    h, rem = divmod(time.time() - t0, 3600)
    m, s = divmod(rem, 60)
    print(f"\ntwo_families sweep complete  {int(h)}h {int(m)}m {s:.0f}s")


if __name__ == "__main__":
    main()
