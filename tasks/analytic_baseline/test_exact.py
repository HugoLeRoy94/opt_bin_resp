#!/usr/bin/env python3
"""analytic_baseline measurement: exact codeword entropy for an infinite ligand cloud.

The trained receptor positions are evaluated against the continuous Gaussian
cloud the ligands were drawn from, not against the finite training set, so the
log2(n_ligands) + log2(R+1) ceiling of the finite set does not apply. Each drawn
codeword's probability is integrated rather than counted: the only error is
statistical, entropy_exact_sem, and it does not grow with H. See estimate/exact.py.

Hard threshold, so H(Y|X) = 0 and the entropy is the MI.

    python -m tasks.analytic_baseline.test_exact
"""
import argparse
import math

from estimate.exact import exact_codeword_entropy
from train.io import RunLogger, find_sweeps, is_finished, load_config

LABEL = "exact"


def measure(run_dir, n_samples, n_chains, device):
    import torch.nn.functional as F
    from analysis.load import load_model

    config = load_config(run_dir)
    assert (config.n_families == 1 and config.distribution_type == "gaussian"
            and config.observation_noise_sigma == 0 and config.shared_unit_energies
            and not config.use_interface_model), f"{run_dir}: outside the exact model"
    env, _physics, receptor_indices, _readout = load_model(run_dir)
    assert (receptor_indices == receptor_indices[:, :1]).all(), "homomers only"
    conc = env.concentration_model
    results = exact_codeword_entropy(
        u=env.unit_latent[receptor_indices[:, 0]].double().numpy(),
        m=env.family_latent[0].double().numpy(),
        sigma=env.family_spread,
        mu_c=conc.mu[0].item() * math.log(10),           # stored in log10
        sigma_c=conc.log_sigma[0].exp().item() * math.log(10),
        e_base=env.base_energies()[0].item(),
        e_max=F.softplus(env.max_energies_raw())[0].item(),
        lam=env.kernel_params[0],
        n_samples=n_samples, n_chains=n_chains, device=device)
    results = {k: [float(v)] for k, v in results.items()}
    RunLogger(run_dir).save_results(results, LABEL)
    return run_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="data/analytic_baseline")
    p.add_argument("--prefix", nargs="+", default=["homomer_R"])
    p.add_argument("--n_samples", type=int, default=4096, help="codewords averaged")
    p.add_argument("--n_chains", type=int, default=512, help="samples per codeword integral")
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    todo = []
    for prefix in a.prefix:
        sweeps = [s for s in find_sweeps(a.data_root) if s.name.startswith(prefix)]
        if not sweeps:
            raise FileNotFoundError(f"No sweep under {a.data_root} starting with {prefix!r}.")
        for sweep_root in sweeps:
            for path in sorted(sweep_root.rglob("config.json")):
                run_dir = path.parent
                if is_finished(run_dir) and (a.overwrite
                                             or not (run_dir / f"test_{LABEL}.json").is_file()):
                    todo.append(run_dir)

    print(f"{len(todo)} run(s)  ->  test_{LABEL}.json  "
          f"({a.n_samples} codewords x {a.n_chains} chains)")
    for n, run_dir in enumerate(todo, 1):
        measure(run_dir, a.n_samples, a.n_chains, a.device)
        print(f"  [{n}] {run_dir}")
    print(f"\nmeasured {len(todo)} run(s) as {LABEL!r}")
