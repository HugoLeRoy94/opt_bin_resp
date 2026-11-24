"""Entropy of hard codewords for a continuous Gaussian ligand cloud.

No finite ligand set and no counting: each drawn codeword's probability is
computed by integration, so the error does not grow with H.

Assumptions: one ligand per sniff, no observation noise, hard threshold
(temperature -> 0), all genes sharing E_base and E_max.

    l ~ N(m, sigma^2 I)       ligand position, D dimensions, density g
    s = ln c ~ N(mu_c, sigma_c^2)
    x_g                       position of gene g
    w_rg                      fraction of receptor r's subunits that are gene g
    K_r(l) = sum_g w_rg exp(-|l - x_g|^2 / lam^2)          match, in (0, 1]
    a_r(l) = -ln K_r(l)                                     mismatch
    theta_r(l) = E_base + E_max (1 - exp(-a_r(l)))          ln EC50

w is the identity for homomers, so a_r = |l - x_r|^2 / lam^2. Receptor r fires
when s > theta_r(l). theta grows with a, so the codeword is the set S of the k
receptors with the smallest mismatch, k set by s:

    p(S) = ∫_V(S) dl g(l) w_S(l)
    w_S(l) = Phi((theta_hi(l) - mu_c)/sigma_c) - Phi((theta_lo(l) - mu_c)/sigma_c)

    V(S)      ligand positions whose k best-matched receptors are exactly S. A
              convex polytope for homomers, possibly non-convex and disconnected
              for heteromers
    theta_lo  largest theta inside S (-inf if S is empty)
    theta_hi  smallest theta outside S (+inf if S is everything)
    Phi       standard normal cumulative distribution

so p(S) = P_g(V) E_g[w_S | V]. H = E[-log2 p(S)] is averaged over drawn stimuli.

P_g(V) by subset simulation. With h(l) = max_{i in S} a_i - min_{j not in S} a_j,
V is {h <= 0}. Starting from samples of g, each level keeps the fraction rho
with the smallest h, so P_g(V) is a product of fractions close to rho, and
refreshes them with elliptical slice sampling, which samples g restricted to
{h <= gamma} exactly. The samples that reach V give E_g[w_S | V].
Relative error on p(S) is about sqrt(levels (1-rho) / (rho n_chains)), with
levels ~ log2(1/P_g(V)) at rho = 1/2. It is random across codewords, so it
averages out of H, leaving an upward bias of order its square. When V(S) is
disconnected the chains may not mix between its pieces, an extra bias the
formula above does not cover.

Two channels, since H(Y | l, s) = 0 makes H(Y) = I(Y; l, s):

    exact_entropy_given_ligand          H(Y | l), closed form per l
    exact_entropy_given_concentration   H(Y | s), same subset simulation

    I(Y; l) = H(Y) - H(Y | l)       I(Y; s) = H(Y) - H(Y | s)
"""
import math

import torch


def _mismatch(l, x, log_w, lam):
    """(..., R) a_r(l) = -ln sum_g w_rg exp(-|l - x_g|^2 / lam^2).

    log_w=None means homomers, a_r = |l - x_r|^2 / lam^2. Otherwise shifted by
    the nearest gene before exponentiating, so nothing underflows.
    """
    d2 = ((l ** 2).sum(-1, keepdim=True) + (x ** 2).sum(-1) - 2 * l @ x.T) / lam ** 2
    if log_w is None:
        return d2
    d2_min = d2.amin(-1, keepdim=True)
    return d2_min - torch.logsumexp(log_w - (d2 - d2_min)[..., None, :], dim=-1)


def _extremes(a, S):
    """Largest mismatch inside S and smallest outside, each (...)."""
    inside = S[:, None, :]
    far_in = a.masked_fill(~inside, -math.inf).amax(-1)
    near_out = a.masked_fill(inside, math.inf).amin(-1)
    return far_in, near_out


def _ess(z, gamma, todo, slack, n_moves, max_shrink=60):
    """Elliptical slice sampling of N(0, I) restricted to {slack <= gamma}, in place."""
    C, n, _ = z.shape
    for _ in range(n_moves):
        nu = torch.randn_like(z)
        theta = 2 * math.pi * torch.rand(C, n, 1, dtype=z.dtype, device=z.device)
        lo, hi = theta - 2 * math.pi, theta.clone()
        pending = todo[:, None].expand(C, n).clone()
        for _ in range(max_shrink):
            prop = z * torch.cos(theta) + nu * torch.sin(theta)
            ok = slack(prop) <= gamma[:, None]
            take = pending & ok
            z[take] = prop[take]
            pending &= ~ok
            if not pending.any():
                break
            neg = theta < 0
            lo = torch.where(pending[..., None] & neg, theta, lo)
            hi = torch.where(pending[..., None] & ~neg, theta, hi)
            theta = lo + (hi - lo) * torch.rand_like(theta)
    return z


def _subset(slack, C, D, dtype, device, n_chains, rho, n_moves, max_levels):
    """Natural log of P_g({slack <= 0}) for C regions at once.

    slack maps standard-normal chains (C, n, D) to (C, n). Also returns, per
    region, the chains of the level that reached it, for an average over V.
    """
    z = torch.randn(C, n_chains, D, dtype=dtype, device=device)
    z_end = torch.empty_like(z)
    log_P = torch.zeros(C, dtype=dtype, device=device)
    active = torch.ones(C, dtype=torch.bool, device=device)
    for _ in range(max_levels):
        h = slack(z)
        frac = (h <= 0).to(dtype).mean(1)
        finish = active & (frac >= rho)
        if finish.any():
            log_P[finish] += torch.log(frac[finish])
            z_end[finish] = z[finish]
            active &= ~finish
        if not active.any():
            return log_P, z_end
        gamma = torch.quantile(h, rho, dim=1).masked_fill(~active, math.inf)
        keep = h <= gamma[:, None]
        log_P[active] += torch.log(keep.to(dtype).mean(1))[active]
        idx = torch.multinomial(keep.to(dtype) + (~active)[:, None], n_chains,
                                replacement=True)
        z = torch.gather(z, 1, idx[..., None].expand_as(z))
        z = _ess(z, gamma, active, slack, n_moves)
    raise RuntimeError(f"{int(active.sum())} region(s) not reached in {max_levels} levels")


def _log_p(S, m, sigma, mu_c, sigma_c, e_base, e_max, lam, x, log_w,
           n_chains, rho, n_moves, max_levels):
    """Natural log of p(S) for a batch of codewords S, shape (C, R)."""
    def extremes(z):
        return _extremes(_mismatch(m + sigma * z, x, log_w, lam), S)

    def slack(z):
        far_in, near_out = extremes(z)
        return far_in - near_out

    log_P, z = _subset(slack, S.shape[0], x.shape[1], x.dtype, x.device,
                       n_chains, rho, n_moves, max_levels)
    far_in, near_out = extremes(z)
    inside = far_in - near_out <= 0
    lo = (_theta(far_in, e_base, e_max) - mu_c) / sigma_c
    # theta(+inf) is E_base + E_max, but with no receptor outside S the window
    # has no upper end at all.
    hi = torch.where(torch.isinf(near_out), math.inf,
                     (_theta(near_out, e_base, e_max) - mu_c) / sigma_c)
    w = torch.where(lo > 0, _Phi(-lo) - _Phi(-hi), _Phi(hi) - _Phi(lo))
    return log_P + torch.log((w * inside).sum(1) / inside.sum(1))


def _theta(a, e_base, e_max):
    return e_base + e_max * (1.0 - torch.exp(-a))


def _Phi(z):
    return torch.special.ndtr(z)


def _setup(u, m, w, device):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(u, dtype=torch.float64, device=device)
    m = torch.as_tensor(m, dtype=torch.float64, device=device)
    log_w = None
    if w is not None:
        w = torch.as_tensor(w, dtype=x.dtype, device=device)
        assert torch.allclose(w.sum(1), torch.ones_like(w[:, 0])), "rows of w must sum to 1"
        log_w = torch.log(w)                                              # -inf off-support
    return x, m, log_w, device


def _summary(name, surprisal):
    n = surprisal.numel()
    return {name: surprisal.mean().item(),
            f"{name}_sem": (surprisal.std() / math.sqrt(n)).item(),
            f"{name}_samples": n}


@torch.no_grad()
def exact_codeword_entropy(u, m, sigma, mu_c, sigma_c, e_base, e_max, lam, w=None,
                           n_samples=4096, n_chains=512, rho=0.5, n_moves=5,
                           batch=128, max_levels=400, device=None):
    """H(Y) in bits, with its standard error over the n_samples drawn codewords.

    u: (G, D) gene positions. w: (R, G) subunit fractions of each receptor, rows
    summing to 1. w=None means homomers, one receptor per gene.
    """
    x, m, log_w, device = _setup(u, m, w, device)
    surprisal = []
    for start in range(0, n_samples, batch):
        C = min(batch, n_samples - start)
        l = m + sigma * torch.randn(C, x.shape[1], dtype=x.dtype, device=device)
        s = mu_c + sigma_c * torch.randn(C, 1, dtype=x.dtype, device=device)
        S = _theta(_mismatch(l, x, log_w, lam), e_base, e_max) < s
        log_p = _log_p(S, m, sigma, mu_c, sigma_c, e_base, e_max, lam, x, log_w,
                       n_chains, rho, n_moves, max_levels)
        surprisal.append(-log_p / math.log(2))
    return _summary("entropy_exact", torch.cat(surprisal))


@torch.no_grad()
def exact_entropy_given_ligand(u, m, sigma, mu_c, sigma_c, e_base, e_max, lam, w=None,
                               n_samples=2 ** 16, batch=4096, device=None):
    """H(Y | l) in bits: what is left once the ligand position is known.

    At fixed l, sort the thresholds theta_(1) <= ... <= theta_(R). Exactly the k
    lowest fire when theta_(k) < s < theta_(k+1), so the codeword takes R+1
    values with probabilities

        P(k | l) = Phi((theta_(k+1) - mu_c)/sigma_c) - Phi((theta_(k) - mu_c)/sigma_c)

    with theta_(0) = -inf and theta_(R+1) = +inf. Their entropy is exact; only the
    average over l is Monte Carlo. I(Y; l) = H(Y) - H(Y | l).
    """
    x, m, log_w, device = _setup(u, m, w, device)
    out = []
    for start in range(0, n_samples, batch):
        C = min(batch, n_samples - start)
        l = m + sigma * torch.randn(C, x.shape[1], dtype=x.dtype, device=device)
        t = _theta(_mismatch(l, x, log_w, lam), e_base, e_max).sort(-1).values
        F = _Phi((t - mu_c) / sigma_c)
        F = torch.cat([torch.zeros_like(F[:, :1]), F, torch.ones_like(F[:, :1])], -1)
        p = F.diff(dim=-1).clamp(min=0)
        out.append(-torch.xlogy(p, p).sum(-1) / math.log(2))
    return _summary("entropy_given_ligand", torch.cat(out))


@torch.no_grad()
def exact_entropy_given_concentration(u, m, sigma, mu_c, sigma_c, e_base, e_max, lam,
                                      w=None, n_samples=4096, n_chains=512, rho=0.5,
                                      n_moves=5, batch=128, max_levels=400, device=None):
    """H(Y | s) in bits, s = ln c: what is left once the concentration is known.

    At fixed s, receptor r fires iff a_r(l) < a*(s) = -ln((E_base + E_max - s)/E_max),
    so the codeword is the set of receptors whose region {a_r < a*} contains l,
    and p(S | s) = P_g(V_s(S)) with

        V_s(S) = {l : a_i(l) < a* for i in S, a_j(l) >= a* for j not in S}

    estimated by the same subset simulation as exact_codeword_entropy, with no
    window weight. s <= E_base fires nothing and s >= E_base + E_max fires
    everything, both with probability 1, so they add no surprisal.
    I(Y; s) = H(Y) - H(Y | s).
    """
    x, m, log_w, device = _setup(u, m, w, device)
    surprisal = []
    for start in range(0, n_samples, batch):
        C = min(batch, n_samples - start)
        l = m + sigma * torch.randn(C, x.shape[1], dtype=x.dtype, device=device)
        s = mu_c + sigma_c * torch.randn(C, dtype=x.dtype, device=device)
        z_dem = (e_base + e_max - s) / e_max                    # demand, fires iff K > z
        free = (z_dem > 0) & (z_dem < 1)
        log_p = torch.zeros(C, dtype=x.dtype, device=device)
        if free.any():
            a_star = -torch.log(z_dem[free])                                # (C',)
            S = _mismatch(l[free], x, log_w, lam) < a_star[:, None]

            def slack(z):
                a = _mismatch(m + sigma * z, x, log_w, lam) - a_star[:, None, None]
                inside = S[:, None, :]
                too_far = a.masked_fill(~inside, -math.inf).amax(-1)
                too_near = (-a).masked_fill(inside, -math.inf).amax(-1)
                return torch.maximum(too_far, too_near)

            log_p[free], _ = _subset(slack, int(free.sum()), x.shape[1], x.dtype,
                                     device, n_chains, rho, n_moves, max_levels)
        surprisal.append(-log_p / math.log(2))
    return _summary("entropy_given_conc", torch.cat(surprisal))
