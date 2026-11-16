"""Entropy of hard homomer codewords for a continuous Gaussian ligand cloud.

No finite ligand set and no counting: each drawn codeword's probability is
computed by integration, so the error does not grow with H.

Assumptions: one ligand per sniff, no observation noise, hard threshold
(temperature -> 0), homomers sharing E_base and E_max.

    l ~ N(m, sigma^2 I)       ligand position, D dimensions, density g
    s = ln c ~ N(mu_c, sigma_c^2)
    u_r                       position of receptor r
    theta_r(l) = E_base + E_max (1 - exp(-|l - u_r|^2 / lam^2))

Receptor r fires when s > theta_r(l). theta grows with distance, so the
codeword is the set S of the k receptors nearest to l, k set by s:

    p(S) = ∫_V(S) dl g(l) w_S(l)
    w_S(l) = Phi((theta_hi(l) - mu_c)/sigma_c) - Phi((theta_lo(l) - mu_c)/sigma_c)

    V(S)      ligand positions whose k nearest receptors are exactly S, a convex
              polytope: 2 (u_j - u_i).l < |u_j|^2 - |u_i|^2 for i in S, j not in S
    theta_lo  largest theta inside S (-inf if S is empty)
    theta_hi  smallest theta outside S (+inf if S is everything)
    Phi       standard normal cumulative distribution

so p(S) = P_g(V) E_g[w_S | V]. H = E[-log2 p(S)] is averaged over drawn stimuli.

P_g(V) by subset simulation. With h(l) = max_{i in S} |l-u_i|^2 - min_{j not in S} |l-u_j|^2,
V is {h <= 0}. Starting from samples of g, each level keeps the fraction rho
with the smallest h, so P_g(V) is a product of fractions close to rho, and
refreshes them with elliptical slice sampling, which samples g restricted to
{h <= gamma} exactly. The samples that reach V give E_g[w_S | V].
Relative error on p(S) is about sqrt(levels (1-rho) / (rho n_chains)), with
levels ~ log2(1/P_g(V)) at rho = 1/2. It is random across codewords, so it
averages out of H, leaving an upward bias of order its square.
"""
import math

import torch


def _h(z, m, sigma, u, S):
    """(C, n) slack: <= 0 inside V(S). Also the extreme squared distances."""
    l = m + sigma * z                                                     # (C, n, D)
    d2 = (l ** 2).sum(-1, keepdim=True) + (u ** 2).sum(-1) - 2 * l @ u.T  # (C, n, R)
    inside = S[:, None, :]
    far_in = d2.masked_fill(~inside, -math.inf).amax(-1)
    near_out = d2.masked_fill(inside, math.inf).amin(-1)
    return far_in - near_out, far_in, near_out


def _ess(z, gamma, todo, m, sigma, u, S, n_moves, max_shrink=60):
    """Elliptical slice sampling of N(0, I) restricted to {h <= gamma}, in place."""
    C, n, _ = z.shape
    for _ in range(n_moves):
        nu = torch.randn_like(z)
        theta = 2 * math.pi * torch.rand(C, n, 1, dtype=z.dtype, device=z.device)
        lo, hi = theta - 2 * math.pi, theta.clone()
        pending = todo[:, None].expand(C, n).clone()
        for _ in range(max_shrink):
            prop = z * torch.cos(theta) + nu * torch.sin(theta)
            ok = _h(prop, m, sigma, u, S)[0] <= gamma[:, None]
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


def _log_p(S, m, sigma, mu_c, sigma_c, e_base, e_max, lam, u,
           n_chains, rho, n_moves, max_levels):
    """Natural log of p(S) for a batch of codewords S, shape (C, R)."""
    C, D = S.shape[0], u.shape[1]
    z = torch.randn(C, n_chains, D, dtype=u.dtype, device=u.device)
    log_P = torch.zeros(C, dtype=u.dtype, device=u.device)
    log_w = torch.zeros(C, dtype=u.dtype, device=u.device)
    active = torch.ones(C, dtype=torch.bool, device=u.device)
    normal = torch.distributions.Normal(0.0, 1.0)

    def theta(d2):
        return e_base + e_max * (1.0 - torch.exp(-d2 / lam ** 2))

    for _ in range(max_levels):
        h, far_in, near_out = _h(z, m, sigma, u, S)
        inside = h <= 0
        frac = inside.to(u.dtype).mean(1)
        finish = active & (frac >= rho)
        if finish.any():
            lo = (theta(far_in) - mu_c) / sigma_c
            # theta(+inf) is E_base + E_max, but with no receptor outside S the
            # window has no upper end at all.
            hi = torch.where(torch.isinf(near_out), math.inf,
                             (theta(near_out) - mu_c) / sigma_c)
            w = torch.where(lo > 0, normal.cdf(-lo) - normal.cdf(-hi),
                            normal.cdf(hi) - normal.cdf(lo))
            mean_w = (w * inside).sum(1) / inside.sum(1)
            log_P[finish] += torch.log(frac[finish])
            log_w[finish] = torch.log(mean_w[finish])
            active &= ~finish
        if not active.any():
            return log_P + log_w
        gamma = torch.quantile(h, rho, dim=1).masked_fill(~active, math.inf)
        keep = h <= gamma[:, None]
        log_P[active] += torch.log(keep.to(u.dtype).mean(1))[active]
        idx = torch.multinomial(keep.to(u.dtype) + (~active)[:, None], n_chains,
                                replacement=True)
        z = torch.gather(z, 1, idx[..., None].expand_as(z))
        z = _ess(z, gamma, active, m, sigma, u, S, n_moves)
    raise RuntimeError(f"{int(active.sum())} codeword(s) not reached in {max_levels} levels")


@torch.no_grad()
def exact_codeword_entropy(u, m, sigma, mu_c, sigma_c, e_base, e_max, lam,
                           n_samples=4096, n_chains=512, rho=0.5, n_moves=5,
                           batch=128, max_levels=400, device=None):
    """H(Y) in bits, with its standard error over the n_samples drawn codewords."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    u = torch.as_tensor(u, dtype=torch.float64, device=device)
    m = torch.as_tensor(m, dtype=torch.float64, device=device)
    surprisal = []
    for start in range(0, n_samples, batch):
        C = min(batch, n_samples - start)
        l = m + sigma * torch.randn(C, u.shape[1], dtype=u.dtype, device=device)
        s = mu_c + sigma_c * torch.randn(C, 1, dtype=u.dtype, device=device)
        d2 = ((l[:, None, :] - u[None]) ** 2).sum(-1)
        S = e_base + e_max * (1.0 - torch.exp(-d2 / lam ** 2)) < s
        log_p = _log_p(S, m, sigma, mu_c, sigma_c, e_base, e_max, lam, u,
                       n_chains, rho, n_moves, max_levels)
        surprisal.append(-log_p / math.log(2))
    surprisal = torch.cat(surprisal)
    return {"entropy_exact": surprisal.mean().item(),
            "entropy_exact_sem": (surprisal.std() / math.sqrt(n_samples)).item(),
            "entropy_exact_samples": n_samples}
