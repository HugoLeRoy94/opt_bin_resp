"""Receptor activation: the EC50-threshold limit of the MWC model.

MWC = Monod-Wyman-Changeux. EC50 = concentration at half activation. For a
heteromer it is the geometric mean of its pocket affinities, so ln EC50 is the
arithmetic mean of the open-state energies of its k_sub pockets.

    ln EC50_r = (1/k_sub) * sum_p n_pr E_open[p, l]       n_pr = composition[p, r]
    p_open_r  = sigmoid( logsumexp_l( ln c_l - ln EC50_rl ) / T )

k_sub = number of subunits (and pockets) per receptor (5 for a pentamer).
T = temperature of the threshold, the width of the sigmoid in log-concentration.
logsumexp over ligands aggregates mixture binding without overflow.
"""
import torch
import torch.nn as nn


class BinaryReceptor(nn.Module):
    def __init__(self, n_genes: int, k_sub: int, temperature: float):
        super().__init__()
        self.n_genes = n_genes
        self.k_sub = k_sub
        self.temperature = temperature

    def forward(self, energies, concentrations, composition):
        """Activation probabilities, shape (B, R).

        energies (B, L, P): one open-state energy per distinct pocket.
        concentrations (B, L). composition (P, R): how many of receptor r's k_sub
        pockets are pocket p. See LigandEnvironment.bind_receptors.
        """
        log_ec50 = torch.matmul(energies, composition) / self.k_sub
        c = concentrations.view(concentrations.shape[0], concentrations.shape[1], 1)
        return torch.clamp(self._p_open(c, log_ec50), 0.0, 1.0)

    def _p_open(self, c_reshaped, log_ec50):
        """Mixture aggregation then threshold, from ln EC50 of shape (B, L, R)."""
        ln_c = torch.log(c_reshaped + 1e-12)
        return torch.sigmoid(torch.logsumexp(ln_c - log_ec50, dim=1) / self.temperature)


@torch.no_grad()
def compute_initial_temperature(env, receptor_indices, calibration_batch_size=2048):
    """T_init = std of the pre-sigmoid term, so the sigmoid starts at unit spread.

    Without it the sigmoid starts either saturated (all 0 or 1, vanishing
    gradient) or mushy (all 0.5, non-discriminating).
    """
    E_open, concs, _ = env.sample_batch(calibration_batch_size)
    log_ec50 = torch.matmul(E_open, env.composition) / receptor_indices.shape[1]

    ln_c = torch.log(concs.unsqueeze(-1) + 1e-12)
    ln_sum_terms = torch.logsumexp(ln_c - log_ec50, dim=1)
    return max(ln_sum_terms.std().item(), 1e-3)
