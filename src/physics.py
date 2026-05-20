# Documented in:
#   doc/theory/02_biophysics_mwc.md  (MWC model, EC50 limit)
#   doc/theory/06_computational_limits.md  (quadrature fallback)
#   doc/theory/07_optimization_pipeline.md  (stage 3: receptor physics)
"""
physics.py — Receptor activation models.

BinaryReceptor implements the EC50-threshold limit of the MWC model:
  ln EC50 = (1/k_sub) Σ_u E_open^(u,ℓ)  →  p = sigmoid(logsumexp_ℓ[ln c − ln EC50] / T)

Key numerical tricks:
  - logsumexp over ligands aggregates mixture binding without float overflow.
  - compute_initial_temperature: sets T_init = std(pre-sigmoid terms) so the
    sigmoid starts with unit standard deviation (avoids vanishing/flat gradients).
  - Gauss-Hermite quadrature for dose-response; falls back to mean-energy
    approximation when grid size would exceed 1e5 points (see §06.2).
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np

class BaseReceptor(nn.Module, ABC):
    """
    Abstract Base Class for Receptor Physics.
    Handles the common boilerplate of gathering units and computing dose-responses.
    """
    def __init__(self, n_genes: int, k_sub: int):
        super().__init__()
        self.n_genes = n_genes
        self.k_sub = k_sub

    @abstractmethod
    def p_open(self, c_reshaped: torch.Tensor, energies_k: torch.Tensor) -> torch.Tensor:
        """
        Calculates the activation probability.
        Must be implemented by the specific physics models.
        """
        pass

    @abstractmethod
    def _extract_mean_energies(self, env, receptor_indices, ligand_id, n_points) -> torch.Tensor:
        """
        Extracts and formats the mean interaction energies from the environment.
        """
        pass

    def forward(self,
                energies: torch.Tensor,
                concentrations: torch.Tensor,
                receptor_indices: torch.Tensor,
                pre_gathered: bool = False) -> torch.Tensor:
        """
        Args:
            energies: Sampled interaction energies.
                      Classic model: (Batch, L, n_genes) for BinaryReceptor or
                                     (Batch, L, n_genes, 2) for MWC.
                      Interface model: (Batch, L, N_Receptors, k_sub) — already
                                       gathered per receptor (pass pre_gathered=True).
            concentrations: (Batch, L) - Sampled concentrations for the mixture
            receptor_indices: (N_Receptors, k_sub) - Receptor layout (needed for
                              n_receptors count even when pre_gathered=True).
            pre_gathered: When True, skip the index-gather step.  The energies
                          tensor is assumed to already have shape
                          (Batch, L, N_Receptors, k_sub).
        """
        batch_size  = energies.shape[0]
        n_ligands   = energies.shape[1]
        n_receptors = receptor_indices.shape[0]

        if pre_gathered:
            # Interface model: energies are (B, L, R, k_sub) — no gather needed.
            energies_k = energies
        else:
            # Classic model: gather per-receptor energies from the unit pool.
            flat_indices = receptor_indices.view(-1)
            gathered_flat = energies[:, :, flat_indices]

            # Reshape dynamically based on the input energy dimensions.
            if energies.dim() == 4:
                # MWC: (Batch, L, R, k_sub, 2)
                energies_k = gathered_flat.view(
                    batch_size, n_ligands, n_receptors, self.k_sub, energies.shape[-1]
                )
            else:
                # BinaryReceptor: (Batch, L, R, k_sub)
                energies_k = gathered_flat.view(batch_size, n_ligands, n_receptors, self.k_sub)

        # 3. Compute Activity
        c_reshaped = concentrations.view(batch_size, n_ligands, 1)
        p_o = self.p_open(c_reshaped, energies_k)

        return torch.clamp(p_o, 0.0, 1.0)

    @torch.no_grad()
    def get_dose_response(self, env, receptor_indices, ligand_id, n_points=200, method='absolute', quadrature_degree=10):
        """
        Calculates the dose-response curve for a given ligand individually.
        If env.distribution_type is 'gaussian' and the receptor is a BinaryReceptor,
        it uses Gauss-Hermite quadrature to integrate over the distribution of
        observation noise for a more accurate response.
        Otherwise, it uses the mean interaction energy.
        """
        # Quadrature over pocket embeddings would require integrating over the
        # averaged pocket position — too complex for the interface model.  Always
        # fall back to the mean-energy approximation in that case.
        if (env.distribution_type != 'gaussian'
                or not isinstance(self, BinaryReceptor)
                or quadrature_degree ** env.latent_dim > 100000
                or getattr(env, 'use_interface_model', False)):
            c_sweep, _ = env.get_concentration_sweep(ligand_id, n_points)
            # Format for a single ligand mixture
            c_reshaped = c_sweep.view(n_points, 1, 1)

            gathered_energies = self._extract_mean_energies(env, receptor_indices, ligand_id, n_points)
            
            p_o = self.p_open(c_reshaped, gathered_energies) 
            
            if method == "self_normalized":
                p_max = torch.max(p_o, dim=0, keepdim=True).values
                p_o =  p_o / (p_max + 1e-8)
            
            return c_sweep.cpu().numpy(), p_o.cpu().numpy()

        # --- Quadrature Method for Gaussian Distribution in a BinaryReceptor ---
        
        device = env.unit_latent.device
        c_sweep, _ = env.get_concentration_sweep(ligand_id, n_points)

        # 1. Get Quadrature nodes and weights for N(0,1)
        nodes, weights = np.polynomial.hermite.hermgauss(quadrature_degree)
        nodes = torch.from_numpy(nodes.astype(np.float32)).to(device) * np.sqrt(2.)
        weights = torch.from_numpy(weights.astype(np.float32)).to(device) / np.sqrt(np.pi)

        # Create grid for latent_dim
        nodes_grid = torch.stack(torch.meshgrid([nodes] * env.latent_dim, indexing='ij'), dim=-1).view(-1, env.latent_dim)
        weights_grid = torch.prod(torch.stack(torch.meshgrid([weights] * env.latent_dim, indexing='ij'), dim=-1).view(-1, env.latent_dim), dim=1)
        n_quad = nodes_grid.shape[0]

        # 2. Get unit and ligand info
        unit_latents = env.unit_latent[receptor_indices]
        ligand_latent = env.ligand_latent[ligand_id]
        base_energies = env.base_energy_u[receptor_indices]

        # 3. Transform nodes to sample from ligand observation noise v ~ N(ligand_latent, env.observation_noise_sigma)
        v_samples = ligand_latent.unsqueeze(0) + nodes_grid * env.observation_noise_sigma

        # 4. Calculate energies using the configured affinity kernel
        diff = v_samples.view(n_quad, 1, 1, env.latent_dim) - unit_latents.view(1, *unit_latents.shape)
        dist_sq = (diff ** 2).sum(dim=-1)
        if env.affinity_kernel == "gaussian":
            max_energies = torch.nn.functional.softplus(env.max_energy_u_raw)[receptor_indices]
            lambda_sq = env.kernel_params[0] ** 2
            E_open_samples = (base_energies.unsqueeze(0)
                              + max_energies.unsqueeze(0) * (1.0 - torch.exp(-dist_sq / lambda_sq)))
        else:  # "quadratic"
            dE = torch.nn.functional.softplus(env.energy_slope_raw)[receptor_indices]
            E_open_samples = base_energies.unsqueeze(0) + dE.unsqueeze(0) * dist_sq

        # 5. Compute p_open for each sample and concentration, then average
        log_ec50 = E_open_samples.mean(dim=-1) # (n_quad, N_r)
        ln_c = torch.log(c_sweep.view(-1, 1, 1) + 1e-12) # (n_points, 1, 1)
        
        # p_o_samples has shape (n_points, n_quad, N_r)
        p_o_samples = torch.sigmoid((ln_c - log_ec50.unsqueeze(0)) / self.temperature)
        
        # Weighted average over quadrature samples
        p_o = torch.einsum('pqr,q->pr', p_o_samples, weights_grid)
        
        if method == "self_normalized":
            p_max = torch.max(p_o, dim=0, keepdim=True).values
            p_o =  p_o / (p_max + 1e-8)
        
        return c_sweep.cpu().numpy(), p_o.cpu().numpy()


@torch.no_grad()
def compute_initial_temperature(env, receptor_indices: torch.Tensor, calibration_batch_size: int = 2048) -> float:
    """
    Returns a well-calibrated T_init by measuring the empirical std of the
    pre-sigmoid log-sum-exp activation over a calibration batch.

    Setting T_init = std(ln_sum_terms) gives the sigmoid arguments unit
    standard deviation at the start of training, avoiding the saturated
    (all 0/1, vanishing gradient) and mushy (all 0.5, non-discriminating)
    regimes.

    Args:
        env: A LigandEnvironment instance (already on the target device).
        receptor_indices: (N_receptors, k_sub) long tensor.
        calibration_batch_size: Number of samples to estimate the spread.

    Returns:
        Scalar float T_init >= 1e-3.
    """
    if getattr(env, 'use_interface_model', False):
        # Interface model: sample_batch returns (B, L, R, k_sub) — already gathered.
        E_open, concs, _ = env.sample_batch(calibration_batch_size, receptor_indices=receptor_indices)
        energies_k = E_open  # (B, L, R, k_sub)
    else:
        E_open, concs, _ = env.sample_batch(calibration_batch_size)
        batch_size  = E_open.shape[0]
        n_ligands   = E_open.shape[1]
        n_receptors = receptor_indices.shape[0]
        k_sub       = receptor_indices.shape[1]
        flat_indices = receptor_indices.view(-1)
        gathered     = E_open[:, :, flat_indices]                          # (B, L, R*k_sub)
        energies_k   = gathered.view(batch_size, n_ligands, n_receptors, k_sub)

    log_ec50     = energies_k.mean(dim=-1)                                 # (B, L, R)
    ln_c         = torch.log(concs.unsqueeze(-1) + 1e-12)                  # (B, L, 1)
    log_terms    = ln_c - log_ec50                                         # (B, L, R)
    ln_sum_terms = torch.logsumexp(log_terms, dim=1)                       # (B, R)

    T_init = ln_sum_terms.std().item()
    return max(T_init, 1e-3)


class MWCReceptor(BaseReceptor):
    """
    The classic MWC Model.
    Expects energies of shape (Batch, L, n_genes, 2) where index 0 is open, 1 is closed.
    """
    def p_open(self, c_reshaped: torch.Tensor, energies_k: torch.Tensor):
        E_open = energies_k[..., 0]
        E_closed = energies_k[..., 1]
        
        ln_c = torch.log(c_reshaped + 1e-12)
        
        # log(c/K) = ln(c) - E
        log_terms_open = ln_c.unsqueeze(-1) - E_open
        # Sum competing ligands in the mixture inside the binding polynomial
        ln_sum_terms_open = torch.logsumexp(log_terms_open, dim=1)
        
        ln_w_open_per_unit = torch.nn.functional.softplus(ln_sum_terms_open)
        ln_W_open = torch.sum(ln_w_open_per_unit, dim=-1)
        
        log_terms_closed = ln_c.unsqueeze(-1) - E_closed
        ln_sum_terms_closed = torch.logsumexp(log_terms_closed, dim=1)
        
        ln_w_closed_per_unit = torch.nn.functional.softplus(ln_sum_terms_closed)
        ln_W_closed = torch.sum(ln_w_closed_per_unit, dim=-1)
        
        return torch.sigmoid(ln_W_open - ln_W_closed)

    def _extract_mean_energies(self, env, receptor_indices, ligand_id, n_points):
        N_Receptors = receptor_indices.shape[0]
        mu_ligand = env.interaction_mu[:, ligand_id, :] 
        mu_receptor = mu_ligand[receptor_indices]
        return mu_receptor.unsqueeze(0).unsqueeze(0).expand(n_points, 1, N_Receptors, self.k_sub, 2)


class BinaryReceptor(BaseReceptor):
    """
    Simplified Threshold (Binary) Model.
    Expects energies of shape (Batch, n_genes) representing the open-state affinity.
    """
    def __init__(self, n_genes: int, k_sub: int, temperature: float):
        super().__init__(n_genes, k_sub)
        self.temperature = temperature

    def p_open(self, c_reshaped: torch.Tensor, energies_k: torch.Tensor):
        # ln(EC50) is the simple average of the subunit open energies
        log_ec50 = energies_k.mean(dim=-1) # Shape: (Batch, L, R)
        
        ln_c = torch.log(c_reshaped + 1e-12) # Shape: (Batch, L, 1)
        
        # Effective binding term per ligand: ln(c) - log_ec50
        log_terms = ln_c - log_ec50 # Shape: (Batch, L, R)
        
        # Sum competing ligands in the mixture
        ln_sum_terms = torch.logsumexp(log_terms, dim=1) # Shape: (Batch, R)
        
        # Temperature-Scaled Binary Activation
        return torch.sigmoid(ln_sum_terms / self.temperature)

    def _extract_mean_energies(self, env, receptor_indices, ligand_id, n_points):
        N_Receptors = receptor_indices.shape[0]

        if getattr(env, 'use_interface_model', False):
            # Interface model: (R, k_sub, n_ligands) → select one ligand → (R, k_sub)
            mu_interface = env.interaction_mu_interface(receptor_indices)  # (R, k_sub, L)
            mu_receptor  = mu_interface[:, :, ligand_id]                   # (R, k_sub)
            return mu_receptor.unsqueeze(0).unsqueeze(0).expand(n_points, 1, N_Receptors, self.k_sub)

        mu_ligand = env.interaction_mu[:, ligand_id]

        if mu_ligand.dim() > 1 and mu_ligand.shape[-1] == 2:
            mu_ligand = mu_ligand[..., 0]

        mu_receptor = mu_ligand[receptor_indices]
        return mu_receptor.unsqueeze(0).unsqueeze(0).expand(n_points, 1, N_Receptors, self.k_sub)