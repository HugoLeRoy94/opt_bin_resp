import math
import torch
import copy
import torch.nn as nn
import torch.distributions as dist
from typing import Tuple
from abc import ABC, abstractmethod

class UniformNBall:
    """
    Custom differentiable sampler for an N-dimensional solid sphere (N-ball).
    """
    def __init__(self, loc: torch.Tensor, radius: float, dim: int):
        self.loc = loc          # Shape: (Batch, latent_dim)
        self.radius = radius    # Scalar
        self.dim = dim          # Scalar (latent_dim)

    def rsample(self) -> torch.Tensor:
        # 1. Sample a perfectly random direction
        direction = torch.randn_like(self.loc)
        direction = torch.nn.functional.normalize(direction, p=2, dim=-1)
        
        # 2. Sample the radius to ensure uniform volume density
        # Draw U ~ Uniform(0, 1)
        u = torch.rand(self.loc.shape[0], 1, device=self.loc.device)
        r = self.radius * (u ** (1.0 / self.dim))
        
        # 3. Scale and shift
        return self.loc + (direction * r)

class ConcentrationModel(nn.Module, ABC):
    """
    Abstract Base Class for different concentration strategies.
    Subclass this to create LogNormal, Normal, Bimodal, etc.
    """
    @abstractmethod
    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Returns concentrations for the ligand pool.
        Shape: (batch_size, n_ligands)
        """
        pass        
    @abstractmethod
    def get_expected_log_c(self) -> torch.Tensor:
        """
        Returns the expected natural logarithm of the concentration for each ligand.
        Shape: (n_ligands,)
        """
        pass
    @abstractmethod
    def get_distribution(self, ligand_id: int) -> dist.Distribution:
        """Returns the torch distribution object for a specific ligand."""
        pass
    @abstractmethod
    def get_sweep_and_pdf(self, ligand_id: int, n_points: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the physical concentration sweep and its PDF."""
        pass

class LogNormalConcentration(ConcentrationModel):
    """
    Classic Biophysics assumption: c spans orders of magnitude.
    log10(c) ~ Normal(mu, sigma)
    """
    def __init__(self, n_ligands: int, init_mean: float, init_scale: float):
        super().__init__()
        
        mu_tensor = torch.tensor(init_mean, dtype=torch.float32)
        if mu_tensor.ndim == 0:
            mu_tensor = torch.ones(n_ligands) * mu_tensor
            
        sigma_tensor = torch.tensor(init_scale, dtype=torch.float32)
        if sigma_tensor.ndim == 0:
            sigma_tensor = torch.ones(n_ligands) * sigma_tensor
        
        self.register_buffer('mu', mu_tensor)
        self.register_buffer('log_sigma', torch.log(sigma_tensor))

    def sample(self, batch_size):
        # Sample Log-Space for the whole batch and all ligands
        batch_mu = self.mu.unsqueeze(0).expand(batch_size, -1)
        batch_sigma = torch.exp(self.log_sigma).unsqueeze(0).expand(batch_size, -1)
        
        dist_log = dist.Normal(batch_mu, batch_sigma)
        log_c = dist_log.rsample()
        
        # Convert to Real-Space
        return torch.pow(10.0, log_c)
        
    def get_expected_log_c(self):
        # mu is log10(c). To get natural log ln(c): ln(c) = log10(c) * ln(10)
        return self.mu * math.log(10.0)
    
    def get_entropy_linear(self):
        """
        Entropy of the base-10 Log-Normal distribution.

        """
        # 1. Entropy of the underlying Normal distribution (log10 space) in bits
        sigma = torch.exp(self.log_sigma)
        h_normal = torch.log2(sigma * math.sqrt(2 * math.pi * math.e))
        
        # 2. Add the Jacobian contribution for the 10^x transformation
        # E[ln(c)] = mu * ln(10)
        jacobian_term = (self.mu * math.log(10.0) + math.log(math.log(10.0))) / math.log(2.0)
        
        return h_normal + jacobian_term
    
    def get_entropy_log(self):
        """
        Entropy of a Normal distribution in bits.
        """
        sigma = torch.exp(self.log_sigma)
        return torch.log2(sigma * math.sqrt(2 * math.pi * math.e))

    @torch.no_grad()
    def get_distribution(self, ligand_id: int):
        mu = self.mu[ligand_id]
        sigma = torch.exp(self.log_sigma[ligand_id])
        return dist.Normal(mu, sigma) # Distribution of log10(c)
        
    @torch.no_grad()
    def get_sweep_and_pdf(self, ligand_id: int, n_points: int = 200):
        d = self.get_distribution(ligand_id)
        # Sweep in log10 space
        x_log10 = torch.linspace(d.mean - 3*d.stddev, d.mean + 3*d.stddev, n_points, device=self.mu.device)
        pdf = torch.exp(d.log_prob(x_log10))
        # Convert to physical concentration (Molar)
        c_sweep = 10.0 ** x_log10
        return c_sweep, pdf

class NormalConcentration(ConcentrationModel):
    """
    Simple Gaussian assumption.
    c ~ Normal(mu, sigma) clamped at 0.
    """
    def __init__(self, n_ligands: int, init_mean: float, init_scale: float):
        super().__init__()
        mu_tensor = torch.tensor(init_mean, dtype=torch.float32)
        if mu_tensor.ndim == 0:
            mu_tensor = torch.ones(n_ligands) * mu_tensor
            
        sigma_tensor = torch.tensor(init_scale, dtype=torch.float32)
        if sigma_tensor.ndim == 0:
            sigma_tensor = torch.ones(n_ligands) * sigma_tensor
            
        self.register_buffer('mu', mu_tensor)
        self.register_buffer('log_sigma', torch.log(sigma_tensor))

    def sample(self, batch_size):
        batch_mu = self.mu.unsqueeze(0).expand(batch_size, -1)
        batch_sigma = torch.exp(self.log_sigma).unsqueeze(0).expand(batch_size, -1)
        
        c = dist.Normal(batch_mu, batch_sigma).rsample()
        return torch.clamp(c, min=1e-12) # Physics constraint (clamp > 0 for safe log)
        
    def get_expected_log_c(self):
        # Approximate expected log(c) as log(mu)
        return torch.log(torch.clamp(self.mu, min=1e-12))
    
    def get_entropy(self):
        """
        Entropy of a Normal distribution in bits.
        """
        sigma = torch.exp(self.log_sigma)
        return torch.log2(sigma * math.sqrt(2 * math.pi * math.e))

    @torch.no_grad()
    def get_distribution(self, ligand_id: int):
        mu = self.mu[ligand_id]
        sigma = torch.exp(self.log_sigma[ligand_id])
        return dist.Normal(mu, sigma) # Distribution of c
    
    @torch.no_grad()
    def get_sweep_and_pdf(self, ligand_id: int, n_points: int = 200):
        d = self.get_distribution(ligand_id)
        # Sweep already in linear physical space
        c_sweep = torch.linspace(d.mean - 3*d.stddev, d.mean + 3*d.stddev, n_points, device=self.mu.device)
        pdf = torch.exp(d.log_prob(c_sweep))
        return c_sweep, pdf

class LigandEnvironment(nn.Module):
    def __init__(self, n_genes: int, n_families: int, conc_model: ConcentrationModel,
                 n_ligands: int, p_presence: list, observation_noise_sigma: float,
                 latent_dim: int, family_spread: float, distribution_type: str,
                 avg_family_distance: float, affinity_length_scale: float = 1.0):
        """
        Args:
            n_genes: Number of gene units
            n_families: Number of ligand families
            conc_model: An INSTANCE of a ConcentrationModel subclass
            n_ligands: Size of the fixed ligand pool
            p_presence: Probability of a ligand appearing in a mixture
            observation_noise_sigma: Observation noise magnitude
            latent_dim: Dimensionality of the chemical latent space
            family_spread: Fixed spatial spread (fuzziness) of the ligand families
            distribution_type: 'gaussian' or 'uniform'
            avg_family_distance: Target average Euclidean distance between ligand families
            affinity_length_scale: Global λ — controls Gaussian breadth of affinity kernel
        """
        super().__init__()
        self.n_genes = n_genes
        self.n_families = n_families
        self.n_ligands = n_ligands
        self.latent_dim = latent_dim
        self.family_spread = family_spread
        self.avg_family_distance = avg_family_distance
        self.p_presence = p_presence
        self.observation_noise_sigma = observation_noise_sigma
        self.affinity_length_scale = affinity_length_scale

        p_tensor = torch.tensor(p_presence, dtype=torch.float32)
        self.register_buffer('p_presence_tensor', p_tensor)

        if distribution_type not in ['gaussian', 'uniform', 'shell']:
            raise ValueError("distribution_type must be 'gaussian', 'uniform', or 'shell'")
        self.distribution_type = distribution_type

        # 1. Inject the Concentration Strategy
        self.concentration_model = conc_model

        # ----------------------------------------------------------------------
        # MECHANISTIC LATENT SPACE INITIALIZATION
        # ----------------------------------------------------------------------

        # Learnable Unit Coordinates
        self.unit_latent = nn.Parameter(torch.randn(n_genes, latent_dim) * 1.0)

        # Per-unit saturation ceiling: E_o(u, ℓ_far) - E_o(u, ℓ_opt) = softplus(max_energy_u_raw)
        # Initialized so softplus(x) ≈ 10.0 (ln(e^10 - 1) ≈ 10)
        self.max_energy_u_raw = nn.Parameter(torch.full((n_genes,), math.log(math.e ** 10.0 - 1.0)))

        # 2. The Environment is Fixed (Family Prototype Coordinates)
        fixed_families = self._generate_family_centers(n_families, latent_dim)
        self.register_buffer('family_latent', fixed_families)

        # 2.5 Ligand Pool Initialization
        with torch.no_grad():
            # Assign each ligand to a family randomly
            ligand_family_assignments = torch.randint(0, n_families, (n_ligands,))
            self.register_buffer('ligand_family_assignments', ligand_family_assignments)

            # Draw permanent base coordinates for the ligands
            base_centers = fixed_families[ligand_family_assignments]
            if self.distribution_type == 'gaussian':
                ligand_dist = dist.Normal(loc=base_centers, scale=self.family_spread)
                fixed_ligands = ligand_dist.rsample()
            elif self.distribution_type == 'uniform_cube':
                low = base_centers - self.family_spread
                high = base_centers + self.family_spread
                ligand_dist = dist.Uniform(low=low, high=high)
                fixed_ligands = ligand_dist.rsample()
            elif self.distribution_type == 'uniform':
                ligand_dist = UniformNBall(loc=base_centers, radius=self.family_spread, dim=self.latent_dim)
                fixed_ligands = ligand_dist.rsample()
            elif self.distribution_type == 'shell':
                # Defeats high-D concentration-of-measure: direction uniform on sphere,
                # radius uniform in [0, family_spread] — equal weight at every shell radius.
                direction = torch.nn.functional.normalize(
                    torch.randn_like(base_centers), p=2, dim=-1
                )
                r = torch.rand(n_ligands, 1) * self.family_spread
                fixed_ligands = base_centers + direction * r

            self.register_buffer('ligand_latent', fixed_ligands)

        # 3. Unit-specific Base Energies
        # E_base(u) = E_o(u, ℓ_opt): open-state energy at the optimal (zero-distance) ligand.
        # Initialised to E[ln c] so EC50 matches the expected concentration at the start.
        global_avg_log_c = self.concentration_model.get_expected_log_c().mean().item()
        self.base_energy_u = nn.Parameter(torch.ones(n_genes) * global_avg_log_c)

    def clone_with_extra_units(self, extra_units: int = 1):
        """
        Creates a copy of the environment with additional units.
        The new units are initialized randomly as in the constructor, 
        while the existing units retain their learned parameters.
        """
        new_conc_model = copy.deepcopy(self.concentration_model)
        
        # Use self.__class__ to automatically support subclasses like SymmetricLigandEnvironment
        new_env = self.__class__(
            n_genes=self.n_genes + extra_units,
            n_families=self.n_families,
            conc_model=new_conc_model,
            n_ligands=self.n_ligands,
            p_presence=self.p_presence,
            observation_noise_sigma=self.observation_noise_sigma,
            latent_dim=self.latent_dim,
            family_spread=self.family_spread,
            distribution_type=self.distribution_type,
            avg_family_distance=self.avg_family_distance,
            affinity_length_scale=self.affinity_length_scale,
        ).to(self.unit_latent.device)

        with torch.no_grad():
            new_env.family_latent.copy_(self.family_latent)
            new_env.ligand_family_assignments.copy_(self.ligand_family_assignments)
            new_env.ligand_latent.copy_(self.ligand_latent)
            new_env.unit_latent.data[:self.n_genes] = self.unit_latent.data.clone()
            new_env.max_energy_u_raw.data[:self.n_genes] = self.max_energy_u_raw.data.clone()
            new_env.base_energy_u.data[:self.n_genes] = self.base_energy_u.data.clone()
            
        return new_env

    def _generate_family_centers(self, n_families: int, latent_dim: int) -> torch.Tensor:
        """
        Default behavior: Prototype centers spread inside a uniform N-ball,
        with the radius calculated to yield the requested average distance.
        """
        # 1. Empirically find the average distance in a unit N-ball
        with torch.no_grad():
            loc_mc = torch.zeros(20000, latent_dim)
            unit_sampler = UniformNBall(loc=loc_mc, radius=1.0, dim=latent_dim)
            p1, p2 = unit_sampler.rsample(), unit_sampler.rsample()
            unit_avg_dist = torch.norm(p1 - p2, dim=-1).mean().item()
            
        # 2. Scale radius to match target average distance
        target_radius = self.avg_family_distance / unit_avg_dist
        
        loc_centers = torch.zeros(n_families, latent_dim)
        sampler = UniformNBall(loc=loc_centers, radius=target_radius, dim=latent_dim)
        return sampler.rsample()

    @property
    def interaction_mu(self) -> torch.Tensor:
        """
        Open-state energy for each unit against each ligand's exact center.
        Shape: (n_genes, n_ligands)
        """
        diff = self.unit_latent.unsqueeze(1) - self.ligand_latent.unsqueeze(0)  # (U, L, D)
        dist_sq = (diff ** 2).sum(dim=-1)  # (U, L)
        max_e = torch.nn.functional.softplus(self.max_energy_u_raw)  # (U,)
        lambda_sq = self.affinity_length_scale ** 2
        mu_open = self.base_energy_u.unsqueeze(1) + max_e.unsqueeze(1) * (1.0 - torch.exp(-dist_sq / lambda_sq))
        return mu_open

    def sample_batch(self, batch_size: int):
        device = self.unit_latent.device
        
        # 1. Sample mixture masks: 1 if ligand is present, 0 if absent
        mixture_masks = torch.bernoulli(self.p_presence_tensor.unsqueeze(0).expand(batch_size, -1))
        
        # 2. Sample physical concentrations and mask out absent ligands
        concs = self.concentration_model.sample(batch_size) * mixture_masks
        
        # 3. Add orientation/observation noise to the ligand latent coordinates
        noise = torch.randn(batch_size, self.n_ligands, self.latent_dim, device=device) * self.observation_noise_sigma
        v_ligands = self.ligand_latent.unsqueeze(0) + noise

        # 4. Calculate Energies using saturating affinity kernel
        # Avoid the (B, L, U, D) broadcast by using ||a-b||² = ||a||² + ||b||² - 2<a,b>.
        a_sq    = (v_ligands ** 2).sum(dim=-1, keepdim=True)                   # (B, L, 1)
        b_sq    = (self.unit_latent ** 2).sum(dim=-1)                          # (U,)
        ab      = torch.einsum('bld,ud->blu', v_ligands, self.unit_latent)     # (B, L, U)
        dist_sq = (a_sq + b_sq[None, None, :] - 2.0 * ab).clamp(min=0.0)     # (B, L, U)
        max_e = torch.nn.functional.softplus(self.max_energy_u_raw)  # (U,)
        lambda_sq = self.affinity_length_scale ** 2
        E_open = (self.base_energy_u.unsqueeze(0).unsqueeze(0)
                  + max_e.unsqueeze(0).unsqueeze(0) * (1.0 - torch.exp(-dist_sq / lambda_sq)))  # (Batch, L, U)
        
        return E_open, concs, mixture_masks

    @torch.no_grad()
    def get_concentration_sweep(self, ligand_id: int, n_points: int = 200):
        return self.concentration_model.get_sweep_and_pdf(ligand_id, n_points)

class SymmetricLigandEnvironment(LigandEnvironment):
    """
    Environment that enforces symmetric, evenly spaced prototype centers 
    to explicitly test heteromerization gaps.
    """
    def _generate_family_centers(self, n_families: int, latent_dim: int) -> torch.Tensor:
        if latent_dim < 2:
            points = torch.linspace(-1, 1, n_families).unsqueeze(-1)
            
        elif latent_dim == 2:
            angles = torch.linspace(0, 2 * math.pi, n_families + 1)[:-1]
            x = torch.cos(angles)
            y = torch.sin(angles)
            points = torch.stack([x, y], dim=1)
            
        elif latent_dim == 3 and n_families == 4:
            points = torch.tensor([
                [ math.sqrt(8/9),  0.0,             -1/3],
                [-math.sqrt(2/9),  math.sqrt(2/3),  -1/3],
                [-math.sqrt(2/9), -math.sqrt(2/3),  -1/3],
                [ 0.0,             0.0,              1.0]
            ], dtype=torch.float32)

        elif latent_dim == 3 and n_families == 6:
            points = torch.tensor([
                [ 1.0,  0.0,  0.0], [-1.0,  0.0,  0.0],
                [ 0.0,  1.0,  0.0], [ 0.0, -1.0,  0.0],
                [ 0.0,  0.0,  1.0], [ 0.0,  0.0, -1.0]
            ], dtype=torch.float32)

        else:
            # Fallback to the base class random initialization if no geometry is defined
            print(f"Warning: No explicit geometric structure defined for D={latent_dim}, F={n_families}. Falling back to random initialization.")
            return super()._generate_family_centers(n_families, latent_dim)
            
        # Scale symmetrically structured points to match avg_family_distance
        dist_matrix = torch.cdist(points, points, p=2.0)
        n_pairs = n_families * (n_families - 1)
        current_avg_dist = dist_matrix.sum().item() / n_pairs if n_pairs > 0 else 1.0
        
        scale_factor = self.avg_family_distance / current_avg_dist
        return points * scale_factor