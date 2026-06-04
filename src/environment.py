# Documented in:
#   doc/theory/03_latent_environment.md  (affinity kernel, latent space, concentration models)
#   doc/theory/07_optimization_pipeline.md  (stages 1 & 2: world building, batch sampling)
"""
environment.py — Chemical latent space and batch sampling.

Implements LigandEnvironment (and SymmetricLigandEnvironment) which holds all
learnable parameters and fixed buffers (family_latent, ligand_latent).
sample_batch() generates one training batch: presence masks → sampled
concentrations → noisy ligand coordinates → open-state energies via the affinity
kernel.

Classic model (use_interface_model=False):
  Learnable per unit: unit_latent (U,D), base_energy_u (U,), max_energy_u_raw (U,)
  sample_batch returns E_open: (B, L, U)

Interface model (use_interface_model=True):
  Each unit has two faces: unit_latent_plus / unit_latent_minus (U,D each).
  Binding pockets sit at the interface between adjacent units in the ring.
  Interface i: pocket embedding = 0.5*(v_plus[u_i] + v_minus[u_{i+1 mod k}]).
  sample_batch(batch_size, receptor_indices) returns E_open: (B, L, R, k_sub),
  already gathered per receptor interface — no further indexing needed in physics.

Presence sampling — hierarchical source→ligand model:
  Ligands are partitioned into K = n_presence_blocks source blocks (deterministic
  seeded partition, block size ≈ n_ligands/K).  Per sniff:
    1. Draw n_src ~ ZTP(mu_sources, max=K); select n_src distinct blocks uniformly.
    2. For each active block k (size m_k): draw n_lig ~ ZTP(mu_ligands_per_source,
       max=m_k); select n_lig distinct ligands within block k uniformly.
    3. Set M[b, picked] = 1; all else 0.  Every row has S ≥ 1 by construction.
  Knobs: mu_sources and mu_ligands_per_source (Poisson rates, zero-truncated).
  ZTP = zero-truncated Poisson sampled on [1, n_max] via unnormalized PMF.

Key numerical tricks:
  - Distance computation: ‖a−b‖² = ‖a‖²+‖b‖²−2⟨a,b⟩ avoids the (B,L,U,D) broadcast.
  - UniformNBall: direction·radius^(1/D) gives uniform density in the N-ball.
  - base_energy init = E[ln c]: EC50 ≈ typical concentration at iteration 0.
  - Gumbel-top-k selection: argsort of Gumbel noise gives uniform subset draws.
"""
import math
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from typing import Optional, Tuple
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
                 n_ligands: int, observation_noise_sigma: float,
                 latent_dim: int, family_spread: float, distribution_type: str,
                 avg_family_distance: float, n_presence_blocks: int,
                 mu_sources: float, mu_ligands_per_source: float,
                 affinity_kernel: str = "gaussian", kernel_params: list = None,
                 use_interface_model: bool = False, block_shared_conc_mean: bool = False):
        """
        Args:
            n_genes: Number of gene units.
            n_families: Number of ligand families.
            conc_model: Instance of a ConcentrationModel subclass.
            n_ligands: Size of the fixed ligand pool.
            observation_noise_sigma: Observation noise magnitude.
            latent_dim: Dimensionality of the chemical latent space.
            family_spread: Fixed spatial spread of ligand families.
            distribution_type: 'gaussian' or 'uniform'.
            avg_family_distance: Target average Euclidean distance between families.
            n_presence_blocks: K source blocks (deterministic partition from
                               (n_ligands, K)).  K=1 means one source, K=n_ligands
                               means each ligand is its own source.
            mu_sources: Poisson rate for the zero-truncated draw of active sources
                        per sniff (capped at K).
            mu_ligands_per_source: Poisson rate for the zero-truncated draw of
                                   active ligands per source (capped at block size).
            affinity_kernel: "gaussian" (E_base + E_max*(1−exp(−d²/λ²))) or
                             "quadratic" (E_base + d²).
            kernel_params: [lambda] for "gaussian", [] for "quadratic".
            use_interface_model: When True, each unit has two faces (+/−) and
                                 binding energy is computed at the interface between
                                 adjacent subunits in the ring.
            block_shared_conc_mean: When True and n_presence_blocks > 1, all ligands
                                    in the same block share one concentration mean.
        """
        super().__init__()
        self.n_genes = n_genes
        self.n_families = n_families
        self.n_ligands = n_ligands
        self.latent_dim = latent_dim
        self.family_spread = family_spread
        self.avg_family_distance = avg_family_distance
        self.mu_sources = mu_sources
        self.mu_ligands_per_source = mu_ligands_per_source
        self.observation_noise_sigma = observation_noise_sigma
        self.affinity_kernel = affinity_kernel
        self.kernel_params = kernel_params if kernel_params is not None else []
        self.use_interface_model = use_interface_model
        self.n_presence_blocks = n_presence_blocks
        self.block_shared_conc_mean = block_shared_conc_mean

        # ----------------------------------------------------------------------
        # PRESENCE BLOCK PARTITION (independent of ligand families)
        # Seeded deterministically from (n_ligands, n_presence_blocks).
        # ----------------------------------------------------------------------
        _g = torch.Generator()
        _g.manual_seed(n_ligands * 131071 + n_presence_blocks)  # 131071 is Mersenne prime
        _perm = torch.randperm(n_ligands, generator=_g)

        _block_size = n_ligands // n_presence_blocks
        _rem = n_ligands % n_presence_blocks
        _sizes = torch.full((n_presence_blocks,), _block_size, dtype=torch.long)
        _sizes[:_rem] += 1  # first _rem blocks get one extra ligand
        _block_ids_sorted = torch.repeat_interleave(
            torch.arange(n_presence_blocks, dtype=torch.long), _sizes
        )
        _block_ids = torch.empty(n_ligands, dtype=torch.long)
        _block_ids[_perm] = _block_ids_sorted
        self.register_buffer('presence_block_id', _block_ids)

        # Hierarchical sampler helper buffers.
        # _block_members: (K, max_m) padded member indices; _block_mask: (K, max_m) validity.
        _max_m = int(_sizes.max().item())
        _block_members = torch.zeros(n_presence_blocks, _max_m, dtype=torch.long)
        _block_valid   = torch.zeros(n_presence_blocks, _max_m, dtype=torch.bool)
        for _k in range(n_presence_blocks):
            _idx = (_block_ids == _k).nonzero(as_tuple=False).squeeze(1)
            _block_members[_k, :len(_idx)] = _idx
            _block_valid[_k,   :len(_idx)] = True
        self.register_buffer('_block_members', _block_members)  # (K, max_m)
        self.register_buffer('_block_valid',   _block_valid)    # (K, max_m)
        self.register_buffer('_block_sizes',   _sizes)          # (K,)

        if distribution_type not in ['gaussian', 'uniform', 'shell']:
            raise ValueError("distribution_type must be 'gaussian', 'uniform', or 'shell'")
        self.distribution_type = distribution_type

        # 1. Inject the Concentration Strategy
        self.concentration_model = conc_model

        # Optional block-shared concentration mean.
        # Turbulent transport: co-occurrence is preserved but concentration ratios are not —
        # only the *mean* is shared per block; per-sniff concentrations draw independently.
        # NOTE: conc_model.mu may already be on CUDA; _block_ids built on CPU, so resolve
        # the target device from conc_model.mu and move before arithmetic.
        if block_shared_conc_mean and n_presence_blocks > 1:
            with torch.no_grad():
                _mu  = conc_model.mu.clone()
                _dev = _mu.device
                _ids = _block_ids.long().to(_dev)
                # bincount is CPU-only; compute there, then move.
                _block_cnt = torch.bincount(
                    _block_ids.long(), minlength=n_presence_blocks
                ).float().to(_dev)
                _block_sum = torch.zeros(
                    n_presence_blocks, dtype=_mu.dtype, device=_dev
                ).scatter_add(0, _ids, _mu)
                _block_mean = _block_sum / _block_cnt
                conc_model.mu.copy_(_block_mean[_ids])

        # ----------------------------------------------------------------------
        # MECHANISTIC LATENT SPACE INITIALIZATION
        # ----------------------------------------------------------------------

        _init_val = math.log(math.e ** 10.0 - 1.0)   # softplus(x) ≈ 10 at init

        if use_interface_model:
            # Each unit has two faces; pockets form at adjacent (+)/(−) interfaces.
            self.unit_latent_plus  = nn.Parameter(torch.randn(n_genes, latent_dim) * 1.0)
            self.unit_latent_minus = nn.Parameter(torch.randn(n_genes, latent_dim) * 1.0)
            if affinity_kernel == "gaussian":
                self.max_energy_u_raw_plus  = nn.Parameter(torch.full((n_genes,), _init_val))
                self.max_energy_u_raw_minus = nn.Parameter(torch.full((n_genes,), _init_val))
            else:  # "quadratic"
                self.energy_slope_raw_plus  = nn.Parameter(torch.full((n_genes,), _init_val))
                self.energy_slope_raw_minus = nn.Parameter(torch.full((n_genes,), _init_val))
        else:
            # Classic per-unit model.
            self.unit_latent = nn.Parameter(torch.randn(n_genes, latent_dim) * 1.0)
            if affinity_kernel == "gaussian":
                self.max_energy_u_raw = nn.Parameter(torch.full((n_genes,), _init_val))
            else:  # "quadratic"
                self.energy_slope_raw = nn.Parameter(torch.full((n_genes,), _init_val))

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
                ligand_dist_obj = dist.Normal(loc=base_centers, scale=self.family_spread)
                fixed_ligands = ligand_dist_obj.rsample()
            elif self.distribution_type == 'uniform_cube':
                low = base_centers - self.family_spread
                high = base_centers + self.family_spread
                ligand_dist_obj = dist.Uniform(low=low, high=high)
                fixed_ligands = ligand_dist_obj.rsample()
            elif self.distribution_type == 'uniform':
                ligand_dist_obj = UniformNBall(loc=base_centers, radius=self.family_spread, dim=self.latent_dim)
                fixed_ligands = ligand_dist_obj.rsample()
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
        # E_base = E_o(u, ℓ_opt): open-state energy at the optimally matched ligand.
        # Initialised to E[ln c] so EC50 matches the expected concentration at start.
        global_avg_log_c = self.concentration_model.get_expected_log_c().mean().item()
        if use_interface_model:
            self.base_energy_u_plus  = nn.Parameter(torch.ones(n_genes) * global_avg_log_c)
            self.base_energy_u_minus = nn.Parameter(torch.ones(n_genes) * global_avg_log_c)
        else:
            self.base_energy_u = nn.Parameter(torch.ones(n_genes) * global_avg_log_c)

    def clone_with_extra_units(self, extra_units: int = 1):
        """Creates a copy of the environment with additional gene units.

        Existing unit parameters are preserved; new units are randomly initialized.
        """
        new_conc_model = copy.deepcopy(self.concentration_model)

        # Resolve device from any parameter (works for both model modes).
        device = next(self.parameters()).device

        # Use self.__class__ to automatically support subclasses like SymmetricLigandEnvironment
        new_env = self.__class__(
            n_genes=self.n_genes + extra_units,
            n_families=self.n_families,
            conc_model=new_conc_model,
            n_ligands=self.n_ligands,
            mu_sources=self.mu_sources,
            mu_ligands_per_source=self.mu_ligands_per_source,
            observation_noise_sigma=self.observation_noise_sigma,
            latent_dim=self.latent_dim,
            family_spread=self.family_spread,
            distribution_type=self.distribution_type,
            avg_family_distance=self.avg_family_distance,
            n_presence_blocks=self.n_presence_blocks,
            affinity_kernel=self.affinity_kernel,
            kernel_params=self.kernel_params,
            use_interface_model=self.use_interface_model,
            block_shared_conc_mean=self.block_shared_conc_mean,
        ).to(device)

        with torch.no_grad():
            new_env.family_latent.copy_(self.family_latent)
            new_env.ligand_family_assignments.copy_(self.ligand_family_assignments)
            new_env.ligand_latent.copy_(self.ligand_latent)

            if self.use_interface_model:
                new_env.unit_latent_plus.data[:self.n_genes]  = self.unit_latent_plus.data.clone()
                new_env.unit_latent_minus.data[:self.n_genes] = self.unit_latent_minus.data.clone()
                if self.affinity_kernel == "gaussian":
                    new_env.max_energy_u_raw_plus.data[:self.n_genes]  = self.max_energy_u_raw_plus.data.clone()
                    new_env.max_energy_u_raw_minus.data[:self.n_genes] = self.max_energy_u_raw_minus.data.clone()
                else:
                    new_env.energy_slope_raw_plus.data[:self.n_genes]  = self.energy_slope_raw_plus.data.clone()
                    new_env.energy_slope_raw_minus.data[:self.n_genes] = self.energy_slope_raw_minus.data.clone()
                new_env.base_energy_u_plus.data[:self.n_genes]  = self.base_energy_u_plus.data.clone()
                new_env.base_energy_u_minus.data[:self.n_genes] = self.base_energy_u_minus.data.clone()
            else:
                new_env.unit_latent.data[:self.n_genes] = self.unit_latent.data.clone()
                if self.affinity_kernel == "gaussian":
                    new_env.max_energy_u_raw.data[:self.n_genes] = self.max_energy_u_raw.data.clone()
                else:
                    new_env.energy_slope_raw.data[:self.n_genes] = self.energy_slope_raw.data.clone()
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
        Shape: (n_genes, n_ligands).

        Classic model only.  For the interface model call interaction_mu_interface().
        """
        diff = self.unit_latent.unsqueeze(1) - self.ligand_latent.unsqueeze(0)  # (U, L, D)
        dist_sq = (diff ** 2).sum(dim=-1)  # (U, L)
        if self.affinity_kernel == "gaussian":
            max_e = F.softplus(self.max_energy_u_raw)  # (U,)
            lambda_sq = self.kernel_params[0] ** 2
            return self.base_energy_u.unsqueeze(1) + max_e.unsqueeze(1) * (1.0 - torch.exp(-dist_sq / lambda_sq))
        else:  # "quadratic"
            dE = F.softplus(self.energy_slope_raw)  # (U,)
            return self.base_energy_u.unsqueeze(1) + dE.unsqueeze(1) * dist_sq

    def interaction_mu_interface(self, receptor_indices: torch.Tensor) -> torch.Tensor:
        """
        Mean pocket energy per interface, evaluated at exact ligand centers.

        Interface i of receptor r: pocket between unit_latent_plus[r_i] (+ face)
        and unit_latent_minus[r_{i+1 mod k}] (− face).

        Args:
            receptor_indices: (R, k_sub) long tensor of ordered ring arrangements.

        Returns:
            (R, k_sub, n_ligands) float tensor — E_pocket^{(r,i,ℓ)}.
        """
        idx_i = receptor_indices                          # (R, k_sub)
        idx_j = receptor_indices.roll(-1, dims=1)         # (R, k_sub) — next unit (cyclic)

        v_plus_r  = self.unit_latent_plus[idx_i]          # (R, k_sub, D)
        v_minus_r = self.unit_latent_minus[idx_j]         # (R, k_sub, D)
        v_pocket  = 0.5 * (v_plus_r + v_minus_r)          # (R, k_sub, D)

        E_base_pocket = 0.5 * (
            self.base_energy_u_plus[idx_i] + self.base_energy_u_minus[idx_j]
        )  # (R, k_sub)

        R, k, D = v_pocket.shape
        L = self.n_ligands
        # Broadcast distance: (R, k_sub, n_ligands)
        diff = v_pocket.view(R, k, 1, D) - self.ligand_latent.view(1, 1, L, D)
        dist_sq = (diff ** 2).sum(dim=-1)  # (R, k, L)

        if self.affinity_kernel == "gaussian":
            E_max_pocket = 0.5 * (
                F.softplus(self.max_energy_u_raw_plus[idx_i])
                + F.softplus(self.max_energy_u_raw_minus[idx_j])
            )  # (R, k_sub)
            lambda_sq = self.kernel_params[0] ** 2
            return (E_base_pocket.unsqueeze(-1)
                    + E_max_pocket.unsqueeze(-1) * (1.0 - torch.exp(-dist_sq / lambda_sq)))
        else:  # "quadratic"
            E_slope_pocket = 0.5 * (
                F.softplus(self.energy_slope_raw_plus[idx_i])
                + F.softplus(self.energy_slope_raw_minus[idx_j])
            )  # (R, k_sub)
            return E_base_pocket.unsqueeze(-1) + E_slope_pocket.unsqueeze(-1) * dist_sq

    def _sample_masks(self, batch_size: int) -> torch.Tensor:
        """(B, L) float presence mask via the hierarchical source→ligand sampler.

        Per sniff:
          1. Draw n_src ~ ZTP(mu_sources, K); select n_src distinct source blocks.
          2. For each active block k: draw n_lig ~ ZTP(mu_ligands_per_source, m_k);
             select n_lig distinct ligands within block k.
          3. M[b, picked] = 1; all else 0.

        ZTP (zero-truncated Poisson) on [1, n_max]: build unnormalized PMF
        lambda^s / s! for s = 1..n_max, normalize, sample via torch.multinomial.

        Distinct selection uses Gumbel-top-k (argsort of Gumbel(0,1) noise);
        entries with rank < n are kept via (arange(width) < n[:, None]).

        # HOOK: the two uniform selections (sources and ligands-per-source) are the
        # only place where per-block or per-ligand frequency weights would re-enter.
        # Replace the zero log-weights (torch.zeros) with log(w_k) or log(w_l) to
        # implement nonuniform sampling over sources or ligands within a source.
        """
        device = self.presence_block_id.device
        B  = batch_size
        K  = self.n_presence_blocks
        L  = self.n_ligands
        mu_s = self.mu_sources
        mu_l = self.mu_ligands_per_source

        # ------------------------------------------------------------------
        # Step 1 — draw n_src per sniff and select distinct source blocks
        # ------------------------------------------------------------------
        # Zero-truncated Poisson on [1, K].
        s_vals = torch.arange(1, K + 1, device=device, dtype=torch.float32)  # (K,)
        log_pmf_s = s_vals * math.log(mu_s) - torch.lgamma(s_vals + 1)       # (K,) unnorm log-PMF
        log_pmf_s = log_pmf_s - log_pmf_s.logsumexp(0)
        # (B,) number of active sources per sniff
        n_src = torch.multinomial(log_pmf_s.exp(), B, replacement=True) + 1  # values in [1,K]

        # Gumbel-top-k over K blocks: (B, K) — zero log-weights → uniform
        # HOOK: replace torch.zeros with log(w_block) for nonuniform source weights.
        gumbel_s = -torch.log(-torch.log(
            torch.rand(B, K, device=device).clamp(1e-9, 1.0 - 1e-9)
        ))  # Gumbel(0,1)
        rank_s = gumbel_s.argsort(dim=1, descending=True)          # (B, K) sorted indices
        active_src = (torch.arange(K, device=device).unsqueeze(0)  # (B, K) bool
                      < n_src.unsqueeze(1))                         # True for top n_src entries
        # active_src_mask[b, k] = True iff source block k is active in sniff b
        active_src_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
        active_src_mask.scatter_(1, rank_s, active_src)

        # ------------------------------------------------------------------
        # Step 2 — for each active block, draw n_lig distinct ligands
        # ------------------------------------------------------------------
        M = torch.zeros(B, L, device=device)
        for k in range(K):
            m_k = int(self._block_sizes[k].item())
            members_k = self._block_members[k, :m_k]   # (m_k,) ligand indices

            # (B,) indicator: is block k active in this sniff?
            active_b = active_src_mask[:, k]            # (B,) bool
            n_active = int(active_b.sum().item())
            if n_active == 0:
                continue

            # Zero-truncated Poisson on [1, m_k].
            l_vals = torch.arange(1, m_k + 1, device=device, dtype=torch.float32)
            log_pmf_l = l_vals * math.log(mu_l) - torch.lgamma(l_vals + 1)
            log_pmf_l = log_pmf_l - log_pmf_l.logsumexp(0)
            # (n_active,) number of ligands picked per active sniff
            n_lig = torch.multinomial(log_pmf_l.exp(), n_active, replacement=True) + 1

            # Gumbel-top-k over m_k ligands for the n_active sniffs.
            # HOOK: replace torch.zeros with log(w_ligand_k) for nonuniform weights.
            gumbel_l = -torch.log(-torch.log(
                torch.rand(n_active, m_k, device=device).clamp(1e-9, 1.0 - 1e-9)
            ))
            rank_l = gumbel_l.argsort(dim=1, descending=True)           # (n_active, m_k)
            picked = (torch.arange(m_k, device=device).unsqueeze(0)
                      < n_lig.unsqueeze(1))                              # (n_active, m_k) bool
            # picked_mask[i, j] = True iff ligand rank_l[i,j] is among top n_lig[i]
            picked_mask = torch.zeros(n_active, m_k, dtype=torch.bool, device=device)
            picked_mask.scatter_(1, rank_l, picked)

            # Map within-block positions back to global ligand indices and scatter into M.
            global_idx = members_k.unsqueeze(0).expand(n_active, -1)   # (n_active, m_k)
            M[active_b] = M[active_b].scatter(1, global_idx, picked_mask.float())

        return M

    def _sample_noisy_ligands(self, batch_size: int) -> torch.Tensor:
        """(B, L, D) ligand coordinates with i.i.d. observation noise."""
        device = self.ligand_latent.device
        noise = (torch.randn(batch_size, self.n_ligands, self.latent_dim, device=device)
                 * self.observation_noise_sigma)
        return self.ligand_latent.unsqueeze(0) + noise  # (B, L, D)

    def _compute_energies(
        self,
        v_ligands: torch.Tensor,
        receptor_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Open-state energies from noisy ligand coordinates.

        Classic model   → (B, L, U)
        Interface model → (B, L, R, k_sub)

        ‖a−b‖² = ‖a‖²+‖b‖²−2⟨a,b⟩ avoids the (B, L, U, D) broadcast.
        For the interface model, pockets are flattened to (R*k, D) then reshaped.
        """
        if not self.use_interface_model:
            # ------------------------------------------------------------------
            # Classic path: (B, L, U)
            # ------------------------------------------------------------------
            a_sq    = (v_ligands ** 2).sum(dim=-1, keepdim=True)               # (B, L, 1)
            b_sq    = (self.unit_latent ** 2).sum(dim=-1)                      # (U,)
            ab      = torch.einsum('bld,ud->blu', v_ligands, self.unit_latent) # (B, L, U)
            dist_sq = (a_sq + b_sq[None, None, :] - 2.0 * ab).clamp(min=0.0)  # (B, L, U)
            if self.affinity_kernel == "gaussian":
                max_e     = F.softplus(self.max_energy_u_raw)                  # (U,)
                lambda_sq = self.kernel_params[0] ** 2
                return (self.base_energy_u[None, None, :]
                        + max_e[None, None, :] * (1.0 - torch.exp(-dist_sq / lambda_sq)))
            else:  # "quadratic"
                dE = F.softplus(self.energy_slope_raw)                         # (U,)
                return self.base_energy_u[None, None, :] + dE[None, None, :] * dist_sq

        # ----------------------------------------------------------------------
        # Interface path: (B, L, R, k_sub)
        # ----------------------------------------------------------------------
        if receptor_indices is None:
            raise ValueError(
                "receptor_indices must be supplied to _compute_energies() "
                "when use_interface_model=True."
            )
        B, L, D = v_ligands.shape
        R, k    = receptor_indices.shape

        idx_i = receptor_indices                      # (R, k)
        idx_j = receptor_indices.roll(-1, dims=1)     # (R, k) — next unit (cyclic)

        # Pocket embedding: average of adjacent +/− faces
        # (observation noise is on the ligand side only, matching the classic model)
        v_pocket      = 0.5 * (self.unit_latent_plus[idx_i]
                                + self.unit_latent_minus[idx_j])              # (R, k, D)
        E_base_pocket = 0.5 * (self.base_energy_u_plus[idx_i]
                                + self.base_energy_u_minus[idx_j])            # (R, k)

        # Flatten pockets to (R*k, D), compute (B, L, R*k), then reshape
        b       = v_pocket.reshape(R * k, D)                                   # (R*k, D)
        a_sq    = (v_ligands ** 2).sum(dim=-1, keepdim=True)                   # (B, L, 1)
        b_sq    = (b ** 2).sum(dim=-1)                                         # (R*k,)
        ab      = torch.einsum('bld,kd->blk', v_ligands, b)                   # (B, L, R*k)
        dist_sq = (a_sq + b_sq[None, None, :] - 2.0 * ab).clamp(min=0.0)     # (B, L, R*k)
        dist_sq = dist_sq.view(B, L, R, k)                                    # (B, L, R, k)

        if self.affinity_kernel == "gaussian":
            E_max_pocket = 0.5 * (
                F.softplus(self.max_energy_u_raw_plus[idx_i])
                + F.softplus(self.max_energy_u_raw_minus[idx_j])
            )  # (R, k)
            lambda_sq = self.kernel_params[0] ** 2
            return (E_base_pocket[None, None, :, :]
                    + E_max_pocket[None, None, :, :] * (1.0 - torch.exp(-dist_sq / lambda_sq)))
        else:  # "quadratic"
            E_slope_pocket = 0.5 * (
                F.softplus(self.energy_slope_raw_plus[idx_i])
                + F.softplus(self.energy_slope_raw_minus[idx_j])
            )  # (R, k)
            return (E_base_pocket[None, None, :, :]
                    + E_slope_pocket[None, None, :, :] * dist_sq)

    def sample_batch(
        self,
        batch_size: int,
        receptor_indices: Optional[torch.Tensor] = None,
    ):
        """Generate one batch of sensory environments.

        Args:
            batch_size:       Number of samples B.
            receptor_indices: (R, k_sub) ring layout; required when
                              use_interface_model=True.

        Returns (classic model):
            E_open:        (B, L, U) — open-state energy per unit per ligand.
            concs:         (B, L)    — sampled concentrations (masked).
            mixture_masks: (B, L)    — hierarchical presence mask.

        Returns (interface model):
            E_open:        (B, L, R, k_sub) — pocket energy per interface.
            concs:         (B, L)
            mixture_masks: (B, L)

        The hierarchical sampler guarantees S = sum(M, dim=-1) >= 1 for every
        row by construction; no rejection loop is needed.
        """
        masks = self._sample_masks(batch_size)
        assert masks.sum(-1).min() >= 1, "hierarchical sampler produced an empty row"
        concs     = self.concentration_model.sample(batch_size) * masks
        v_ligands = self._sample_noisy_ligands(batch_size)
        E_open    = self._compute_energies(v_ligands, receptor_indices)
        return E_open, concs, masks

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
