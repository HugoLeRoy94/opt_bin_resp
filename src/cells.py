# Documented in:
#   doc/theory/07_optimization_pipeline.md  (stage 3b: cell readout)
#   doc/theory/02_biophysics_mwc.md         (receptor activation this layer pools over)
"""
cells.py — Cell-level readout layered on top of the receptor array.

Point of view shift: the sensory unit is no longer a receptor but a CELL, defined
by the set of subunit genes G_c it expresses.  A cell assembles EVERY receptor its
genes allow, so the array is specified by C gene sets rather than by R receptors.

Three objects:
  CellArray    — gene sets → deduplicated receptor pool + abundance matrix W (C, R_pool)
  CellReadout  — (B, R_pool) receptor activity → (B, C) cell activity
  cell_activity — chunked physics+readout so peak memory is O(B·L·chunk), not O(B·L·R_pool)

Terminology (see doc/theory/01_nomenclature.md):
  k_sub    number of subunits per receptor (5)
  G_c      set of genes expressed by cell c;  g = |G_c|
  R_pool   size of the deduplicated union of all cells' repertoires
  w_cr     abundance of receptor r in cell c, normalised so sum_r w_cr = 1
  S_bc     drive of cell c on sniff b: the abundance-weighted open fraction
  theta    firing threshold, ONE scalar shared by all cells (learnable)
  T_cell   sharpness of the cell firing step (annealed, as for the receptor step)

Repertoire size grows fast in g: a cell expressing g genes makes C(g+k_sub-1, k_sub)
multiset receptors (g=2 -> 6, g=5 -> 126, g=8 -> 792), or (1/k_sub) sum_{d|k_sub}
phi(d) g^(k_sub/d) ring arrangements under the interface model.  R_pool is the
binding cost of the whole pipeline; see doc/theory/06_computational_limits.md.
"""
import itertools
import math
import random
from math import factorial
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.geometry import _canonical_rotation


# ---------------------------------------------------------------------------
# Repertoire expansion
# ---------------------------------------------------------------------------

def expand_gene_set(genes: Sequence[int], k_sub: int,
                    use_interface_model: bool = False) -> List[Tuple[int, ...]]:
    """Every receptor a cell expressing `genes` can assemble.

    Standard model: unordered multisets of size k_sub drawn from `genes`.
        {0,2}, k_sub=5 -> (0,0,0,0,0) (0,0,0,0,2) (0,0,0,2,2)
                          (0,0,2,2,2) (0,2,2,2,2) (2,2,2,2,2)

    Interface model: canonical cyclic ring arrangements (rotations identified,
    reflections distinct — the +/- face asymmetry breaks mirror symmetry, same
    convention as geometry.generate_ordered_receptor_indices).

    Returns a sorted list, so the pool order is deterministic across runs.
    """
    genes = sorted(set(int(g) for g in genes))
    if not genes:
        raise ValueError("expand_gene_set: a cell must express at least one gene.")

    if not use_interface_model:
        return sorted(itertools.combinations_with_replacement(genes, k_sub))

    forms = {_canonical_rotation(w) for w in itertools.product(genes, repeat=k_sub)}
    return sorted(forms)


def _multiset_multiplicity(receptor: Tuple[int, ...], k_sub: int) -> int:
    """Number of ordered arrangements collapsing onto this multiset: k! / prod(n_i!)."""
    counts: dict = {}
    for u in receptor:
        counts[u] = counts.get(u, 0) + 1
    denom = 1
    for n in counts.values():
        denom *= factorial(n)
    return factorial(k_sub) // denom


def _ring_multiplicity(receptor: Tuple[int, ...]) -> int:
    """Number of distinct rotations of a canonical ring word (= k_sub / |stabiliser|)."""
    k = len(receptor)
    return len({receptor[i:] + receptor[:i] for i in range(k)})


def repertoire_weights(receptors: Sequence[Tuple[int, ...]], n_genes_expressed: int,
                       k_sub: int, stoichiometry: str = "multinomial",
                       use_interface_model: bool = False) -> List[float]:
    """Abundance of each receptor in a cell, normalised to sum to 1.

    stoichiometry='multinomial' (random assembly): all subunits are produced in
      equal amounts and assemble independently into the k_sub slots, so each of
      the g^k_sub ordered words is equally likely and a receptor's abundance is
      proportional to how many words collapse onto it:
        multiset model  w_r ∝ k_sub! / prod_i n_i!     (n_i = copies of gene i)
        interface model w_r ∝ number of distinct rotations of the ring word
      This suppresses homomers (1 word out of k_sub!) relative to mixed receptors.

    stoichiometry='uniform': every receptor type the cell can make is present in
      equal amount; implies per-type assembly control rather than random mixing.
    """
    if stoichiometry == "uniform":
        w = 1.0 / len(receptors)
        return [w] * len(receptors)
    if stoichiometry != "multinomial":
        raise ValueError(f"Unknown stoichiometry {stoichiometry!r}. "
                         f"Choose 'multinomial' or 'uniform'.")

    if use_interface_model:
        mult = [float(_ring_multiplicity(r)) for r in receptors]
    else:
        mult = [float(_multiset_multiplicity(r, k_sub)) for r in receptors]
    total = sum(mult)
    return [m / total for m in mult]


# ---------------------------------------------------------------------------
# Cell array
# ---------------------------------------------------------------------------

class CellArray:
    """A set of cells, expanded into a shared receptor pool plus abundances.

    Attributes:
        gene_sets:        list of C sorted gene tuples.
        receptor_indices: (R_pool, k_sub) long tensor — the DEDUPLICATED union of
                          every cell's repertoire.  Fed unchanged to the existing
                          physics; a receptor shared by several cells is simulated once.
        W:                (C, R_pool) float tensor of abundances, rows summing to 1.
                          W[c, r] = 0 when cell c cannot assemble receptor r.
    """

    def __init__(self, gene_sets: Sequence[Sequence[int]], k_sub: int,
                 stoichiometry: str = "multinomial",
                 use_interface_model: bool = False):
        self.k_sub = k_sub
        self.stoichiometry = stoichiometry
        self.use_interface_model = use_interface_model
        self.gene_sets = [tuple(sorted(set(int(g) for g in gs))) for gs in gene_sets]

        per_cell = [expand_gene_set(gs, k_sub, use_interface_model) for gs in self.gene_sets]

        pool = sorted({r for reps in per_cell for r in reps})
        pos = {r: i for i, r in enumerate(pool)}

        W = torch.zeros(len(self.gene_sets), len(pool), dtype=torch.float32)
        for c, (gs, reps) in enumerate(zip(self.gene_sets, per_cell)):
            weights = repertoire_weights(reps, len(gs), k_sub,
                                         stoichiometry, use_interface_model)
            for r, w in zip(reps, weights):
                W[c, pos[r]] = w

        self.receptor_indices = torch.tensor(pool, dtype=torch.long)
        self.W = W

    @property
    def n_cells(self) -> int:
        return self.W.shape[0]

    @property
    def pool_size(self) -> int:
        return self.receptor_indices.shape[0]

    def to(self, device) -> "CellArray":
        self.receptor_indices = self.receptor_indices.to(device)
        self.W = self.W.to(device)
        return self

    def __repr__(self) -> str:
        sizes = [len(gs) for gs in self.gene_sets]
        return (f"CellArray(C={self.n_cells}, R_pool={self.pool_size}, "
                f"genes/cell={min(sizes)}-{max(sizes)}, "
                f"stoichiometry={self.stoichiometry!r})")


# ---------------------------------------------------------------------------
# Gene-set samplers
# ---------------------------------------------------------------------------

def sample_gene_sets_bernoulli(n_cells: int, n_genes: int,
                               gene_probs: Sequence[float],
                               max_genes: Optional[int] = None,
                               seed: Optional[int] = None) -> List[Tuple[int, ...]]:
    """Each gene is expressed independently with probability gene_probs[u].

    Cells expressing nothing are rejected and redrawn.  max_genes (optional) caps
    the repertoire size per cell — C(g+k_sub-1, k_sub) grows fast in g and R_pool
    is the memory bottleneck.
    """
    rng = random.Random(seed)
    if len(gene_probs) != n_genes:
        raise ValueError(f"gene_probs has length {len(gene_probs)}, expected n_genes={n_genes}.")

    cells: List[Tuple[int, ...]] = []
    while len(cells) < n_cells:
        genes = tuple(u for u in range(n_genes) if rng.random() < gene_probs[u])
        if not genes:
            continue
        if max_genes is not None and len(genes) > max_genes:
            continue
        cells.append(genes)
    return cells


def sample_gene_sets_by_size(n_cells: int, n_genes: int,
                             size_pmf: Sequence[float],
                             seed: Optional[int] = None) -> List[Tuple[int, ...]]:
    """Two-stage sampler: draw the NUMBER of expressed genes, then which ones.

    size_pmf[i] is the (unnormalised) probability that a cell expresses i+1 genes,
    so len(size_pmf) sets the maximum.  This is the hook for an arbitrarily complex
    expression model — a distribution conditioned on how many genes are already
    expressed is exactly a choice of size_pmf.  Given the size, the genes are drawn
    uniformly without replacement.
    """
    rng = random.Random(seed)
    total = float(sum(size_pmf))
    if total <= 0:
        raise ValueError("size_pmf must have positive total mass.")
    probs = [p / total for p in size_pmf]
    sizes = list(range(1, len(probs) + 1))
    if sizes[-1] > n_genes:
        raise ValueError(f"size_pmf allows {sizes[-1]} genes but only {n_genes} exist.")

    cells = []
    for _ in range(n_cells):
        g = rng.choices(sizes, weights=probs, k=1)[0]
        cells.append(tuple(sorted(rng.sample(range(n_genes), g))))
    return cells


def build_cell_array(
    k_sub: int,
    n_genes: int,
    gene_sets: Optional[Sequence[Sequence[int]]] = None,
    n_cells: Optional[int] = None,
    strategy: str = "bernoulli",
    gene_probs: Optional[Sequence[float]] = None,
    size_pmf: Optional[Sequence[float]] = None,
    max_genes: Optional[int] = None,
    seed: Optional[int] = None,
    stoichiometry: str = "multinomial",
    use_interface_model: bool = False,
) -> CellArray:
    """Unified entry point: explicit gene sets, or sample them.

    strategy='bernoulli' uses gene_probs (defaults to uniform 2/n_genes, i.e. two
    expressed genes per cell on average); strategy='size_pmf' uses size_pmf.
    """
    if gene_sets is None:
        if n_cells is None:
            raise ValueError("build_cell_array: provide gene_sets or n_cells.")
        if strategy == "bernoulli":
            if gene_probs is None:
                gene_probs = [2.0 / n_genes] * n_genes
            gene_sets = sample_gene_sets_bernoulli(n_cells, n_genes, gene_probs,
                                                   max_genes=max_genes, seed=seed)
        elif strategy == "size_pmf":
            if size_pmf is None:
                raise ValueError("strategy='size_pmf' requires size_pmf.")
            gene_sets = sample_gene_sets_by_size(n_cells, n_genes, size_pmf, seed=seed)
        else:
            raise ValueError(f"Unknown strategy {strategy!r}. Choose 'bernoulli' or 'size_pmf'.")

    return CellArray(gene_sets, k_sub, stoichiometry=stoichiometry,
                     use_interface_model=use_interface_model)


# ---------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------

class CellReadout(nn.Module):
    """Maps receptor activity (B, R_pool) to cell activity (B, C).

    All three modes reduce over the receptor pool through a term that is LINEAR in
    the per-receptor contribution, so the pool can be traversed in chunks and the
    accumulator summed — that is what keeps memory bounded (see cell_activity).

    mode='threshold' (default):
        S_bc = sum_r w_cr p_br                    abundance-weighted open fraction
        A_bc = sigmoid( (S_bc - theta) / T_cell )     theta shared across cells
      The cell's ionic current is proportional to the number of open receptors it
      carries, and it fires above a threshold.  A_bc is a genuine firing
      probability, which is what the Bernoulli-mixture entropy estimators in
      bin_loss.py assume.

    mode='noisy_or':
        A_bc = 1 - exp( sum_r k_sub*w_cr*log(1 - p_br) )
      Cell active if any of its receptors opens.  No threshold parameter, but
      saturates toward 1 for cells with large repertoires.

    mode='mean':
        A_bc = S_bc.  Simplest; note a mean of probabilities is not a firing
      probability, so the entropy estimators over-read it.  Diagnostic use.

    theta is a SINGLE SCALAR shared by every cell, learnable by default.  Sharing it
    is what lets cell-to-cell heterogeneity be real: with a per-cell threshold every
    cell is forced to the same firing rate, whereas one shared threshold lets a cell
    whose repertoire runs hot fire often and a narrowly-tuned one stay sparse.  It is
    still INITIALISED from the calibrated median (see calibrate_cell_readout) so
    training does not start on a dead gradient.

    temperature (T_cell) is NOT learnable: it is annealed on a schedule by the
    training loop, exactly like BinaryReceptor.temperature.
    """

    _MODES = ("threshold", "noisy_or", "mean")

    def __init__(self, W: torch.Tensor, mode: str = "threshold",
                 threshold: float = 0.5, temperature: float = 0.1,
                 k_sub: int = 5, learnable_threshold: bool = True):
        super().__init__()
        if mode not in self._MODES:
            raise ValueError(f"Unknown cell readout {mode!r}. Choose from {self._MODES}.")
        self.mode = mode
        self.k_sub = k_sub
        self.temperature = temperature
        self.learnable_threshold = learnable_threshold and mode == "threshold"
        # W is a buffer: it moves with .to(device) and is saved in the checkpoint,
        # but carries no gradient — the repertoire composition is fixed, not learned.
        self.register_buffer("W", W)
        theta0 = torch.tensor(float(threshold))
        if self.learnable_threshold:
            self.theta = nn.Parameter(theta0)
        else:
            self.register_buffer("theta", theta0)

    @property
    def n_cells(self) -> int:
        return self.W.shape[0]

    # --- chunkable pieces -------------------------------------------------

    def accumulate(self, p_chunk: torch.Tensor, r0: int, r1: int) -> torch.Tensor:
        """Contribution of receptor-pool slice [r0:r1] to the (B, C) accumulator."""
        W_chunk = self.W[:, r0:r1]                                  # (C, m)
        if self.mode == "noisy_or":
            log1m = torch.log1p(-p_chunk.clamp(max=1.0 - 1e-7))     # (B, m)
            return self.k_sub * (log1m @ W_chunk.T)                 # (B, C)
        return p_chunk @ W_chunk.T                                  # (B, C)

    def finalize(self, acc: torch.Tensor) -> torch.Tensor:
        """Accumulator (B, C) → cell activity (B, C) in [0, 1]."""
        if self.mode == "noisy_or":
            return 1.0 - torch.exp(acc)
        if self.mode == "mean":
            return acc.clamp(0.0, 1.0)
        return torch.sigmoid((acc - self.theta) / self.temperature)   # theta is scalar

    def drive(self, activity: torch.Tensor) -> torch.Tensor:
        """The pre-threshold accumulator S (B, C) — used for calibration."""
        return self.accumulate(activity, 0, self.W.shape[1])

    def forward(self, activity: torch.Tensor) -> torch.Tensor:
        return self.finalize(self.drive(activity))


# ---------------------------------------------------------------------------
# Chunked physics + readout
# ---------------------------------------------------------------------------

def cell_activity(physics, readout: CellReadout,
                  energies: torch.Tensor, concentrations: torch.Tensor,
                  receptor_indices: torch.Tensor,
                  pre_gathered: bool = False,
                  chunk_size: Optional[int] = None,
                  recompute: bool = False,
                  composition: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Receptor physics + cell pooling, traversing the receptor pool in chunks.

    The physics tensors are (B, L, R_pool[, k_sub]) — R_pool is the union of every
    cell's repertoire and is the memory bottleneck of the cell picture.  Because
    CellReadout.accumulate is linear in the pool axis, the pool can be walked in
    slices of `chunk_size` receptors and the (B, C) accumulator summed, so the
    peak forward tensor is O(B·L·chunk) instead of O(B·L·R_pool).

    chunk_size=None runs the pool in one pass (identical result, matching the
    receptor-array behaviour).

    recompute=True gradient-checkpoints each chunk so its activations are
    recomputed in backward rather than retained: without it the retained graph is
    still O(B·L·R_pool) during training and chunking only helps under no_grad.
    Exact — no effect on values or gradients (same convention as
    DiscreteExactLoss.recompute_backward).
    """
    R_pool = receptor_indices.shape[0]
    if chunk_size is None or chunk_size >= R_pool:
        return readout(physics(energies, concentrations, receptor_indices,
                               pre_gathered=pre_gathered, composition=composition))

    def _chunk(r0: int, r1: int) -> torch.Tensor:
        if composition is not None:
            # Slice the composition COLUMNS (one per receptor); the per-source energy
            # tensor is shared by every chunk and must not be sliced.
            p = physics(energies, concentrations, receptor_indices[r0:r1],
                        composition=composition[:, r0:r1])
        else:
            e = energies[:, :, r0:r1] if pre_gathered else energies
            p = physics(e, concentrations, receptor_indices[r0:r1], pre_gathered=pre_gathered)
        return readout.accumulate(p, r0, r1)

    acc = None
    do_ckpt = recompute and torch.is_grad_enabled() and energies.requires_grad
    for r0 in range(0, R_pool, chunk_size):
        r1 = min(r0 + chunk_size, R_pool)
        if do_ckpt:
            # checkpoint needs tensor args to track; r0/r1 ride along as ints.
            part = checkpoint(lambda a, b: _chunk(int(a), int(b)),
                              torch.tensor(r0), torch.tensor(r1), use_reentrant=False)
        else:
            part = _chunk(r0, r1)
        acc = part if acc is None else acc + part

    return readout.finalize(acc)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_cell_readout(readout: CellReadout, env, physics,
                           receptor_indices: torch.Tensor,
                           calibration_batch_size: int = 2048,
                           chunk_size: Optional[int] = None,
                           set_threshold: bool = True,
                           set_temperature: bool = True) -> dict:
    """Set theta and T_cell from the empirical distribution of the drive S.

    Same rationale as physics.compute_initial_temperature: a threshold placed off
    the support of S leaves every cell permanently silent (or saturated) and the
    array carries zero entropy, while a temperature far from the spread of S gives
    either a hard step with vanishing gradient or a mushy all-0.5 response.

      theta   <- median over ALL (b, c) of S      the population fires ~50% of the time
      T_cell  <- std over (b, c) of (S - theta)   the sigmoid argument starts unit-spread

    theta is a single scalar shared by every cell, so the median is taken over the
    pooled drive rather than per cell.  This is deliberate (see CellReadout): a shared
    threshold lets cells differ in how often they fire.  The cost is that a cell whose
    drive never reaches theta is silent — hence the 'firing_fraction' diagnostic in the
    returned dict.  A few silent cells at init is fine (theta is learnable and the
    environment adapts); MOST cells silent means the repertoires are too heterogeneous
    for one threshold, and the run will waste channels.

    No-op for 'noisy_or' and 'mean', which have neither parameter.

    Returns a dict of diagnostics: theta, T_cell, firing_fraction (per cell, at the
    calibrated theta), n_silent / n_saturated (cells outside [1%, 99%] firing).
    """
    if readout.mode != "threshold":
        return {"theta": float(readout.theta), "T_cell": readout.temperature}

    ri = receptor_indices if getattr(env, "use_interface_model", False) else None
    E, concs, _ = env.sample_batch(calibration_batch_size, receptor_indices=ri)
    comp = getattr(env, "composition", None)
    pre_g = getattr(env, "use_interface_model", False) and comp is None

    R_pool = receptor_indices.shape[0]
    step = chunk_size if (chunk_size is not None and chunk_size < R_pool) else R_pool
    acc = None
    for r0 in range(0, R_pool, step):
        r1 = min(r0 + step, R_pool)
        if comp is not None:
            p = physics(E, concs, receptor_indices[r0:r1], composition=comp[:, r0:r1])
        else:
            e = E[:, :, r0:r1] if pre_g else E
            p = physics(e, concs, receptor_indices[r0:r1], pre_gathered=pre_g)
        part = readout.accumulate(p, r0, r1)
        acc = part if acc is None else acc + part                   # (B, C)

    if set_threshold:
        readout.theta.data.fill_(acc.median().item())      # pooled over (b, c)
    if set_temperature:
        spread = (acc - readout.theta).std().item()
        readout.temperature = max(spread, 1e-3)

    firing = (acc > readout.theta).float().mean(dim=0)     # (C,) fraction of sniffs
    return {
        "theta":            readout.theta.detach().item(),
        "T_cell":           readout.temperature,
        # Scale of the drive. The ONLY meaningful reference for T_cell: the drive is a
        # weighted mean of probabilities, so its spread is set by the environment, not
        # by any fixed number. A T_cell comparable to drive_std leaves cells soft (their
        # response is mostly conditional entropy, which carries no information); T_cell
        # << drive_std makes them near-deterministic, which is what the entropy
        # objective needs to be a valid proxy for information.
        "drive_std":        (acc - acc.median()).std().item(),
        "firing_fraction":  firing,
        "n_silent":         int((firing < 0.01).sum()),
        "n_saturated":      int((firing > 0.99).sum()),
    }


if __name__ == "__main__":
    from src.physics import BinaryReceptor

    # 1) repertoire of a cell expressing genes {0, 2}, k_sub=5.
    reps = expand_gene_set([0, 2], 5)
    assert reps == [(0,0,0,0,0), (0,0,0,0,2), (0,0,0,2,2),
                    (0,0,2,2,2), (0,2,2,2,2), (2,2,2,2,2)], reps
    w_mn = repertoire_weights(reps, 2, 5, "multinomial")
    assert [round(x * 32) for x in w_mn] == [1, 5, 10, 10, 5, 1], w_mn
    assert abs(sum(w_mn) - 1) < 1e-9
    print(f"[1] g=2 repertoire: {len(reps)} receptors, multinomial ∝ 1,5,10,10,5,1")

    # 2) pool deduplication: (0,0,0,0,0) is shared by cells {0,2} and {0,1}.
    ca = CellArray([[0, 2], [0, 1], [3]], k_sub=5)
    assert ca.pool_size == 12, ca.pool_size          # 6 + 6 + 1 - 1 shared homomer
    assert torch.allclose(ca.W.sum(1), torch.ones(3))
    shared = ca.receptor_indices.tolist().index([0, 0, 0, 0, 0])
    assert ca.W[0, shared] > 0 and ca.W[1, shared] > 0 and ca.W[2, shared] == 0
    print(f"[2] {ca}  (shared homomer simulated once)")

    # 3) interface model: Burnside count for the cyclic group C_5.
    ca_i = CellArray([[0, 2]], k_sub=5, use_interface_model=True)
    assert ca_i.pool_size == (2 ** 5 + 4 * 2) // 5 == 8, ca_i.pool_size
    assert abs(ca_i.W.sum().item() - 1) < 1e-6
    print(f"[3] interface repertoire of g=2: {ca_i.pool_size} ring arrangements")

    # 4) chunking the receptor pool is exact for every readout mode.
    torch.manual_seed(0)
    ca4 = CellArray([[0, 2], [0, 1], [1, 2, 3], [4]], k_sub=5)
    B, L, U = 64, 3, 6
    E = torch.randn(B, L, U).abs()
    c = torch.rand(B, L) * 10
    phys = BinaryReceptor(U, 5, temperature=0.5)
    for mode in CellReadout._MODES:
        ro = CellReadout(ca4.W, mode=mode, temperature=0.2, k_sub=5)
        full = cell_activity(phys, ro, E, c, ca4.receptor_indices, chunk_size=None)
        chunked = cell_activity(phys, ro, E, c, ca4.receptor_indices, chunk_size=3)
        assert full.shape == (B, ca4.n_cells), full.shape
        assert torch.allclose(full, chunked, atol=1e-6), (mode, (full - chunked).abs().max())
        assert (full >= 0).all() and (full <= 1).all()
        print(f"[4] {mode:9s}: chunked == unchunked, activity in [0,1]")

    # 5) gradients reach the environment through the chunked/checkpointed readout.
    E5 = E.clone().requires_grad_(True)
    ro = CellReadout(ca4.W, mode="threshold", temperature=0.2, k_sub=5)
    cell_activity(phys, ro, E5, c, ca4.receptor_indices,
                  chunk_size=3, recompute=True).sum().backward()
    assert E5.grad is not None and E5.grad.abs().sum() > 0
    print("[5] gradient flows through chunked + gradient-checkpointed readout")

    # 6) samplers honour their constraints.
    ca6 = build_cell_array(k_sub=5, n_genes=8, n_cells=20, strategy="bernoulli",
                           gene_probs=[0.3] * 8, max_genes=3, seed=1)
    assert max(len(g) for g in ca6.gene_sets) <= 3
    ca7 = build_cell_array(k_sub=5, n_genes=8, n_cells=20, strategy="size_pmf",
                           size_pmf=[0.4, 0.4, 0.2], seed=1)
    assert sorted({len(g) for g in ca7.gene_sets}) == [1, 2, 3]
    assert build_cell_array(k_sub=5, n_genes=8, n_cells=20, strategy="size_pmf",
                            size_pmf=[0.4, 0.4, 0.2], seed=1).gene_sets == ca7.gene_sets
    print(f"[6] samplers: bernoulli(max_genes=3) {ca6}\n              size_pmf {ca7} (seeded, reproducible)")

    print("\nall cells.py self-tests passed.")
