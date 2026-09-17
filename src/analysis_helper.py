# Documented in:
#   doc/theory/07_optimization_pipeline.md  (stage 6: evaluation metrics table)
"""
analysis_helper.py — Evaluation metrics and visualisation utilities.

All metric functions are registered in run.py::MEASUREMENT_REGISTRY and called
via signature introspection, accepting whichever of (env, physics, receptor_indices,
loss_fn, activity, concs, mixture_masks, family_labels, epoch) they need.

Key metrics:
  full_array_entropy      — collision H2 + blocked Shannon of soft joint activity
  codeword_entropy        — hard plug-in entropy + Miller-Madow bias correction
  mean_specialization_index — S_r = (A_max − A_bg) / (A_max + A_bg)
  rank_ordered_distances  — average energy gap from preferred ligand (rank-ordered)

Miller-Madow correction: H_MM = H_plugin + (K_hat − 1) / (2·B·ln2) partially corrects
the downward bias of the plug-in estimator for finite batch sizes.
"""
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns

from src.environment import LogNormalConcentration # Adjust import path as needed
from src.bin_loss import (compute_shannon_joint_entropy, compute_collision_entropy,
                          compute_blocked_entropy, compute_kt_entropy,
                          compute_kt_upper_entropy, compute_response_conditional_entropy,
                          KT_EPS)


@torch.no_grad()
def plot_ligand_summary(env, physics, receptor_indices, n_points=200, axes=None):
    """
    Creates a comprehensive summary plot for each ligand adapted to the discrete model:
    1. Dose-response curves as step functions (Main Frame)
    2. Concentration Distribution (Bottom Frame)
    3. Discrete Binary Assignment / Marginal Probabilities (Right Frame)
    """
    device = next(env.parameters()).device
    N_Receptors = receptor_indices.shape[0]
    n_ligands = env.n_ligands

    # Generate a color palette for the receptors
    colors = plt.cm.viridis(np.linspace(0, 0.9, N_Receptors))

    if axes is None:
        fig = plt.figure(figsize=(8, 6))
        gs = GridSpec(4, 4, figure=fig, hspace=0.1, wspace=0.1)
        ax_main = fig.add_subplot(gs[0:3, 0:3])
        ax_bottom = fig.add_subplot(gs[3, 0:3], sharex=ax_main)
        ax_right = fig.add_subplot(gs[0:3, 3], sharey=ax_main)
    else:
        ax_main, ax_bottom, ax_right = axes
        fig = ax_main.figure

    for l_idx in range(n_ligands):
        
        # =====================================================================
        # 1. DATA PREPARATION
        # =====================================================================
        
        # A. Bottom Frame: Exact Concentration PDF
        c_sweep_tensor, c_pdf_tensor = env.get_concentration_sweep(l_idx, n_points=n_points)
        c_sweep_np = c_sweep_tensor.cpu().numpy()
        c_pdf_np = c_pdf_tensor.cpu().numpy()
        
        # Normalize the PDF to act as discrete weights for our exact probability calculation
        c_weights = c_pdf_np / (np.sum(c_pdf_np) + 1e-12)
        
        # B. Main Frame: Dose Response (Sharp Sigmoid / Steps)
        # We don't need method='self_normalized' anymore because it's a true probability [0, 1]
        _, p_o_np = physics.get_dose_response(env, receptor_indices, l_idx, n_points=n_points, method='absolute')
        
        # =====================================================================
        # 2. PLOTTING
        # =====================================================================
        
        # --- Main Frame (Dose Response Steps) ---
        for r in range(N_Receptors):
            # Using standard plot; because of low temperature, it naturally forms a sharp step
            ax_main.plot(c_sweep_np, p_o_np[:, r], color=colors[r], lw=2, label=f"R {r}")
            
        ax_main.set_ylabel("Activity Probability $p(a=1)$", fontsize=9)
        ax_main.tick_params(labelbottom=False, direction='in') 
        ax_main.set_title(f"Receptor Array Binary Response: Ligand {l_idx}", fontsize=10, fontweight='bold')
        ax_main.set_ylim(-0.05, 1.05)
        # If your concentration spans orders of magnitude, uncomment the next line!
        ax_main.set_xscale('log')

        # --- Bottom Frame (Concentration Distribution) ---
        ax_bottom.fill_between(c_sweep_np, c_pdf_np, color='gray', alpha=0.4)
        ax_bottom.plot(c_sweep_np, c_pdf_np, color='black', lw=1)
        ax_bottom.set_xlabel("Concentration (M)", fontsize=9)
        ax_bottom.set_ylabel("p(c)", fontsize=9)
        ax_bottom.set_yticks([]) 
        ax_bottom.tick_params(direction='in')
        
        # --- Right Frame (Discrete Binary Assignment) ---
        # Calculate the exact expected marginal probability of firing for each receptor
        # P(a=1) = Sum over all concentrations of: P(fire | c) * P(c)
        p_active = np.sum(p_o_np * c_weights[:, None], axis=0) # Shape: (N_Receptors,)
        p_inactive = 1.0 - p_active
        
        # Plot Grouped Horizontal Bars at y=0.0 (Inactive) and y=1.0 (Active)
        bar_height = 0.6 / N_Receptors # Dynamic scaling so bars don't overlap
        
        for r in range(N_Receptors):
            # Offset each receptor slightly so they stack neatly
            y_offset = (r - N_Receptors/2) * bar_height
            
            # Bar for Inactive Bin (y = 0.0)
            ax_right.barh(0.0 + y_offset, p_inactive[r], height=bar_height, color=colors[r], alpha=0.8)
            # Bar for Active Bin (y = 1.0)
            ax_right.barh(1.0 + y_offset, p_active[r], height=bar_height, color=colors[r], alpha=0.8)
            
        ax_right.set_xlabel("Mass", fontsize=9)
        ax_right.tick_params(labelleft=False, direction='in') 
        
        # Draw dotted lines at exactly 0.0 and 1.0 to guide the eye
        ax_right.axhline(0.0, color='black', linestyle='--', linewidth=0.5, zorder=0)
        ax_right.axhline(1.0, color='black', linestyle='--', linewidth=0.5, zorder=0)
        
    axes = (ax_main, ax_bottom, ax_right)
        
    return fig, axes

@torch.no_grad()
def plot_summary(env, physics, receptor_indices, loss_fn=None, n_points=200, axes=None):
    """
    Creates a SINGLE comprehensive summary plot for all ligands.
    """
    device = next(env.parameters()).device
    N_Receptors = receptor_indices.shape[0]
    n_ligands = env.n_ligands

    colors = plt.cm.viridis(np.linspace(0, 0.9, N_Receptors))
    
    if axes is None:
        fig = plt.figure(figsize=(4, 3))
        gs = GridSpec(4, 4, figure=fig, hspace=0.1, wspace=0.1)
        ax_main = fig.add_subplot(gs[0:3, 0:3])
        ax_bottom = fig.add_subplot(gs[3, 0:3], sharex=ax_main)
        ax_right = fig.add_subplot(gs[0:3, 3], sharey=ax_main)
    elif not isinstance(axes, (list, tuple, np.ndarray)):
        # If a single axis is passed, create the gridspec inside it
        ax = axes
        fig = ax.figure
        spec = ax.get_subplotspec()
        ax.set_visible(False)

        gs = GridSpecFromSubplotSpec(4, 4, subplot_spec=spec, hspace=0.1, wspace=0.1)
        ax_main = fig.add_subplot(gs[0:3, 0:3])
        ax_bottom = fig.add_subplot(gs[3, 0:3], sharex=ax_main)
        ax_right = fig.add_subplot(gs[0:3, 3], sharey=ax_main)
    else:
        ax_main, ax_bottom, ax_right = axes
        fig = ax_main.figure
    
    if isinstance(env.concentration_model, LogNormalConcentration):
        ax_main.set_xscale('log')
    
    global_p_active = np.zeros(N_Receptors)

    for l_idx in range(n_ligands):
        c_sweep_tensor, c_pdf_tensor = env.get_concentration_sweep(l_idx, n_points=n_points)
        c_sweep_np = c_sweep_tensor.cpu().numpy()
        c_pdf_np = c_pdf_tensor.cpu().numpy()
        c_weights = c_pdf_np / (np.sum(c_pdf_np) + 1e-12)
        
        _, p_o_np = physics.get_dose_response(env, receptor_indices, l_idx, n_points=n_points, method='absolute')

        p_plot_active = p_o_np
        
        for r in range(N_Receptors):
            ax_main.plot(c_sweep_np, p_plot_active[:, r], color=colors[r], lw=2.5, alpha=1.)
            
        ax_bottom.fill_between(c_sweep_np, c_pdf_np, color='gray', alpha=0.15)
        ax_bottom.plot(c_sweep_np, c_pdf_np, color='black', lw=1., alpha=1.)
        
        ligand_p_active = np.sum(p_plot_active * c_weights[:, None], axis=0)
        global_p_active += (ligand_p_active / n_ligands)
        
    global_p_inactive = 1.0 - global_p_active

    ax_main.set_ylabel("Activity Probability $p(a=1)$", fontsize=9)
    ax_main.tick_params(labelbottom=False, direction='in') 
    ax_main.set_title("Global Receptor Array Binary Response", fontsize=9, fontweight='bold')
    
    ax_bottom.set_xlabel("Concentration (M)", fontsize=9)
    ax_bottom.set_ylabel("p(c)", fontsize=9)
    ax_bottom.set_yticks([]) 
    ax_bottom.tick_params(direction='in')
    
    bar_height = 0.2 / N_Receptors 
    padding = (N_Receptors / 2) * bar_height * 1.3
    ax_main.set_ylim(-padding, 1.0 + padding)
    
    for r in range(N_Receptors):
        r_rev = (N_Receptors-1)-r
        y_offset = (r - N_Receptors/2) * bar_height
        ax_right.barh(0.0 + y_offset, global_p_inactive[r_rev], height=bar_height, edgecolor=colors[r_rev], alpha=0.8, facecolor='none', linewidth=2.5)
        ax_right.barh(1.0 + y_offset, global_p_active[r_rev], height=bar_height, edgecolor=colors[r_rev], alpha=0.8, facecolor='none', linewidth=2.5)
        
    ax_right.set_xlabel("Global Mass", fontsize=9)
    ax_right.tick_params(labelleft=False, direction='in') 
    ax_right.axhline(0.0, color='black', linestyle='--', linewidth=0.5, zorder=0)
    ax_right.axhline(1.0, color='black', linestyle='--', linewidth=0.5, zorder=0)
    
    return fig, (ax_main, ax_bottom, ax_right)

@torch.no_grad()
def evaluate_model(env,physics,receptor_indices,loss_fn,n_samples=2000):
    device = env.interaction_mu.device
    N_Receptors = receptor_indices.shape[0]
    # draw random ligands
    energies,concs,mixture_masks = env.sample_batch(batch_size = n_samples)
    # compute the activity array
    activity = physics(energies, concs, receptor_indices)

    soft_assign = loss_fn.compute_soft_assignment(activity)
    entropy_fn = compute_collision_entropy if getattr(loss_fn, 'entropy_type', 'shannon') == 'collision' else compute_shannon_joint_entropy
    val = entropy_fn(soft_assign)
    return val.item() if isinstance(val, torch.Tensor) else val


# ── Post-hoc measurement from a reloaded checkpoint ──────────────────────────
# Drive a FROZEN (env, physics) — e.g. reconstructed via plotlib.load_model — to
# produce fresh samples and measure entropy WITHOUT retraining. Reusable by any
# post-hoc analysis (test-size scaling, extra estimators, resampling, …): the sampler
# is estimator-agnostic, so a script picks whatever estimator it wants on the result.
EVAL_TILE = 2048     # KT internal tile (also the eval memory-cap denominator)
FWD_CHUNK = 16384    # forward sub-batch (no_grad) — bounds the sampling forward pass


@torch.no_grad()
def sample_activity(env, physics, receptor_indices, n_samples, fwd_chunk=FWD_CHUNK,
                    readout=None, pool_chunk=None):
    """Generate `n_samples` of receptor activity from a (reloaded) env + physics.

    Samples are drawn in `fwd_chunk` sub-batches and concatenated, so the forward-pass
    memory stays bounded regardless of `n_samples`. Interface-model-aware. Returns a
    (n_samples, R) tensor; apply any estimator to it (KT, collision, codeword, …).

    readout: a CellReadout (src/cells.py) reloaded from the checkpoint. When given,
    the receptor pool is pooled into cells and the result is (n_samples, C) instead;
    every downstream estimator is agnostic to which of the two it receives.
    pool_chunk bounds the receptor-pool axis per forward pass (None → one pass).
    """
    ri_fwd = receptor_indices if env.use_interface_model else None
    acts, n = [], 0
    while n < n_samples:
        b = min(fwd_chunk, n_samples - n)
        E, concs, _ = env.sample_batch(b, receptor_indices=ri_fwd)
        if readout is None:
            acts.append(physics(E, concs, receptor_indices,
                                pre_gathered=env.use_interface_model))
        else:
            from src.cells import cell_activity
            acts.append(cell_activity(physics, readout, E, concs, receptor_indices,
                                      pre_gathered=env.use_interface_model,
                                      chunk_size=pool_chunk))
        n += b
    return torch.cat(acts, dim=0)


@torch.no_grad()
def count_sampled_responses(env, physics, receptor_indices, n_samples,
                            fwd_chunk=FWD_CHUNK, readout=None, pool_chunk=None,
                            response_generator=None):
    """Estimate response entropy and MI from streamed stochastic binary outputs.

    Fresh inputs are forwarded in bounded chunks.  For each input, one conditionally
    independent Bernoulli response vector is sampled and moved to CPU; the analytic
    conditional entropy is accumulated with the corresponding sample weight.  Thus
    this avoids both retaining a ``(n_samples, R)`` probability tensor and KT's
    quadratic pairwise calculation.  The returned keys are those of
    :func:`response_counting_metrics`.

    ``response_generator`` may be supplied to isolate response-sampling randomness
    from the environment's input-sampling RNG.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    ri_fwd = receptor_indices if env.use_interface_model else None
    codes, n_done = [], 0
    h_cond_sum = None
    while n_done < n_samples:
        b = min(fwd_chunk, n_samples - n_done)
        E, concs, _ = env.sample_batch(b, receptor_indices=ri_fwd)
        if readout is None:
            activity = physics(E, concs, receptor_indices,
                               pre_gathered=env.use_interface_model)
        else:
            from src.cells import cell_activity
            activity = cell_activity(physics, readout, E, concs, receptor_indices,
                                     pre_gathered=env.use_interface_model,
                                     chunk_size=pool_chunk)
        activity = activity.clamp(KT_EPS, 1.0 - KT_EPS)
        h_cond_chunk = compute_response_conditional_entropy(activity).double() * b
        h_cond_sum = h_cond_chunk if h_cond_sum is None else h_cond_sum + h_cond_chunk
        codes.append(torch.bernoulli(activity, generator=response_generator).bool().cpu())
        n_done += b

    return response_counting_metrics(torch.cat(codes), (h_cond_sum / n_done).item())


@torch.no_grad()
def kt_bracket(env, physics, receptor_indices, n_samples, tile=EVAL_TILE, fwd_chunk=FWD_CHUNK,
               readout=None, pool_chunk=None):
    """(kt_lower, kt_upper) in bits on `n_samples` fresh samples from a reloaded model.

    Both are certified bounds on the joint entropy H(s). KT tiles internally at `tile`,
    so peak memory is bounded by the tile, not `n_samples` (time is O(n_samples²))."""
    activity = sample_activity(env, physics, receptor_indices, n_samples, fwd_chunk,
                               readout=readout, pool_chunk=pool_chunk)
    soft = torch.stack([1.0 - activity, activity], dim=-1)
    return (compute_kt_entropy(soft, chunk_size=tile).item(),
            compute_kt_upper_entropy(soft, chunk_size=tile).item())


def eval_batch_cap(mem_free_bytes, tile=EVAL_TILE):
    """Largest test batch whose (tile, B) KT row buffer fits ~80% of free GPU memory."""
    return max(tile, int(mem_free_bytes * 0.8) // (tile * 4 * 4))


@torch.no_grad()
def plot_latent_radar_chart(env, receptor_indices, receptors_to_plot=None, family_names=None, ax=None):
    """
    Creates a radar chart showing the relative binding strength of 
    fully assembled receptors (heteromers) across all families.
    """
    n_families = env.n_families
    
    # 1. Compute exact Unit-Family Interaction Energies: (n_genes, n_families)
    diff = env.unit_latent.unsqueeze(1) - env.family_latent.unsqueeze(0)  # (U, F, D)
    dist_sq = (diff ** 2).sum(dim=-1)  # (U, F)
    if env.affinity_kernel == "gaussian":
        max_e = torch.nn.functional.softplus(env.max_energy_u_raw)  # (U,)
        lambda_sq = env.kernel_params[0] ** 2
        unit_family_energies = (env.base_energy_u.unsqueeze(1)
                                + max_e.unsqueeze(1) * (1.0 - torch.exp(-dist_sq / lambda_sq))).cpu()
    else:  # "quadratic"
        dE = torch.nn.functional.softplus(env.energy_slope_raw)  # (U,)
        unit_family_energies = (env.base_energy_u.unsqueeze(1) + dE.unsqueeze(1) * dist_sq).cpu()
    
    # 2. Compute Receptor Energies
    # Indexing yields (N_Receptors, k_sub, n_families)
    # Mean across k_sub yields (N_Receptors, n_families)
    receptor_energies = unit_family_energies[receptor_indices].mean(dim=1).numpy()
    
    # Select which specific receptors to plot (default to first 5 if not specified to avoid clutter)
    if receptors_to_plot is None:
        receptors_to_plot = list(range(min(5, len(receptor_indices))))
        
    selected_energies = receptor_energies[receptors_to_plot]
    
    # 3. Convert Energy to Affinity Score
    #max_energy = np.max(receptor_energies)
    #affinity_scores = max_energy - selected_energies 
    # Normalize to [0, 1] so radar charts are comparable across different runs/epochs
    max_energy = np.max(receptor_energies)
    min_energy = np.min(receptor_energies)
    
    # Add a tiny epsilon to the denominator to prevent division by zero
    affinity_scores = 1 - (max_energy - selected_energies) / (max_energy - min_energy + 1e-8)

    # 4. Setup Radar Chart Angles
    if family_names is None:
        family_names = [f"Fam {i}" for i in range(n_families)]
        
    angles = np.linspace(0, 2 * np.pi, n_families, endpoint=False).tolist()
    angles += angles[:1] # Close the loop
    
    # 5. Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4), subplot_kw=dict(polar=True))
    else:
        fig = ax.figure
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(family_names, fontsize=10)
    #ax.set_yticks([]) 
    
    # Optional: explicitly set the radial bounds to [0, 1] for the normalized scores
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([]) # Keep it clean by hiding the numeric labels

    colors = plt.cm.tab10.colors 
    
    # Plot each Receptor's polygon
    for i, r_idx in enumerate(receptors_to_plot):
        values = affinity_scores[i].tolist()
        values += values[:1] 
        
        c = colors[i % len(colors)]
        ax.plot(angles, values, linewidth=2, color=c, label=f"Receptor {r_idx}")
        ax.fill(angles, values, color=c, alpha=0.15)
        
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    #plt.title("Assembled Receptor Affinity Profile", y=1.08, fontweight='bold')
    if ax is None:
        plt.tight_layout()
    
    return fig, ax

# Note: if UniformNBall is in environment.py, import it:
# from core.environment import UniformNBall

@torch.no_grad()
def build_latent_umap(env, receptor_indices, n_samples_per_family=1000, random_state=42):
    """
    Projects the N-dimensional chemical latent space into 2D using UMAP.
    Returns projected families, ligands, receptor centroids and family samples.
    Reuse this result for all panels of the same model. Sampling and UMAP use
    private seeded RNGs; they do not change the simulation's torch RNG state.

    Args:
        env: Instantiated LigandEnvironment.
        receptor_indices: Tensor of shape (N_Receptors, k_sub) mapping units to receptors.
        n_samples_per_family: How many points to sample per family to generate the gradient.
    """
    import umap  # lazy: only needed here, keeps src importable without umap-learn
    n_families = env.n_families
    n_ligands = env.n_ligands
    n_receptors = receptor_indices.shape[0]

    # =====================================================================
    # 1. EXTRACT CENTERS AND ASSEMBLED RECEPTORS
    # =====================================================================
    v_families = env.family_latent.detach().cpu().numpy()
    v_ligands = env.ligand_latent.detach().cpu().numpy()
    ligand_assignments = env.ligand_family_assignments.detach().cpu().numpy()

    # Receptor centroids: pocket midpoints for interface model, unit embeddings otherwise
    if getattr(env, 'use_interface_model', False):
        idx_i = receptor_indices
        idx_j = receptor_indices.roll(-1, dims=1)
        v_pocket = 0.5 * (env.unit_latent_plus[idx_i] + env.unit_latent_minus[idx_j])
        v_receptors = v_pocket.mean(dim=1).detach().cpu().numpy()
    else:
        v_receptors = env.unit_latent[receptor_indices].mean(dim=1).detach().cpu().numpy()
    
    # =====================================================================
    # 2. SAMPLE THE LIGAND REGIONS (To generate the gradient)
    # =====================================================================
    if n_samples_per_family < 2:
        raise ValueError("n_samples_per_family must be at least 2")
    rng = np.random.default_rng(random_state)
    shape = (n_families, n_samples_per_family, env.latent_dim)
    if env.distribution_type == 'gaussian':
        offsets = rng.normal(size=shape) * env.family_spread
    elif env.distribution_type == 'uniform_cube':
        offsets = rng.uniform(-env.family_spread, env.family_spread, size=shape)
    elif env.distribution_type in ('uniform', 'shell'):
        direction = rng.normal(size=shape)
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        radius = rng.uniform(size=(*shape[:2], 1))
        if env.distribution_type == 'uniform':
            radius = radius ** (1.0 / env.latent_dim)
        offsets = direction * radius * env.family_spread
    else:
        raise ValueError(f"Unsupported family distribution: {env.distribution_type}")
    sampled_points = (v_families[:, None, :] + offsets).reshape(-1, env.latent_dim)
    sampled_labels = np.repeat(np.arange(n_families), n_samples_per_family)
    
    # =====================================================================
    # 3. FIT UMAP PROJECTION
    # =====================================================================
    all_data = np.vstack([v_families, v_ligands, v_receptors, sampled_points])
    
    print("Fitting UMAP... (This may take a few seconds)")
    reducer = umap.UMAP(n_neighbors=min(15, len(all_data) - 1), min_dist=0.1,
                       metric='euclidean', random_state=random_state, n_jobs=1)
    embedding = reducer.fit_transform(all_data)
    
    # Unpack the embeddings
    emb_families = embedding[:n_families]
    emb_ligands = embedding[n_families : n_families + n_ligands]
    emb_receptors = embedding[n_families + n_ligands : n_families + n_ligands + n_receptors]
    emb_samples = embedding[n_families + n_ligands + n_receptors :]

    return dict(families=emb_families, ligands=emb_ligands, receptors=emb_receptors,
                samples=emb_samples, sample_labels=sampled_labels,
                ligand_assignments=ligand_assignments, latent_dim=env.latent_dim)


def plot_latent_umap(env, receptor_indices, n_samples_per_family=1000, random_state=42,
                     ax=None, *, embedding=None):
    """Plot the chemical landscape; optionally reuse a build_latent_umap result.

    Existing calls and the (fig, ax) return value are preserved. Supplied embeddings
    must describe the same environment and displayed receptor list.
    """
    if embedding is None:
        embedding = build_latent_umap(env, receptor_indices, n_samples_per_family, random_state)
    return _draw_latent_umap(embedding, ax=ax)


def _draw_latent_umap(embedding, ax=None, ligand_values=None, show_legend=True):
    """Shared drawing layer for a landscape and individual cell response panels."""
    emb_families, emb_ligands = embedding['families'], embedding['ligands']
    emb_receptors, emb_samples = embedding['receptors'], embedding['samples']
    sampled_labels = embedding['sample_labels']
    ligand_assignments = embedding['ligand_assignments']
    n_families, n_receptors = len(emb_families), len(emb_receptors)
    
    # =====================================================================
    # 4. PLOTTING
    # =====================================================================
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure
    
    # Generate a distinct color palette
    colors = plt.cm.tab10.colors if n_families <= 10 else plt.cm.viridis(np.linspace(0, 1, n_families))
    
    # Plot the Density Gradients (Regions)
    for f_idx in range(n_families):
        pts = emb_samples[sampled_labels == f_idx]
        c = colors[f_idx % len(colors)]
        
        # Seaborn KDE creates the beautiful topographical contour gradients
        sns.kdeplot(
            x=pts[:, 0], y=pts[:, 1], 
            ax=ax, fill=True, color=c, alpha=0.3 if ligand_values is None else 0.10,
            levels=5, thresh=0.05
        )
        
        # Plot the exact family center
        ax.scatter(
            emb_families[f_idx, 0], emb_families[f_idx, 1], 
            marker='o', s=100//n_families, color=c, edgecolor='black', linewidth=1.2,
            zorder=4, label=f'Fam {f_idx}' if f_idx < 10 else ""
        )
        
        # Plot ligands assigned to this family
        if ligand_values is None:
            ligand_pts = emb_ligands[ligand_assignments == f_idx]
            ax.scatter(
                ligand_pts[:, 0], ligand_pts[:, 1],
                marker='.', s=20, color=c, edgecolor='black', linewidth=0.5,
                zorder=3
            )

    if ligand_values is not None:
        ax.scatter(emb_ligands[:, 0], emb_ligands[:, 1], c=ligand_values,
                   cmap='magma', vmin=0, vmax=1, marker='D', s=85,
                   edgecolor='black', linewidth=.7, zorder=6)
        for ligand_id, xy in enumerate(emb_ligands):
            ax.annotate(f'L{ligand_id}', xy, xytext=(5, 5), textcoords='offset points',
                        fontsize=8, zorder=7)
        
    # Plot the Assembled Receptors as numbered labels
    for r_idx in range(n_receptors):
        if ligand_values is not None:
            # Homomers can lie directly under their preferred ligand. Offset their
            # labels with leader lines so ligand colour and gene ID remain visible.
            offsets = ((-16, -16), (0, -24), (16, -16), (16, 16), (0, 24), (-16, 16))
            ax.annotate(str(r_idx), emb_receptors[r_idx],
                        xytext=offsets[r_idx % len(offsets)], textcoords='offset points',
                        fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='0.5', boxstyle='circle,pad=0.2'),
                        arrowprops=dict(arrowstyle='-', color='0.5', lw=.5), zorder=5)
            continue
        ax.text(
            emb_receptors[r_idx, 0], emb_receptors[r_idx, 1], str(r_idx),
            fontsize=8, ha='center', va='center', fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle,pad=0.2', alpha=0.8),
            zorder=5
        )
    
    # Add a dummy scatter point so "Receptors" appears cleanly in the legend
    ax.scatter([], [], marker='o', color='white', edgecolor='black', label='Receptors')
    
    # Clean up aesthetics
    ax.set_title(f"UMAP Projection of {embedding['latent_dim']}D Chemical Latent Space", fontsize=9, fontweight='bold')
    ax.set_xlabel("UMAP Dimension 1", fontsize=9)
    ax.set_ylabel("UMAP Dimension 2", fontsize=9)

    ax.set_xticks([])
    ax.set_yticks([])
    
    # Shrink current axis by 20% to put legend outside
    if show_legend:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    
    return fig, ax


@torch.no_grad()
def cell_ligand_responses(env, physics, receptor_indices, readout, concentration=1.0):
    """Return (ligands, cells) probabilities for isolated ligands at fixed concentration.

    Uses exact saved ligand positions (no observation noise or concentration draws),
    the FULL receptor pool and existing cell_activity physics. The readout must have
    its checkpoint W, theta and calibrated temperature restored, in this pool order.
    This is a conditional response map, not an average over the training environment.
    """
    from src.cells import cell_activity

    if not math.isfinite(concentration) or concentration <= 0:
        raise ValueError("concentration must be finite and positive")
    if readout.W.shape[1] != len(receptor_indices):
        raise ValueError("Readout W must match the full receptor pool")
    if env.use_interface_model:
        energies = env.interaction_mu_interface(receptor_indices).permute(2, 0, 1)[:, None]
    else:
        energies = env.interaction_mu.T[:, None]
    concentrations = energies.new_full((env.n_ligands, 1), concentration)
    return cell_activity(physics, readout, energies, concentrations, receptor_indices,
                         pre_gathered=env.use_interface_model)


def plot_cell_response_umap(embedding, responses, gene_sets, concentration=1.0):
    """One panel per cell on a shared chemical UMAP, with a shared probability scale.

    Pass the homomer embedding from build_latent_umap and the (L, C) probabilities
    from cell_ligand_responses. Family colours indicate geometry only; numbered
    receptor markers identify homomers, and diamonds L0, L1, ... show responses.
    """
    if isinstance(responses, torch.Tensor):
        responses = responses.detach().cpu().numpy()
    responses = np.asarray(responses)
    n_cells = len(gene_sets)
    if n_cells == 0 or responses.shape != (len(embedding['ligands']), n_cells):
        raise ValueError("Expected one response per ligand and cell")
    if not np.isfinite(responses).all() or np.any((responses < 0) | (responses > 1)):
        raise ValueError("Cell responses must be finite probabilities in [0, 1]")
    ncols = min(3, n_cells)
    nrows = math.ceil(n_cells / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             sharex=True, sharey=True, squeeze=False, layout='constrained')
    for cell, ax in enumerate(axes.flat):
        if cell >= n_cells:
            ax.set_visible(False)
            continue
        _draw_latent_umap(embedding, ax=ax, ligand_values=responses[:, cell], show_legend=False)
        ax.set_title(f"Cell {cell} — genes {', '.join(map(str, gene_sets[cell]))}", fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap='magma'),
                 ax=list(axes.flat)[:n_cells], label='Cell firing probability', shrink=.8)
    fig.suptitle(f"Single-ligand cell responses at concentration {concentration:g}\n"
                 "Diamonds: ligands • numbered circles: gene homomers • clouds: families")
    return fig, axes

@torch.no_grad()
def receptor_distances(env, receptor_indices):
    """
    Computes the pairwise Euclidean distance between receptors in the latent space.

    Classic model: centroid of unit_latent embeddings.
    Interface model: centroid of pocket embeddings (average of adjacent +/- faces).

    Args:
        env: Instantiated LigandEnvironment.
        receptor_indices: Tensor of shape (N_Receptors, k_sub) mapping units to receptors.

    Returns:
        dist_matrix: A (N_Receptors, N_Receptors) numpy array of pairwise distances.
    """
    if getattr(env, 'use_interface_model', False):
        # Pocket centroid: mean over k_sub interfaces, each averaged from +/- faces.
        idx_i = receptor_indices
        idx_j = receptor_indices.roll(-1, dims=1)
        v_plus_r  = env.unit_latent_plus[idx_i]   # (R, k, D)
        v_minus_r = env.unit_latent_minus[idx_j]  # (R, k, D)
        v_pocket  = 0.5 * (v_plus_r + v_minus_r)  # (R, k, D)
        v_receptors = v_pocket.mean(dim=1)          # (R, D)
    else:
        v_receptors = env.unit_latent[receptor_indices].mean(dim=1)  # (R, D)

    dist_matrix = torch.cdist(v_receptors, v_receptors, p=2.0)
    return dist_matrix.cpu().numpy()

@torch.no_grad()
def mean_receptor_distance(env, receptor_indices):
    """Computes the exact average pairwise distance between distinct receptors."""
    dist_matrix = receptor_distances(env, receptor_indices)
    # Extract the upper triangle, excluding the diagonal (k=1)
    upper_triangle = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    if len(upper_triangle) == 0:
        return 0.0
    return float(upper_triangle.mean())

@torch.no_grad()
def marginal_entropy(activity, loss_fn):
    """Computes the sum of marginal entropies for the receptor array."""
    act = activity.detach()
    if hasattr(loss_fn, '_compute_soft_histogram_entropy'):
        return loss_fn._compute_soft_histogram_entropy(act).sum().item()
    elif hasattr(loss_fn, 'compute_kde_marginal_entropies'):
        return loss_fn.compute_kde_marginal_entropies(act).sum().item()
    elif hasattr(loss_fn, '_compute_analytical_marginal_entropies'):
        return loss_fn._compute_analytical_marginal_entropies(act).sum().item()
    return 0.0

def miller_madow_entropy(activity: torch.Tensor):
    """
    Hard-codeword plug-in entropy and Miller-Madow bias correction.

    Binarises activity at 0.5, treats each R-bit row as a symbol, and computes:
      H_plugin  = -Σ p_k log2(p_k)   (plug-in estimator, biased downward)
      H_MM      = H_plugin + (K_hat - 1) / (2 · B · ln2)   (bias-corrected)
      log2_B    = log2(B)             (trivial upper ceiling)
      K_hat     = # distinct codewords observed
      K_frac    = K_hat / 2^R         (fraction of the binary alphabet seen)

    Uses torch.unique (sort-based, O(B log B)) — no 2^R allocation.
    """
    B, R = activity.shape
    codes = (activity > 0.5).long()
    _, counts = torch.unique(codes, dim=0, return_counts=True)
    K_hat = counts.numel()
    p = counts.float() / B
    H_plugin = -(p * torch.log2(p.clamp(min=1e-12))).sum().item()
    H_MM = H_plugin + (K_hat - 1) / (2 * B * math.log(2))
    return H_plugin, H_MM, K_hat, math.log2(B), math.ldexp(float(K_hat), -R)


@torch.no_grad()
def _measure_entropy(loss_fn, act, entropy_type):
    """One entropy estimator on the soft joint activity (bits), or None if the loss
    does not support it. ``entropy_type=None`` → the loss's NATIVE estimator (its
    compute_entropy default: collision for a collision loss, blocked for annealed,
    blocked_corrected for blocked_to_corrected, kt for a kt loss, …)."""
    if entropy_type == 'kt':
        # Not exposed via compute_entropy on every loss (AnnealedEntropyLoss rejects
        # it), so go through the soft assignment. Measured on ALL samples given (KT
        # tiles internally, so its memory is bounded by chunk_size, not B); the
        # sample count is chosen by the caller (_eval_stats measures on the full
        # test_batch_size = min(2^R, memory)).
        soft = loss_fn.compute_soft_assignment(act)
        return compute_kt_entropy(soft).item()
    if entropy_type == 'kt_upper':
        # KT certified UPPER bound (KL kernel) — same all-pairs cost, all samples.
        soft = loss_fn.compute_soft_assignment(act)
        return compute_kt_upper_entropy(soft).item()
    if hasattr(loss_fn, 'compute_entropy'):
        kw = {} if entropy_type is None else {'entropy_type': entropy_type}
        return loss_fn.compute_entropy(act, use_cache=False, **kw).item()
    # Legacy losses without compute_entropy: collision or shannon on the soft assign.
    if hasattr(loss_fn, 'compute_soft_assignment'):
        soft = loss_fn.compute_soft_assignment(act)
        fn = (compute_collision_entropy
              if getattr(loss_fn, 'entropy_type', 'shannon') == 'collision'
              else compute_shannon_joint_entropy)
        return fn(soft).item()
    if hasattr(loss_fn, 'compute_knn_joint_entropy'):
        return loss_fn.compute_knn_joint_entropy(act, k=5).item()
    return None


def full_array_entropy(activity, loss_fn):
    """Native joint entropy of the array — the estimator the loss trains on.

    Logs ONLY ``full_array_entropy`` (= ``loss_fn.compute_entropy`` with its default
    type). To also record OTHER estimators as separate columns, add the opt-in
    measurements ``entropy_collision`` / ``entropy_blocked`` /
    ``entropy_blocked_corrected`` / ``entropy_kt`` to ``measurement_fns`` — each is a
    full extra evaluation, so they are off by default. Add ``codeword_entropy`` for
    the hard-codeword plug-in / Miller-Madow estimates.
    """
    val = _measure_entropy(loss_fn, activity.detach(), entropy_type=None)
    return {'full_array_entropy': val if val is not None else 0.0}


def _opt_entropy(activity, loss_fn, entropy_type, key):
    """Body for the opt-in single-estimator measurements. Silent (empty dict) when
    the estimator is unsupported for this loss, so it never breaks a run."""
    try:
        val = _measure_entropy(loss_fn, activity.detach(), entropy_type)
    except (ValueError, AttributeError, RuntimeError):
        val = None
    return {key: val} if val is not None else {}


def entropy_collision(activity, loss_fn):
    """Opt-in: collision H2 (Rényi-2) of the joint activity → full_array_entropy_collision."""
    return _opt_entropy(activity, loss_fn, 'collision', 'full_array_entropy_collision')


def entropy_blocked(activity, loss_fn):
    """Opt-in: blocked Shannon of the joint activity → full_array_entropy_blocked."""
    return _opt_entropy(activity, loss_fn, 'blocked', 'full_array_entropy_blocked')


def entropy_blocked_corrected(activity, loss_fn):
    """Opt-in: blocked-corrected Shannon → full_array_entropy_blocked_corrected."""
    return _opt_entropy(activity, loss_fn, 'blocked_corrected',
                        'full_array_entropy_blocked_corrected')


def entropy_kt(activity, loss_fn):
    """Opt-in: Kolchinsky-Tracey lower bound on H(s) on all provided samples
    → full_array_entropy_kt."""
    return _opt_entropy(activity, loss_fn, 'kt', 'full_array_entropy_kt')


def entropy_kt_upper(activity, loss_fn):
    """Opt-in: Kolchinsky-Tracey UPPER bound on H(s) on all provided samples
    → full_array_entropy_kt_upper. Pairs with entropy_kt to bracket H(s)."""
    return _opt_entropy(activity, loss_fn, 'kt_upper', 'full_array_entropy_kt_upper')


def conditional_entropy_response(activity):
    """H(Y|X) from the Bernoulli outputs; X includes sampled observation noise."""
    return compute_response_conditional_entropy(activity).item()


def mutual_information_kt(activity, loss_fn):
    """KT lower bound on information about the complete sampled input."""
    return compute_kt_entropy(loss_fn.compute_soft_assignment(activity), return_mi=True).item()


def mutual_information_kt_upper(activity, loss_fn):
    """KT upper bound on information about the complete sampled input."""
    return compute_kt_upper_entropy(loss_fn.compute_soft_assignment(activity), return_mi=True).item()


def response_counting_metrics(codes, h_cond):
    """Sampled-output H minus analytic H(Y|X), with no population-bound claim.

    Negative estimates are retained: finite-sample bias or Monte Carlo error must
    not be hidden by clamping. K_hat / B diagnoses sparse observed-code coverage.
    """
    plugin, mm, k_hat, log2_b, _ = miller_madow_entropy(codes)
    return {
        'response_entropy_plugin': plugin,
        'response_entropy_mm': mm,
        'mutual_information_counting_plugin': plugin - h_cond,
        'mutual_information_counting_mm': mm - h_cond,
        'conditional_entropy_response': h_cond,
        'response_counting_K_hat': float(k_hat),
        'response_counting_unique_fraction': k_hat / codes.shape[0],
        'response_counting_samples': int(codes.shape[0]),
        'response_counting_log2B': log2_b,
    }


@torch.no_grad()
def mutual_information_counting(activity):
    """Count one independent Bernoulli output per sniff, then subtract H(Y|X).

    For large evaluations SimulationRunner streams the draws on a separate RNG
    and stores only CPU binary codes. This differs from thresholded codeword_entropy.
    """
    a = activity.clamp(KT_EPS, 1.0 - KT_EPS)
    return response_counting_metrics(torch.bernoulli(a).bool(),
                                     compute_response_conditional_entropy(a).item())


@torch.no_grad()
def codeword_entropy(activity):
    """
    Hard-codeword plug-in entropy and Miller-Madow bias correction.

    Binarises activity at 0.5, counts distinct R-bit patterns, and returns:
      'codeword_entropy_plugin' — plug-in estimator (biased downward, ≤ log₂B)
      'codeword_entropy_mm'     — Miller-Madow corrected estimate
      'codeword_entropy_log2B'  — log₂(B), trivial ceiling for the plugin
      'codeword_entropy_K_hat'  — number of distinct codewords observed
      'codeword_entropy_K_frac' — K_hat / 2^R, fraction of alphabet sampled

    Add 'codeword_entropy' to measurement_fns to opt in.
    When eval_chunk_size < test_batch_size, _eval_stats accumulates hard codes
    across all chunks so that the full test_batch_size budget is used here.
    """
    act = activity.detach()
    H_plugin, H_MM, K_hat, log2_B, K_frac = miller_madow_entropy(act)
    return {
        'codeword_entropy_plugin': H_plugin,
        'codeword_entropy_mm':     H_MM,
        'codeword_entropy_log2B':  log2_B,
        'codeword_entropy_K_hat':  float(K_hat),
        'codeword_entropy_K_frac': K_frac,
    }

@torch.no_grad()
def total_correlation(activity, loss_fn):
    """Computes the total correlation (redundancy) of the array using Rényi entropy."""
    h_marginals = marginal_entropy(activity, loss_fn)
    h_joint     = full_array_entropy(activity, loss_fn)
    h_joint_val = h_joint.get('full_array_entropy', 0.0) if isinstance(h_joint, dict) else h_joint
    return h_marginals - h_joint_val

@torch.no_grad()
def conditional_entropy_ligand(activity, mixture_masks, loss_fn):
    """Mean over ligands of H(A | L_l), where L_l ∈ {0,1} is the binary
    presence indicator for ligand l (independent marginal conditioning).

    For each ligand l:
      H(A | L_l) = P(l=1)·H(A | l present) + P(l=0)·H(A | l absent)

    Returns (1/N_l) Σ_l H(A | L_l).  The corresponding MI is the mean
    pairwise I(A ; L_l) over ligands — see mutual_information_ligand.
    """
    if not hasattr(loss_fn, 'compute_soft_assignment'):
        return 0.0

    soft_assign = loss_fn.compute_soft_assignment(activity)
    entropy_fn = (compute_collision_entropy
                  if getattr(loss_fn, 'entropy_type', 'shannon') == 'collision'
                  else compute_shannon_joint_entropy)

    B, n_ligands = mixture_masks.shape[0], mixture_masks.shape[1]
    total_cond_h = 0.0

    for l in range(n_ligands):
        present = mixture_masks[:, l].bool()
        absent  = ~present
        cond_h_l = 0.0
        for mask in (present, absent):
            n = mask.sum().item()
            if n > 1:
                h = entropy_fn(soft_assign[mask])
                cond_h_l += (n / B) * (h.item() if isinstance(h, torch.Tensor) else h)
        total_cond_h += cond_h_l

    return total_cond_h / n_ligands

@torch.no_grad()
def mutual_information_ligand(activity, mixture_masks, loss_fn):
    """Mean over ligands of I(A ; L_l), where L_l is the binary presence
    indicator for ligand l.

    Formula: (1/N_l) Σ_l I(A ; L_l) = H(A) - conditional_entropy_ligand(...)
    """
    if not hasattr(loss_fn, 'compute_soft_assignment'):
        return 0.0

    soft_assign = loss_fn.compute_soft_assignment(activity)
    entropy_fn = (compute_collision_entropy
                  if getattr(loss_fn, 'entropy_type', 'shannon') == 'collision'
                  else compute_shannon_joint_entropy)
    h_a = entropy_fn(soft_assign)
    h_a_val = h_a.item() if isinstance(h_a, torch.Tensor) else h_a
    h_a_given_m = conditional_entropy_ligand(activity, mixture_masks, loss_fn)
    return h_a_val - h_a_given_m

@torch.no_grad()
def _binned_conditional_entropy(soft_present, c_present, entropy_fn, n_c_bins):
    """H(A | C) over a present-only subset: sort by concentration c, split into
    n_c_bins equal-count bins, return the count-weighted mean bin entropy."""
    n = soft_present.shape[0]
    order = torch.argsort(c_present)
    a_sorted = soft_present[order]
    bin_size = max(1, n // n_c_bins)
    cond_h = 0.0
    for b in range(n_c_bins):
        start = b * bin_size
        end   = start + bin_size if b < n_c_bins - 1 else n
        chunk = a_sorted[start:end]
        if chunk.shape[0] > 1:
            h = entropy_fn(chunk)
            cond_h += (chunk.shape[0] / n) * (h.item() if isinstance(h, torch.Tensor) else h)
    return cond_h


@torch.no_grad()
def conditional_entropy_concentration(activity, concs_dense, mixture_masks, loss_fn, n_c_bins=10):
    """Mean over ligands of H(A | C_l, l present).

    For each ligand l we restrict to the samples where l is actually present
    (mixture_masks[:, l] == 1), bin those by the dense per-ligand concentration
    concs_dense[:, l] into n_c_bins equal-count bins, and average H(A | bin).

    concs_dense is the DENSE (B, L) per-ligand concentration (0 for absent ligands),
    identity-aligned to mixture_masks — NOT the sparse present-slot tensor.  The
    present-only restriction removes the presence/absence confound, so the result
    measures genuine concentration (level) coding.  Ligands present in < 2*n_c_bins
    samples are skipped; returns the mean over the scored ligands.
    """
    if not hasattr(loss_fn, 'compute_soft_assignment'):
        return 0.0

    soft_assign = loss_fn.compute_soft_assignment(activity)
    entropy_fn = (compute_collision_entropy
                  if getattr(loss_fn, 'entropy_type', 'shannon') == 'collision'
                  else compute_shannon_joint_entropy)

    L = mixture_masks.shape[1]
    total_cond_h, scored = 0.0, 0
    for l in range(L):
        present = mixture_masks[:, l].bool()
        n_p = int(present.sum())
        if n_p < 2 * n_c_bins:
            continue
        total_cond_h += _binned_conditional_entropy(
            soft_assign[present], concs_dense[present, l], entropy_fn, n_c_bins)
        scored += 1

    return total_cond_h / scored if scored else 0.0


@torch.no_grad()
def mutual_information_concentration(activity, concs_dense, mixture_masks, loss_fn, n_c_bins=10):
    """Mean over ligands of I(A ; C_l | l present) — how much the array output
    depends on ligand l's concentration among the samples where l is present:

        I_l = H(A | l present) - H(A | C_l, l present)

    Uses the DENSE (B, L) concentration, so it is free of the presence/padding
    confound of the old sparse-slot version.  This is a per-ligand MARGINAL
    average and is NOT on the same scale as the joint full_array_entropy; for a
    split that sums to the total, see identity_channel / concentration_channel.
    """
    if not hasattr(loss_fn, 'compute_soft_assignment'):
        return 0.0

    soft_assign = loss_fn.compute_soft_assignment(activity)
    entropy_fn = (compute_collision_entropy
                  if getattr(loss_fn, 'entropy_type', 'shannon') == 'collision'
                  else compute_shannon_joint_entropy)

    L = mixture_masks.shape[1]
    total_mi, scored = 0.0, 0
    for l in range(L):
        present = mixture_masks[:, l].bool()
        n_p = int(present.sum())
        if n_p < 2 * n_c_bins:
            continue
        a_present = soft_assign[present]
        h_present = entropy_fn(a_present)
        h_present = h_present.item() if isinstance(h_present, torch.Tensor) else h_present
        h_cond = _binned_conditional_entropy(
            a_present, concs_dense[present, l], entropy_fn, n_c_bins)
        total_mi += (h_present - h_cond)
        scored += 1

    return total_mi / scored if scored else 0.0


@torch.no_grad()
def concentration_channel(activity, mixture_masks, loss_fn):
    """Concentration channel H(A | M): condition on the full presence pattern M.

    Samples are grouped by identical presence pattern (torch.unique over rows);
    within a group the composition is fixed, so the only thing left varying is the
    concentrations -> this residual IS the concentration channel I(A; c | M).
    With identity_channel it forms an EXACT, total-comparable split:
        H(A) = identity_channel + concentration_channel = I(A;M) + H(A|M).

    Estimable when patterns repeat (single-ligand / low-mu sniffs, where M is
    effectively the categorical ligand id).  In dense mixtures most rows are
    unique -> singleton groups -> this collapses toward 0 and identity_channel
    -> H(A) spuriously; treat it as a low-mu-regime metric.
    """
    if not hasattr(loss_fn, 'compute_soft_assignment'):
        return 0.0

    soft_assign = loss_fn.compute_soft_assignment(activity)
    entropy_fn = (compute_collision_entropy
                  if getattr(loss_fn, 'entropy_type', 'shannon') == 'collision'
                  else compute_shannon_joint_entropy)

    B = activity.shape[0]
    _, inverse = torch.unique(mixture_masks.bool(), dim=0, return_inverse=True)
    total_cond_h = 0.0
    for g in inverse.unique():
        grp = inverse == g
        n = int(grp.sum())
        if n > 1:
            h = entropy_fn(soft_assign[grp])
            total_cond_h += (n / B) * (h.item() if isinstance(h, torch.Tensor) else h)
    return total_cond_h


@torch.no_grad()
def identity_channel(activity, mixture_masks, loss_fn):
    """Identity channel I(A ; M) = H(A) - H(A | M): how much the codeword reveals
    about which ligands are present, on the same joint scale as full_array_entropy.
    Together with concentration_channel the two sum to H(A).  See
    concentration_channel for the single-ligand-regime caveat.
    """
    if not hasattr(loss_fn, 'compute_soft_assignment'):
        return 0.0

    soft_assign = loss_fn.compute_soft_assignment(activity)
    entropy_fn = (compute_collision_entropy
                  if getattr(loss_fn, 'entropy_type', 'shannon') == 'collision'
                  else compute_shannon_joint_entropy)
    h_a = entropy_fn(soft_assign)
    h_a_val = h_a.item() if isinstance(h_a, torch.Tensor) else h_a
    return h_a_val - concentration_channel(activity, mixture_masks, loss_fn)


# ---------------------------------------------------------------------------
# Family-level mutual information
# ---------------------------------------------------------------------------

@torch.no_grad()
def conditional_entropy_family(activity, family_labels, loss_fn):
    """Mean over families of H(A | F_f), where F_f ∈ {0,1} is the binary
    presence indicator for family f (independent marginal conditioning).

    For each family f:
      H(A | F_f) = P(f=1)·H(A | f present) + P(f=0)·H(A | f absent)

    Returns (1/N_f) Σ_f H(A | F_f).  The corresponding MI is the mean
    pairwise I(A ; F_f) over families — see mutual_information_family.
    """
    if not hasattr(loss_fn, 'compute_soft_assignment'):
        return 0.0

    soft_assign = loss_fn.compute_soft_assignment(activity)
    entropy_fn = (compute_collision_entropy
                  if getattr(loss_fn, 'entropy_type', 'shannon') == 'collision'
                  else compute_shannon_joint_entropy)

    B, n_fam = family_labels.shape
    total_cond_h = 0.0

    for f in range(n_fam):
        present = family_labels[:, f]   # (B,) bool
        absent  = ~present
        cond_h_f = 0.0
        for mask in (present, absent):
            n = mask.sum().item()
            if n > 1:
                h = entropy_fn(soft_assign[mask])
                cond_h_f += (n / B) * (h.item() if isinstance(h, torch.Tensor) else h)
        total_cond_h += cond_h_f

    return total_cond_h / n_fam


@torch.no_grad()
def mutual_information_family(activity, family_labels, loss_fn):
    """Mean over families of I(A ; F_f), where F_f is the binary presence
    indicator for family f.

    Formula: (1/N_f) Σ_f I(A ; F_f) = H(A) - conditional_entropy_family(...)
    """
    if not hasattr(loss_fn, 'compute_soft_assignment'):
        return 0.0

    soft_assign = loss_fn.compute_soft_assignment(activity)
    entropy_fn = (compute_collision_entropy
                  if getattr(loss_fn, 'entropy_type', 'shannon') == 'collision'
                  else compute_shannon_joint_entropy)
    h_a = entropy_fn(soft_assign)
    h_a_val = h_a.item() if isinstance(h_a, torch.Tensor) else h_a
    h_a_given_f = conditional_entropy_family(activity, family_labels, loss_fn)
    return h_a_val - h_a_given_f


# ---------------------------------------------------------------------------
# Block-level mutual information (presence / source blocks)
# ---------------------------------------------------------------------------

@torch.no_grad()
def conditional_entropy_block(activity, block_labels, loss_fn):
    """Mean over presence blocks of H(A | B_b), where B_b ∈ {0,1} is the
    binary indicator that ≥1 ligand from presence block b is present in a
    sample (independent marginal conditioning, one block at a time).

    Mirrors conditional_entropy_family but conditions on the source
    blocks (env.presence_block_id) rather than the latent-space families.
    The two partitions are orthogonal by construction, so this captures a
    genuinely different structure.

    For each block b:
      H(A | B_b) = P(b=1)·H(A | b present) + P(b=0)·H(A | b absent)

    Returns (1/N_b) Σ_b H(A | B_b).
    """
    if not hasattr(loss_fn, 'compute_soft_assignment'):
        return 0.0

    soft_assign = loss_fn.compute_soft_assignment(activity)
    entropy_fn = (compute_collision_entropy
                  if getattr(loss_fn, 'entropy_type', 'shannon') == 'collision'
                  else compute_shannon_joint_entropy)

    B, n_blocks = block_labels.shape
    total_cond_h = 0.0

    for b in range(n_blocks):
        present = block_labels[:, b]   # (B,) bool
        absent  = ~present
        cond_h_b = 0.0
        for mask in (present, absent):
            n = mask.sum().item()
            if n > 1:
                h = entropy_fn(soft_assign[mask])
                cond_h_b += (n / B) * (h.item() if isinstance(h, torch.Tensor) else h)
        total_cond_h += cond_h_b

    return total_cond_h / n_blocks


@torch.no_grad()
def mutual_information_block(activity, block_labels, loss_fn):
    """Mean over presence blocks of I(A ; B_b), where B_b ∈ {0,1} is the
    binary indicator that ≥1 ligand from presence block b is present.

    Formula: (1/N_b) Σ_b I(A ; B_b) = H(A) − conditional_entropy_block(...)

    Use alongside mutual_information_family to disentangle the contribution
    of source co-occurrence (blocks) from chemical similarity (families).
    """
    if not hasattr(loss_fn, 'compute_soft_assignment'):
        return 0.0

    soft_assign = loss_fn.compute_soft_assignment(activity)
    entropy_fn = (compute_collision_entropy
                  if getattr(loss_fn, 'entropy_type', 'shannon') == 'collision'
                  else compute_shannon_joint_entropy)
    h_a = entropy_fn(soft_assign)
    h_a_val = h_a.item() if isinstance(h_a, torch.Tensor) else h_a
    h_a_given_b = conditional_entropy_block(activity, block_labels, loss_fn)
    return h_a_val - h_a_given_b


@torch.no_grad()
def rank_ordered_distances(env, receptor_indices):
    """
    Calculates the rank-ordered energy gap from each assembled receptor to the ligands.
    Returns a dictionary of the average energy penalty relative to the preferred ligand
    (0.0 = preferred ligand, higher values = weaker affinity / stronger rejection).
    This bypasses the label-switching problem to measure functional tuning.
    (Note: the dictionary keys are kept as 'dist_rank_i' for backward compatibility with plotting scripts)
    """
    if getattr(env, 'use_interface_model', False):
        # Interface model: (R, k_sub, n_ligands) → mean over k_sub → (R, n_ligands)
        receptor_energies = env.interaction_mu_interface(receptor_indices).mean(dim=1)
    else:
        # Classic model: (n_genes, n_ligands) → gather → (R, k_sub, n_ligands) → mean → (R, n_ligands)
        unit_energies = env.interaction_mu          # (n_genes, n_ligands)
        receptor_energies = unit_energies[receptor_indices].mean(dim=1)
    
    # 3. Normalize energies per receptor: compute the energy gap relative to the preferred ligand
    min_energies = receptor_energies.min(dim=1, keepdim=True)[0]
    normalized_energies = receptor_energies - min_energies
    
    # 4. Sort normalized energies for each receptor in ascending order (0.0 = strongest affinity)
    sorted_energies, _ = torch.sort(normalized_energies, dim=1) 
    
    # 5. Average across all receptors
    mean_sorted_energies = sorted_energies.mean(dim=0).cpu().numpy()
    
    result = {}
    for i, energy in enumerate(mean_sorted_energies):
        result[f"dist_rank_{i+1}"] = float(energy)
        
    return result

@torch.no_grad()
def mean_specialization_index(activity, mixture_masks):
    """
    Computes the signal-to-noise ratio of receptor activation.
    S_r = (A_max - A_bg) / (A_max + A_bg)
    A value of 1.0 means a perfect specialist, 0.0 means a perfect generalist.
    """
    B, R = activity.shape
    
    powers = 2 ** torch.arange(mixture_masks.shape[1], device=mixture_masks.device, dtype=mixture_masks.dtype)
    mixture_ids = (mixture_masks * powers).sum(dim=-1).long()
    unique_mixtures = torch.unique(mixture_ids)
    n_mixtures = len(unique_mixtures)
    
    if n_mixtures <= 1:
        return 0.0 
        
    avg_act = torch.zeros(n_mixtures, R, device=activity.device)
    
    for i, m_idx in enumerate(unique_mixtures):
        mask = (mixture_ids == m_idx)
        if mask.any():
            avg_act[i] = activity[mask].mean(dim=0)
            
    A_max, _ = avg_act.max(dim=0) # (R,)
    A_bg = (avg_act.sum(dim=0) - A_max) / (n_mixtures - 1) # (R,)
    
    S_r = (A_max - A_bg) / (A_max + A_bg + 1e-12)
    return float(S_r.mean().item())

@torch.no_grad()
def receptor_conditioned_entropy(activity, mixture_masks, threshold=0.5):
    """
    Computes the average entropy of the mixture distribution conditioned on a receptor firing:
    H(M | a_r > threshold). 
    A lower value indicates a more highly specialized receptor.
    """
    B, R = activity.shape
    powers = 2 ** torch.arange(mixture_masks.shape[1], device=mixture_masks.device, dtype=mixture_masks.dtype)
    mixture_ids = (mixture_masks * powers).sum(dim=-1).long()
    unique_mixtures = torch.unique(mixture_ids)
    
    if len(unique_mixtures) <= 1:
        return 0.0
        
    total_entropy = 0.0
    valid_receptors = 0
    
    for r in range(R):
        active_mask = activity[:, r] > threshold
        if not active_mask.any():
            continue # Receptor never fired in this batch, skip
            
        active_mixtures = mixture_ids[active_mask]
        
        # Compute mixture frequencies for when this receptor fires
        mixture_counts = torch.bincount(active_mixtures, minlength=torch.max(unique_mixtures)+1)
        mixture_counts = mixture_counts[unique_mixtures] 
        
        p_f = mixture_counts.float() / active_mixtures.size(0)
        p_f = p_f[p_f > 0] # Filter out zeros to avoid log2(0) NaN
        
        h_r = -torch.sum(p_f * torch.log2(p_f))
        total_entropy += h_r.item()
        valid_receptors += 1
        
    if valid_receptors == 0:
        return 0.0
        
    return total_entropy / valid_receptors
