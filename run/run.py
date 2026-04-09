import sys
sys.path.append('/app')
# unit_test/test_single_receptor.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import inspect
from itertools import cycle


from src import (LigandEnvironment,
                SymmetricLigandEnvironment,
                BinaryReceptor,
                BaseReceptor,
                generate_receptor_indices, 
                plot_family_summary,
                LogNormalConcentration,
                plot_latent_radar_chart,
                evaluate_model,
                plot_summary,
                plot_latent_umap,
                marginal_entropy,
                full_array_entropy,
                total_correlation)
from objectives import DiscreteProxyLoss,DiscreteExactLoss

def estimate_memory_usage(CONF: dict) -> float:
    """
    Estimates the peak VRAM memory footprint for the simulation.
    Returns the estimated peak memory in Megabytes (MB).
    """
    bytes_per_float = 4
    B_train = CONF.get('batch_size', 2000)
    B_eval = CONF.get('eval_batch_size', 100_000)
    U = CONF.get('n_units', 26)
    D = CONF.get('latent_dim', 3)
    F = CONF.get('n_families', 10)
    receptor_indices = CONF.get('receptor_indices', torch.zeros((1, 5)))
    R = len(receptor_indices)
    k_sub = CONF.get('k_sub', 5)
    exact_loss = CONF.get('exact_loss', False)
    
    # 1. Parameter memory (Negligible)
    mem_params = (F * D + U * D + U) * bytes_per_float
    
    def forward_memory(B):
        mem_ligands = B * D * bytes_per_float
        mem_dists = B * U * bytes_per_float * 2 # dist_sq and E_open
        mem_physics = B * R * k_sub * bytes_per_float
        mem_activity = B * R * bytes_per_float
        
        mem_loss = 0
        if exact_loss:
            if R <= 10:
                mem_loss = B * (2**R) * bytes_per_float
            else:
                M = min(B, 2048)
                mem_loss = (B * M) * bytes_per_float
        else:
            mem_loss = (B * R * 2 + R * R) * bytes_per_float
            
        return mem_ligands + mem_dists + mem_physics + mem_activity + mem_loss
        
    mem_train_forward = forward_memory(B_train)
    mem_train_total = mem_train_forward * 3 # Rough autograd graph multiplier
    
    mem_eval = forward_memory(B_eval) # Evaluation requires no gradients
    
    # PyTorch Base CUDA Context Overhead
    cuda_context = 1024 * 1024 * 1024 
    
    peak_mem = max(mem_train_total, mem_eval) + mem_params + cuda_context
    to_mb = lambda b: b / (1024 * 1024)
    
    print(f"--- VRAM Memory Estimation ---")
    print(f"CUDA Context Overhead : {to_mb(cuda_context):.1f} MB")
    print(f"Train Pass (w/ Grad)  : {to_mb(mem_train_total):.1f} MB  (Batch={B_train})")
    print(f"Evaluation Pass       : {to_mb(mem_eval):.1f} MB  (Batch={B_eval})")
    print(f"------------------------------")
    print(f"Estimated Peak VRAM   : {to_mb(peak_mem):.1f} MB")
    print(f"------------------------------")
    
    return to_mb(peak_mem)

def initialize(CONF:dict,SymmetricEnv=False)->tuple[LigandEnvironment,BaseReceptor,nn.Module,optim.Optimizer]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conc_strategy = LogNormalConcentration(n_families=CONF['n_families'], 
                                            init_mean=CONF['init_means'])
                                            
    if SymmetricEnv:
        env = SymmetricLigandEnvironment(CONF['n_units'],
                            CONF['n_families'], 
                            conc_model=conc_strategy,
                            latent_dim=CONF['latent_dim'],
                            shape_sigma=CONF.get('shape_sigma', 0.5),
                            avg_family_distance=CONF.get("average_family_distance", 1.0)).to(device)
    else:
        env = LigandEnvironment(CONF['n_units'],
                            CONF['n_families'], 
                            conc_model=conc_strategy,
                            latent_dim=CONF['latent_dim'],
                            shape_sigma=CONF.get('shape_sigma', 0.5),
                            avg_family_distance=CONF.get("average_family_distance", 1.0)).to(device)
    physics = BinaryReceptor(CONF["n_units"], CONF["k_sub"],temperature=CONF["temperature"]).to(device)
    
    if CONF.get("exact_loss", False):
        loss_fn = DiscreteExactLoss().to(device)
    else:
        loss_fn = DiscreteProxyLoss(cov_weight = CONF["cov_weight"]).to(device)
    optimizer = optim.Adam(list(env.parameters()) + 
                            list(physics.parameters()),
                            lr=CONF["lr"])
    
    return env,physics,loss_fn,optimizer


def train(CONF:dict,
        env:LigandEnvironment,
        physics:BaseReceptor,
        loss_fn:nn.Module,
        optimizer:optim.Optimizer,
        measurement_fns:list=None)->list:

    #env,physics,loss_fn,optimizer = initialize(CONF)

    print(f"Training for {CONF['epochs']} epochs...")

#    if measurement_fns is None:
#        measurement_fns = [full_array_entropy, marginal_entropy, total_correlation]

    scheduler = None
    if CONF.get('use_scheduler', False):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONF['epochs'], eta_min=1e-5)

    stats = []
    for epoch in range(CONF['epochs']):
        optimizer.zero_grad()
        
        # A. Sample Batch
        # energies: (B, 1, 2), concs: (B,)
        energies, concs, _ = env.sample_batch(CONF['batch_size'])
        
        # B. Physics
        # activity: (B, 1)
        activity = physics(energies, concs, CONF["receptor_indices"])
        

        # C. Loss (Maximize Entropy)
        loss = loss_fn(activity)
        
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        if epoch % (CONF['epochs']//100) == 0:
            with torch.no_grad():
                # 1. Generate a large evaluation batch (configurable to prevent OOM)
                eval_batch = CONF.get('eval_batch_size', 100_000)
                E_open_stats, concs_stats, _ = env.sample_batch(batch_size=eval_batch)
                
                # 2. Get probabilities
                activity_stats = physics(E_open_stats, concs_stats, CONF["receptor_indices"])
                
                stat = {}
                
                # 3. Measurements
                for fn in measurement_fns:
                    sig = inspect.signature(fn)
                    kwargs = {}
                    # Automatically map the requested arguments
                    if 'env' in sig.parameters: kwargs['env'] = env
                    if 'physics' in sig.parameters: kwargs['physics'] = physics
                    if 'receptor_indices' in sig.parameters: kwargs['receptor_indices'] = CONF["receptor_indices"]
                    if 'loss_fn' in sig.parameters: kwargs['loss_fn'] = loss_fn
                    if 'activity' in sig.parameters: kwargs['activity'] = activity_stats
                    if 'epoch' in sig.parameters: kwargs['epoch'] = epoch
                    
                    result = fn(**kwargs)
                    if isinstance(result, dict):
                        stat.update(result)
                    else:
                        name = getattr(fn, '__name__', str(fn))
                        stat[name] = result
                stat['lr'] = optimizer.param_groups[0]['lr']
                
                stats.append(stat)
    stats = {key:[stats[i][key] for i in range(stats.__len__())] for key in stats[0].keys()}
    return stats

def test(CONF:dict,
    env:LigandEnvironment,
    physics:BaseReceptor,
    loss_fn:nn.Module,
    optimizer:optim.Optimizer,
    indices:torch.Tensor,
    N_samples:int,
    epoch:int = 100)->list:        
    ents = []
    for _ in range(epoch):
        val = evaluate_model(env=env,physics=physics,receptor_indices=indices,loss_fn=loss_fn,n_samples=N_samples)
        ents.append(val)
    return ents