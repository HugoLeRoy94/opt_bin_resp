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
import math
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
from objectives import (DiscreteProxyLoss, 
                        DiscreteExactLoss,
                        MaximizeMutualInformationFamilyLoss,
                        MaximizeMutualInformationConcentrationLoss)

def initialize(CONF:dict,SymmetricEnv=False, prev_env=None)->tuple[LigandEnvironment,BaseReceptor,nn.Module,optim.Optimizer]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conc_strategy = LogNormalConcentration(n_families=CONF['n_families'], 
                                            init_mean=CONF['init_means'])
                                            

    env = LigandEnvironment(CONF['n_units'],
                        CONF['n_families'], 
                        conc_model=conc_strategy,
                        latent_dim=CONF['latent_dim'],
                        shape_sigma=CONF['shape_sigma'],
                        avg_family_distance=CONF["average_family_distance"]).to(device)
                        
    # Curriculum Learning: Inject the state of the previously optimized array
    if prev_env is not None:
        with torch.no_grad():
            # Copy the parameters of the previously optimized units
            n_prev = prev_env.n_units
            n_new = CONF['n_units']
            copy_len = min(n_prev, n_new)
            
            env.unit_latent.data[:copy_len] = prev_env.unit_latent.data[:copy_len].clone()
            env.base_energy_u.data[:copy_len] = prev_env.base_energy_u.data[:copy_len].clone()
            
            # Crucial: Lock the environment (families) to be exactly the same as the previous run
            if prev_env.n_families == CONF['n_families']:
                env.family_latent.data = prev_env.family_latent.data.clone()
                env.concentration_model.mu.data = prev_env.concentration_model.mu.data.clone()
                env.concentration_model.log_sigma.data = prev_env.concentration_model.log_sigma.data.clone()

    physics = BinaryReceptor(CONF["n_units"], CONF["k_sub"],temperature=CONF["temperature"]).to(device)
    
    if CONF["loss"] == "exact":
        loss_fn = DiscreteExactLoss().to(device)
    elif CONF["loss"] == "family":
        loss_fn = MaximizeMutualInformationFamilyLoss().to(device)
    elif CONF["loss"]=="conc":
        loss_fn = MaximizeMutualInformationConcentrationLoss().to(device)
    elif CONF["loss"] == "proxy":
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

    # Set up Temperature Annealing
    start_temp = 1.0
    end_temp = CONF.get("temperature", 0.1)
    print(end_temp)

    scheduler = None
    if CONF.get('use_scheduler', False):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONF['epochs'], eta_min=1e-5)

    stats = []
    for epoch in range(CONF['epochs']):
        optimizer.zero_grad()
        
        # Slower linear temperature decay allows receptors to traverse the high-dimensional void without freezing
        current_temp = end_temp + (start_temp - end_temp) * (1.0 - (epoch / CONF['epochs']))
        if hasattr(physics, 'temperature'):
            physics.temperature = current_temp

        # A. Sample Batch
        # energies: (B, 1, 2), concs: (B,)
        energies, concs, family_ids = env.sample_batch(CONF['batch_size'])
        
        # B. Physics
        # activity: (B, 1)
        activity = physics(energies, concs, CONF["receptor_indices"])
        

        # C. Loss (Maximize Entropy)
        if isinstance(loss_fn, MaximizeMutualInformationFamilyLoss):
            loss = loss_fn(activity, family_ids=family_ids)
        elif isinstance(loss_fn, MaximizeMutualInformationConcentrationLoss):
            loss = loss_fn(activity, concs=concs)
        else:
            loss = loss_fn(activity)
        
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        if epoch % (CONF['epochs']//100) == 0:
            with torch.no_grad():
                # 1. Generate a large evaluation batch (configurable to prevent OOM)
                eval_batch = CONF.get('eval_batch_size', 2**12)
                E_open_stats, concs_stats, family_ids_stats = env.sample_batch(batch_size=eval_batch)
                
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
                    if 'concs' in sig.parameters: kwargs['concs'] = concs_stats
                    if 'family_ids' in sig.parameters: kwargs['family_ids'] = family_ids_stats
                    
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