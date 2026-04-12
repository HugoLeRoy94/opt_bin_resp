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