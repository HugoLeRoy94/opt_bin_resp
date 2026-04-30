# Running Simulations and Sweeps

This framework is designed around two main execution flows: **Single Experiments** and **Parameter Sweeps**. Both flows utilize configurations to maintain strict reproducibility and automatically handle file I/O operations (logging, checkpointing).

---

## 1. Running via Docker (Recommended)

For isolated dependencies and guaranteed GPU access, use the provided Docker compose setup.

**Standard Run:**
```bash
docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/<script_name>.py <args>
```

**Run in Background (Detached/Silent):**
Add the `-d` flag to run without tying up your terminal.
```bash
docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run -d --rm gpu-runner python3 /app/run/script/<script_name>.py <args>
```

**Specify a Target GPU:**
Prepend the `MY_GPU` environment variable.
```bash
MY_GPU=2 docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/<script_name>.py <args>
```

---

## 2. Single Experiments

A single experiment relies on a `SingleRunConfig` and `SimulationRunner`. It is best for isolated tests, specific model evaluations, or custom iteration loops (like testing generated heteromers).

**Example Script:** `/app/run/script/opt_heteromers.py`

**Execution:**
```bash
# Run with 10 ligand families
python3 opt_heteromers.py 10
```

**Code Paradigm:**
```python
from src.config import SingleRunConfig
from src.run import SimulationRunner
from src.IO import ExperimentLogger

# 1. Define specific parameters
config = SingleRunConfig(
    n_families=10, 
    latent_dim=3, 
    n_units=5, 
    # ...
)

# 2. Set output directory
logger = ExperimentLogger(run_dir="/app/data/my_experiment")
logger.save_config(config)

# 3. Run execution
runner = SimulationRunner(config, logger)
runner.run(receptor_indices=my_indices)
```

---

## 3. Parameter Sweeps

A sweep automates the iteration over grid combinations (like varying `latent_dim` and `n_units` simultaneously). It uses `SweepConfig` to generate nested outputs cleanly and safely passes state forwards (to speed up training ascending array sizes).

**Example Script:** `/app/run/script/opt_homomers.py`

**Execution:**
```bash
# Basic run with 5 families
python3 opt_homomers.py 5

# Advanced run mapping out the proxy loss architecture over 10 samples per grid node
python3 opt_homomers.py 5 --samples 10 --loss_type proxy --env_type symmetric
```

**Analyzing Sweeps:**
Because the Sweep execution creates a strict hierarchical folder tree, use `SweepLoader` from `src.IO` in your Jupyter notebooks to automatically aggregate all generated `.csv` files back into a single massive Pandas DataFrame for analysis.