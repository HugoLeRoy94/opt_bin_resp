# 🐳 Docker & DevPod Cheat Sheet

## 1. Concepts
* **Docker:** The "Box." It creates an isolated, virtual computer (container) inside your machine that has all the specific software (Python, CUDA, PyTorch) your project needs. It ensures your code runs exactly the same everywhere.
* **DevPod:** The "Bridge." It connects your local IDE (VSCodium) to the Docker container. It handles the complex SSH tunneling and file syncing so you can edit code inside the container as if it were on your own hard drive.

---

## 2. Key Files
| File | Role | Metaphor |
| :--- | :--- | :--- |
| **`Dockerfile`** | **The Recipe.** Tells Docker how to build the OS. "Start with NVIDIA Linux, add Python 3.10, install these libraries." | The Blueprint |
| **`docker-compose.yaml`** | **The Orchestrator.** Tells Docker how to run the container. "Use the GPU, map port 8888, sync the local folder to `/app`, and set shared memory to 2GB." | The Director |
| **`.devcontainer/devcontainer.json`** | **The Config for DevPod.** Tells DevPod which `docker-compose` file to use and which IDE extensions (Python, Jupyter) to install inside the container automatically. | The Connection Settings |

---

## 3. Command Line Survival Guide

### 🐳 Docker Commands (The Engine)
* **Start Everything:**
    ```bash
    sudo docker-compose up       # Starts the container and shows logs
    sudo docker-compose up -d    # Starts in "detached" mode (background)
    sudo docker-compose up --build # Force rebuilds the image (use if you changed requirements.txt)
    ```

* **Check Status:**
    ```bash
    sudo docker ps               # List currently running containers
    sudo docker ps -a            # List ALL containers (including stopped ones)
    ```

* **Stop Everything:**
    ```bash
    sudo docker-compose stop     # Pauses the containers (keeps state)
    sudo docker-compose down     # Stops and removes containers/networks (Clean slate)
    docker-compose up -d         # update the container, if I just changed the docker-compose file
    ```

* **Kill/Remove Containers (Fix Conflicts):**
    ```bash
    sudo docker rm -f <container_name>  # Force remove a specific container
    sudo docker rm -f $(sudo docker ps -aq)  # Nuke option: Remove ALL containers
    ```

* **View Logs:**
    ```bash
    sudo docker logs -f <container_name> # Follow the logs in real-time
    ```

### 😈 DevPod Commands (The IDE Bridge)
* **Start Work (The Magic Command):**
    ```bash
    devpod up .                 # Reads .devcontainer.json and launches Codium
    ```

* **Specify IDE (If it opens browser):**
    ```bash
    devpod up . --ide codium    # Forces VSCodium to open
    devpod up . --recreate      # Remake the workspace, if docker-compose has been changed for instance
    ```

* **Manage Workspaces:**
    ```bash
    devpod list                 # See active workspaces
    devpod stop <name>          # Stop a workspace to save GPU/RAM
    devpod delete <name>        # Delete the workspace setup (Files remain safe)
    ```

* **Fix IDE Path (Fedora Specific):**
    ```bash
    devpod ide set-options codium -o COMMAND=$(which codium)
    ```

# Run a simple Python Script
```
docker exec -it optimize_binary python3 your_script.py
```

-it is just to keep it interactive

# 🚀 GPU Cluster Survival Guide (e4-seminara)

This guide summarizes the final working configuration for running code on the University A100 GPUs using **Docker** while bypassing the broken system Python and complex SSH tunnels.

---

## 1. The Setup Architecture
We are using **Rootless Docker** to manage dependencies and GPU access without needing `sudo` permissions.

* **Host Server:** `leroy@10.187.172.7`
* **Container Name:** `optimize_binary`
* **GPU Hardware:** 4x NVIDIA A100
* **Mapping:** Your local folder `/home/leroy/optimize_binary_entropy` is mirrored inside the container at `/app`.

---

## 2. How to Work (The Workflow)
To avoid the "SSH Inception" and connection errors, we use VSCodium to edit files on the host and the terminal to execute them in the container.

### Step 1: Connect to the Server
1. Open **VSCodium** on your laptop.
2. Use the **Remote - SSH** extension to connect to `leroy@10.187.172.7`.
3. Open your project folder: `/home/leroy/optimize_binary_entropy`.
4. **Edit your code here.** Saving a file here automatically updates it inside the Docker container.

### Step 2: Run Your Code (The Docker "Bridge")
Do not run `python` directly in the VSCodium terminal. Instead, "inject" your commands into the running Docker container using `docker exec`.

* **To check GPU availability:**
    ```bash
    docker exec -it optimize_binary nvidia-smi
    ```
* **To run a Python script:**
    ```bash
    docker exec -it optimize_binary python3 your_script.py
    ```
* **To get an interactive shell inside the GPU environment:**
    ```bash
    docker exec -it optimize_binary bash
    ```

---

## 3. Important Configurations

### University Proxy (Inside the Container)
If your code needs to download data or packages, run these exports inside the container terminal:
```bash
export http_proxy=[http://proxy.unige.it:8080/](http://proxy.unige.it:8080/)
export https_proxy=[http://proxy.unige.it:8080/](http://proxy.unige.it:8080/)
```

## 4. Simulation Memory Limits & Scaling + +The simulation is highly optimized and avoids

the exponential memory scaling often associated with high-dimensional integration (the "Curse of Dimensionality"). In the physics.py module, when calculating the expected activation over a Gaussian ligand family, the code dynamically falls back from a dense Gauss-Hermite quadrature grid to a mean-energy approximation whenever the grid size would exceed 100,000 points ($10^D > 100,000$). This prevents Out Of Memory (OOM) errors in high latent dimensions. 

Here is how the computational memory footprint scales with respect to the main parameters:

### 1. Ligand Generation & Distances: 
$\mathcal{O}(B \cdot U \cdot D)$ 
* Variables: Batch Size ($B$), Number of Units ($U$), Latent Dimension ($D$). 
* Impact: Linear. A tensor of shape (B, U, D) is extremely lightweight (e.g., $B=2000, U=26, D=10 \approx 2$ MB).

### 2. Receptor Physics (Combinatorics): 
$\mathcal{O}(B \cdot R \cdot k_{sub})$ 
* Variables: Batch Size ($B$), Number of Receptors ($R$), Subunits per receptor ($k_{sub}=5$). 
* Impact: Linear. Even for 10,000 receptors, the energy tensor takes $\approx 400$ MB. Very manageable.
### 3. Number of Families ($F$): 
$\mathcal{O}(F \cdot D)$ 
* Impact: Almost zero impact on training memory. Families only dictate the static coordinates of the environment. You could simulate $1,000,000$ families and it would only consume $\approx 40$ MB.
### 4. Loss Calculation (The Bottleneck)
The memory footprint heavily depends on which loss function strategy is utilized:

* **`DiscreteProxyLoss` (Scalable Marginals + Penalty):** $\mathcal{O}(B \cdot R) + \mathcal{O}(R^2)$
  * **Marginal Entropy:** A simple average over the batch. $\mathcal{O}(B \cdot R)$ ($\approx$ negligible).
  * **Penalty Tensor (Covariance/Repulsion):** Creates an $(R, R)$ pairwise matrix between receptors. This quadratic term $\mathcal{O}(R^2)$ is the primary memory bottleneck. If $R=10,000$, this matrix takes $\approx 400$ MB.
* **`DiscreteExactLoss` (True Joint Entropy):** $\mathcal{O}(B \cdot 2^R)$ or $\mathcal{O}(B \cdot M)$
  * **Small Arrays ($R \le 10$):** Uses exact state enumeration. Memory scales exponentially as $\mathcal{O}(B \cdot 2^R)$. For $R=10, B=2000$, it takes $\approx 8$ MB.
  * **Large Arrays ($R > 10$):** Dynamically switches to Monte Carlo estimation. Scales linearly as $(B, M)$, capped at $M=2048$ due to the subsampling trick. $\approx 16$ MB. *(Note: This loss completely avoids the $\mathcal{O}(R^2)$ penalty bottleneck!)*

### Summary: 
Maximum capacities on a standard 8GB-12GB GPU +Because everything scales either linearly or strictly quadratically for the number of receptors, you can comfortably run massive simulations: +* Max Latent Dimension ($D$): $\approx 1,000+$ 
* Max Number of Families ($F$): $\approx 100,000+$ +* Max Number of Units ($U$): $\approx 1,000+$ 
* Max Batch Size ($B$): $\approx 20,000$ 
* Max Number of Receptors ($R$): $\approx 10,000$ (Limited entirely by the $\mathcal{O}(R^2)$ covariance/repulsion matrix) 

Rule of Thumb: If you encounter a CUDA Out of Memory error, the first parameter to reduce is the Batch Size ($B$), followed by the Number of Receptors ($R$). The latent space dimension ($D$) and number of families ($F$) are virtually "free" to scale up!

# Tmux cheat sheet

### 📋 Tmux Quick Cheat Sheet

| Category | Action | Command / Shortcut |
| :--- | :--- | :--- |
| **Sessions** | **Start** a new session | `tmux` |
| | **Start** session with name | `tmux new -s <name>` |
| | **List** all sessions | `tmux ls` |
| | **Attach** to last session | `tmux attach` |
| | **Attach** to specific session | `tmux attach -t <name>` |
| | **Kill/Delete** a session | `tmux kill-session -t <name>` |
| **Inside Tmux** | **Detach** (Leave it running) | `Ctrl + b` then `d` |
| | **Scroll** mode (Arrows to move) | `Ctrl + b` then `[` (Press `q` to exit scroll) |
| | **New Window** | `Ctrl + b` then `c` |
| | **Switch Window** | `Ctrl + b` then `0-9` (The window number) |
| | **Split Vertically** | `Ctrl + b` then `%` |
| | **Split Horizontally** | `Ctrl + b` then `"` |
| | **Close** current pane/window | Type `exit` |

---

### Reference for Docker setup
Since you are using `docker-compose.server.yaml`, your workflow inside `tmux` should look like this:

1.  **Enter Tmux:** `tmux new -s gpu_task`
2.  [cite_start]**Start Docker:** `docker compose -f docker-compose.server.yaml run --rm gpu-runner python src/your_script.py` [cite: 1]
3.  **Leave Safely:** `Ctrl + b` then `d`.
4.  **Disconnect SSH.**
5.  **Return later:** `tmux attach -t gpu_task`.

[cite_start]**Note:** In your `Dockerfile`, you've set the user to `devuser` with UID `1000`, but your `docker-compose.server.yaml` overrides this by setting `user: "root"`[cite: 1, 2]. This means any files created by your script while running in `tmux` might be owned by **root** on your host machine.