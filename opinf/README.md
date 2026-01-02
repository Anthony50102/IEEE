# Discrete-Time Operator Inference ROM Pipeline

This implements reduced-order modeling via Operator Inference (OpInf)
for learning dynamical system operators from snapshot data.

## Overview

The pipeline has three stages:

| Step | Script | Function | Parallelism |
|------|--------|----------|-------------|
| 1 | `step_1_preprocess.py` | Load data, compute POD basis, project snapshots | MPI |
| 2 | `step_2_train.py` | Train ROMs via hyperparameter sweep | MPI |  
| 3 | `step_3_evaluate.py` | Ensemble predictions and metrics | Serial |

## Method

We learn discrete-time operators from a regression:

```
x_{k+1} = A @ x_k + F @ x_k^{(2)}
```

where:
- `x_k ∈ ℝʳ` are reduced coordinates (POD coefficients)
- `x_k^{(2)}` are non-redundant quadratic terms, dimension `r(r+1)/2`
- `A ∈ ℝʳˣʳ` is the linear state operator
- `F ∈ ℝʳˣʳ⁽ʳ⁺¹⁾ᐟ²` is the quadratic operator

The operators are learned via Tikhonov-regularized least squares:

```
min ||D @ [A; F]ᵀ - Y||²_F + λ_A ||A||²_F + λ_F ||F||²_F
```

## Usage

### 1. Create a configuration file

Copy and modify `config/example.yaml`:

```yaml
paths:
  data_dir: /path/to/simulation/data
  base_output_dir: /path/to/output

physics:
  time_step: 0.025
  variables: [density, vorticity]

pod:
  total_modes: 100

truncation:
  r_linear: [20, 40, 60, 80, 100]
  r_quadratic: [10, 20, 30, 40, 50]

training:
  lambda_linear: [1e-8, 1e-6, 1e-4, 1e-2, 1]
  lambda_quadratic: [1e-8, 1e-6, 1e-4, 1e-2, 1]
```

### 2. Run the pipeline

**On HPC (SLURM):**
```bash
# Run all steps
sbatch run_opinf.slurm all /path/to/config.yaml

# Run individual steps  
sbatch run_opinf.slurm 1 /path/to/config.yaml
sbatch run_opinf.slurm 2 /path/to/config.yaml
sbatch run_opinf.slurm 3 /path/to/config.yaml
```

**Locally:**
```bash
# Run all steps with 4 MPI ranks
./run_local.sh all config.yaml 4

# Run individual steps
./run_local.sh 1 config.yaml 4
./run_local.sh 2 config.yaml 4  
./run_local.sh 3 config.yaml
```

### 3. Outputs

Each run creates a timestamped directory containing:

```
output/opinf_run_20240101_120000/
├── pod_basis.h5           # POD basis and singular values (Step 1)
├── projected_data.h5      # Reduced coordinates (Step 1)
├── learning_matrices.h5   # D and Y matrices (Step 1)
├── hyperparameter_results.h5  # All sweep results (Step 2)
├── best_models.h5         # Selected operators (Step 2)
├── predictions.h5         # Ensemble predictions (Step 3)
├── metrics.json           # Error metrics (Step 3)
├── figures/               # Diagnostic plots
└── config.yaml            # Configuration used
```

## Module Structure

```
opinf/
├── __init__.py          
├── core.py              # Core OpInf algorithms
├── utils.py             # Configuration and utilities
├── step_1_preprocess.py # Data loading and POD
├── step_2_train.py      # Hyperparameter sweep
├── step_3_evaluate.py   # Evaluation and plotting
├── run_opinf.slurm      # HPC launcher
├── run_local.sh         # Local launcher
└── config/
    └── example.yaml     # Configuration template
```

### core.py

Core mathematical operations:

- `get_quadratic_terms(q)`: Compute non-redundant quadratic terms
- `solve_difference_model(q, dt)`: Prepare learning matrices D, Y
- `solve_opinf_operators(D, Y, ...)`: Solve regularized regression
- `build_data_matrix(q)`: Build data matrix with linear + quadratic terms

### utils.py

Configuration and utilities:

- `OpInfConfig`: Dataclass for all configuration options
- `load_config() / save_config()`: YAML I/O
- `setup_logging()`: Configure logging with rank awareness
- MPI utilities: `distribute_indices()`, `chunked_bcast()`, `create_shared_array()`

## Diagnostic Logging

Set `debug: true` in config or use environment variable:

```bash
export OPINF_DEBUG=1
mpirun -n 4 python step_1_preprocess.py config.yaml
```

This enables detailed logging including:
- Timing for each phase
- Memory usage per rank
- Intermediate matrix dimensions
- Communication diagnostics
