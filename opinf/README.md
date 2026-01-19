# Discrete-Time Operator Inference ROM Pipeline

This implements reduced-order modeling via Operator Inference (OpInf)
for learning dynamical system operators from snapshot data.

## Overview

The pipeline has three stages:

| Step | Script | Function | Parallelism |
|------|--------|----------|-------------|
| 1 | `step_1_preprocess.py` | Load data, compute basis, project snapshots | MPI |
| 2 | `step_2_train.py` | Train ROMs via hyperparameter sweep | MPI |  
| 3 | `step_3_evaluate.py` | Ensemble predictions and metrics | Serial |

## Dimensionality Reduction Methods

Two reduction methods are supported (set via `reduction.method` in config):

### Linear POD (`method: "linear"`)
Standard POD via distributed Gram matrix eigendecomposition:
```
x ≈ V @ z + μ
```
where `V ∈ ℝⁿˣʳ` are the POD modes.

### Quadratic Manifold (`method: "manifold"`)
Nonlinear reduction via greedy mode selection:
```
x ≈ V @ z + W @ h(z) + μ
```
where `h(z)` are quadratic features `z_i * z_j`.

#### Manifold-Aware Training (New)

When using the manifold method, you can enable **manifold-aware training** which
makes the ROM operators respect the manifold structure during optimization:

1. **Consistency Loss**: Penalizes predictions that don't decode well in full space.
   The ROM is trained to stay on the learned manifold by adding:
   ```
   ||z_pred - encode(decode(z_pred))||
   ```
   to the loss function.

2. **Re-encode for Output**: Before computing output quantities (like Gamma),
   the predicted reduced state is decoded to full space and re-encoded:
   ```
   z_corrected = V.T @ (V @ z + W @ h(z))
   ```
   This adds a quadratic correction `V.T @ W @ h(z)` that better captures
   nonlinear output dependence on the modes.

Enable via config:
```yaml
reduction:
  method: "manifold"
  manifold_aware_training: true      # Use consistency loss
  manifold_consistency_weight: 1.0   # Weight for consistency term
  manifold_reencode_output: true     # Decode→re-encode for output
```

Both methods report **reconstruction error** so you can compare them directly.

## ROM Learning

We learn discrete-time operators from a regression:

```
z_{k+1} = A @ z_k + F @ z_k^{(2)}
```

where:
- `z_k ∈ ℝʳ` are reduced coordinates
- `z_k^{(2)}` are non-redundant quadratic terms, dimension `r(r+1)/2`
- `A ∈ ℝʳˣʳ` is the linear state operator
- `F ∈ ℝʳˣʳ⁽ʳ⁺¹⁾ᐟ²` is the quadratic operator

The operators are learned via Tikhonov-regularized least squares.

## Usage

### 1. Create a configuration file

Copy and modify `config/example.yaml`:

```yaml
reduction:
  method: "linear"     # or "manifold"
  r: 75
  target_energy: 0.9999
  
  # For manifold method only:
  n_vectors_to_check: 200
  reg_magnitude: 1.0e-6

paths:
  data_dir: /path/to/simulation/data
  output_base: /path/to/output

physics:
  dt: 0.025
  n_fields: 2
  n_x: 256
  n_y: 256
```

### 2. Run the pipeline

**On HPC (SLURM):**
```bash
sbatch run_opinf.slurm /path/to/config.yaml
```

**Locally:**
```bash
./run_local.sh config/example.yaml 4
```

## Module Structure

```
opinf/
├── step_1_preprocess.py  # Data loading + dimensionality reduction
├── step_2_train.py       # Hyperparameter sweep
├── step_3_evaluate.py    # Ensemble evaluation
├── utils.py              # Configuration, logging, helpers
├── data.py               # Data I/O and manipulation
├── pod.py                # POD + quadratic manifold algorithms
├── training.py           # OpInf fitting and model selection
├── evaluation.py         # Time-stepping and metrics
├── core.py               # Operator inference core algorithm
├── config/               # Example configurations
├── run_local.sh          # Local execution script
└── run_opinf.slurm       # SLURM job script
```

## Key Outputs

After running the pipeline, `run_dir/` contains:

- `pod_basis.npy` or `pod_basis_basis.npz`: Learned basis
- `Xhat_train.npy`, `Xhat_test.npy`: Projected data
- `ensemble_results.npz`: Selected model operators
- `predictions.npz`: Time series predictions
- `preprocessing_info.npz`: Includes `reduction_method` and reconstruction error
