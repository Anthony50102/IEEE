# Operator Inference (OpInf) Method Instructions

## Overview

The `opinf/` directory implements **Operator Inference** — a non-intrusive, data-driven ROM that learns discrete-time quadratic operators from snapshot data:

$$z_{k+1} = Az_k + F z_k^{(2)}$$

where $A \in \mathbb{R}^{r \times r}$ is the linear operator, $F \in \mathbb{R}^{r \times r(r+1)/2}$ is the quadratic operator, and $z_k^{(2)}$ contains unique quadratic products. Operators are learned via Tikhonov-regularized least squares.

A **quadratic output model** maps reduced states to transport coefficients: $\Gamma \approx Cz + Gz^{(2)} + c$.

This is the most MPI-intensive method in the repo. Steps 1 and 2 use distributed computation.

---

## File Layout

| File | Purpose |
|------|---------|
| `step_1_preprocess.py` | **MPI-parallel** data loading, distributed Gram matrix, POD or quadratic manifold basis, projection |
| `step_2_train.py` | **MPI-parallel** hyperparameter sweep over 4D regularization grid, model selection |
| `step_3_evaluate.py` | Serial ensemble prediction, Gamma computation, metrics, plots |
| `core.py` | Quadratic term generation, discrete-time integration, regularized solve |
| `pod.py` | Distributed POD (Sirovich method), quadratic manifold via greedy mode selection, `BasisData` container |
| `training.py` | Parallel hyperparameter sweep, model selection logic, operator computation |
| `evaluation.py` | Trajectory prediction, ensemble predictions (mean ± std), metrics |
| `data.py` | Distributed data loading, centering/scaling, MPI shared-memory arrays |
| `utils.py` | `OpInfConfig` dataclass, run management, logging, MPI wrappers, pipeline status |
| `config/` | YAML configs (example, manifold variants, temporal split variants) |
| `run_local.sh` | Local pipeline runner |

---

## Pipeline Details

### Step 1 — Preprocessing (MPI-parallel)

1. **Distributed data loading**: each MPI rank loads a slice of rows from HDF5 files
2. **Centering**: distributed mean subtraction (allreduce for global mean)
3. **Optional scaling**: per-field normalization
4. **Dimensionality reduction** — two methods:
   - **Linear POD** (`method: "linear"`): distributed Gram matrix eigendecomposition (Sirovich's method — cheaper when `n_time << n_spatial`)
   - **Quadratic manifold** (`method: "manifold"`): `x ≈ Vz + Wh(z) + μ` with greedy mode selection. Requires gathering full data to rank 0 for the greedy search.
5. **Project data**: distributed `Xhat = U_r^T @ Q`
6. **Prepare learning matrices**: assemble `D = [Z^(k) ; Z^(k)(2)]` and `R = Z^(k+1)` for least-squares

**Important**: Uses `chunked_gather` and `chunked_bcast` from shared/MPI utilities to avoid the MPI 32-bit message size limit (~2GB). This is critical for large grids (512×512 × 2 fields = ~1M rows).

### Step 2 — Training (MPI-parallel)

1. **Load data via shared memory**: uses `MPI.Win.Allocate_shared` for node-local shared arrays. Only one rank per node allocates; others attach.
2. **4D hyperparameter sweep**: regularization parameters `(α_state_lin, α_state_quad, α_output_lin, α_output_quad)` are expanded from `{min, max, num, scale}` grids. Total combinations = `num₁ × num₂ × num₃ × num₄`.
3. **Distributed sweep**: each rank gets a chunk of parameter combinations, fits operators, evaluates training error
4. **Model selection**: threshold-based (`threshold_mean`, `threshold_std`) filtering, then sort by total error
5. **Recompute operators**: selected models have their operators recomputed in parallel and saved to disk

**At HPC scale**: 16 nodes × 56 cores = 896 MPI ranks sweeping over ~10,000 parameter combinations. Each fit is a regularized least-squares solve — fast per-combination but massive in aggregate.

### Step 3 — Evaluation (serial)

1. Loads ensemble of selected models
2. For each test trajectory: time-steps each model forward via `z_{k+1} = Az_k + Fz_k^(2)`
3. Computes ensemble mean and standard deviation of Gamma predictions
4. Generates plots using `shared/plotting.py` (supports ±2σ uncertainty bands)
5. Computes metrics: relative errors in mean/std of Gamma_n and Gamma_c

---

## Config Structure (`OpInfConfig`)

`OpInfConfig` is the **base config dataclass** used by both OpInf and DMD (DMD extends it).

Key sections:

```yaml
training_mode:
  mode: "multi_trajectory"    # or "temporal_split"
  train_start: 8000
  train_end: 16000
  test_start: 16000
  test_end: 24000

reduction:
  method: "linear"            # or "manifold"
  r: 75                       # POD truncation rank
  target_energy: 0.9999       # Energy threshold for automatic rank
  n_vectors_to_check: 200     # Greedy manifold: candidates per iteration
  reg_magnitude: 1.0e-6       # Greedy manifold: regression regularization

regularization:
  state_lin:  {min: 1.0e0,  max: 1.0e7,  num: 10, scale: "log"}
  state_quad: {min: 1.0e4,  max: 1.0e12, num: 10, scale: "log"}
  output_lin: {min: 1.0e-2, max: 1.0e4,  num: 10, scale: "log"}
  output_quad:{min: 1.0e-2, max: 1.0e6,  num: 10, scale: "log"}

model_selection:
  threshold_mean: 0.05        # Max relative error in Gamma mean
  threshold_std: 0.20         # Max relative error in Gamma std
```

Regularization grids are expanded by `_build_reg_array()` in `utils.py`: `"log"` scale uses `np.logspace`, `"linear"` uses `np.linspace`.

---

## Key Functions

### `core.py`
- `get_quadratic_terms(z)` — upper-triangular products: `z_i * z_j` for `i ≤ j`. Returns vector of length `r(r+1)/2`.
- `tikhonov_solve(D, R, alphas)` — regularized least squares: `(D^T D + diag(α)) \ (D^T R)`. Uses Cholesky or fallback to `np.linalg.solve`.
- `step_forward(z, A, F)` — single discrete-time step: `z_next = A @ z + F @ quad(z)`
- `integrate_model(z0, A, F, n_steps)` — full rollout with NaN/Inf divergence checking

### `pod.py`
- `compute_pod_distributed(Q_local, comm, ...)` — Sirovich method: form `G = Q^T Q` distributedly, eigendecompose, return modes
- `project_data_distributed(...)` — distributed projection onto POD basis
- `compute_manifold_greedy(Q, r, n_check, reg, logger)` — greedy selection of POD modes maximizing reconstruction accuracy under the quadratic manifold `x ≈ Vz + Wh(z) + μ`
- `BasisData` — container for basis type, `V` (linear), `W` (quadratic, if manifold), shift, rank, eigenvalues, selected indices
- `encode(Q, basis)` / `decode(z, basis)` — project to/from reduced space (handles both linear and manifold)

### `training.py`
- `parallel_hyperparameter_sweep(cfg, data, logger, comm)` — main distributed sweep loop
- `select_models(results, thresh_mean, thresh_std, logger)` — filter + sort
- `recompute_operators_parallel(selected, data, r, out_dir, comm, logger)` — regenerate operators for selected models

### `evaluation.py`
- `predict_trajectory(model, z0, n_steps)` — rollout one model
- `ensemble_predict(models, z0, n_steps)` — run all models, return mean ± std for Gamma
- `compute_gamma_from_reduced(z_traj, output_model)` — apply C, G, c to reduced trajectory

### `data.py`
- `load_all_data_distributed(cfg, run_dir, comm, ...)` — each rank loads its row-slice of HDF5
- `center_data_distributed(Q_local, comm, ...)` — distributed mean subtraction
- `load_data_shared_memory(paths, comm, logger)` — MPI shared-memory arrays for Step 2
- `save_ensemble(models, path, cfg, logger)` — serialize selected models

---

## Two Reduction Methods

### Linear POD
Standard Proper Orthogonal Decomposition: `x ≈ Vz + μ` where `V` is the matrix of leading `r` POD modes.

### Quadratic Manifold
Nonlinear extension: `x ≈ Vz + Wh(z) + μ` where `W` maps quadratic products of reduced coordinates back to the high-dimensional space. Uses a greedy algorithm to select which POD modes to include in `V`, optimizing for reconstruction under the manifold ansatz. This can capture more variance with fewer modes but is more expensive to compute.

Set `reduction.method: "manifold"` in config. Relevant parameters: `n_vectors_to_check` (candidates per greedy iteration), `reg_magnitude` (Tikhonov for the manifold regression).

---

## MPI Considerations

- **Step 1** distributes data rows across ranks (each rank holds `n_spatial/n_ranks` rows)
- **Step 2** distributes hyperparameter combinations across ranks
- **Step 3** runs serially (single rank)
- Always lazy-import `mpi4py` — see the critical rule in `copilot-instructions.md`
- `MPI.Win.Allocate_shared` is used in Step 2 for zero-copy data sharing within a node
- `chunked_gather` / `chunked_bcast` split messages to stay under MPI's ~2GB limit
- MPI window handles are stored in a list and freed in a `finally` block — never skip this cleanup

---

## Common Issues

- **Sweep produces no valid models**: loosen `threshold_mean` / `threshold_std`, or expand the regularization grid range
- **Memory on shared-memory allocation**: `MPI.Win.Allocate_shared` fails if the node doesn't have enough RAM for the full learning matrices. Reduce `n_steps` or the grid to mitigate.
- **Divergent rollouts**: the discrete-time integration can diverge (NaN/Inf). The code checks at each step and truncates. This is expected for some regularization values — that's what the sweep filters out.
- **Manifold computation**: gathering full data to rank 0 for the greedy search limits scalability. For very large grids, consider running the manifold on a high-memory node.
