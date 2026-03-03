# Fourier Neural Operator (FNO) Method Instructions

## Overview

The `fno/` directory implements a **Fourier Neural Operator** surrogate for HW2D plasma turbulence. The FNO learns a single-step state-to-state mapping in physical space:

$$u_{k+1} = \mathcal{F}_\theta(u_k)$$

where $u_k$ is the 2-channel (density, potential) 2D field at time $k$. Autoregressive rollout produces multi-step forecasts for Gamma prediction.

**Key libraries:** `torch`, `neuralop` (provides the `FNO` architecture).

---

## File Layout

| File | Purpose |
|------|---------|
| `step_1_train.py` | Train single-step FNO: predict next state from current state |
| `step_2_train.py` | Curriculum-based autoregressive fine-tuning with scheduled sampling |
| `step_3_evaluate.py` | Full autoregressive rollout, Gamma computation from predicted fields, metrics, plots |
| `utils.py` | PyTorch datasets, model save/load, training loops, rollout inference, metric computation |
| `config/` | YAML configs (`fno_temporal_split.yaml`, test variant) |
| `run_local.sh` | Local pipeline runner |

---

## Pipeline Details

### Step 1 — Single-Step Training

1. Loads HDF5 data via `h5py`, extracts `density` and `phi` fields as `(T, 2, H, W)` tensors
2. Creates `SingleStepDataset` — memory-efficient, stores data once, returns consecutive pairs `(u_k, u_{k+1})` on-the-fly
3. Trains FNO with MSE loss on single-step predictions
4. Uses cosine learning rate scheduler with warmup
5. Saves checkpoints every N epochs

**Memory optimization**: data stays as numpy arrays and is converted to tensors on-the-fly in `__getitem__`. This avoids doubling memory for input/target copies.

### Step 2 — Curriculum Autoregressive Training

The key innovation for stable long-horizon rollout. Curriculum stages gradually increase the rollout length while decreasing teacher forcing:

```yaml
curriculum:
  # [rollout_length, teacher_forcing_ratio, epochs]
  - [5,  0.8, 20]    # Short rollout, mostly teacher-forced
  - [10, 0.6, 20]    # Longer rollout
  - [20, 0.4, 20]    # Even longer
  - [40, 0.2, 20]    # Mostly autoregressive
  - [80, 0.0, 30]    # Fully autoregressive
```

**Teacher forcing ratio**: at each step during rollout, with probability `ratio` use the ground truth as input (teacher forcing), otherwise use the model's own prediction (autoregressive). Decreasing the ratio across stages trains the model to handle its own errors.

**Gradient checkpointing**: enabled via `torch.utils.checkpoint.checkpoint` to reduce memory during long rollouts. Without this, gradients through 80 autoregressive steps would exceed GPU memory.

### Step 3 — Evaluation

1. Loads the trained FNO checkpoint
2. Performs fully autoregressive rollout over the test portion
3. Splits predicted `(T, 2, H, W)` tensor into density and phi fields
4. **Computes Gamma from predicted fields** using the physics formulas from `shared/physics.py`:
   - $\Gamma_n = -\langle n \cdot \partial\phi/\partial y \rangle$ via periodic central differences
   - $\Gamma_c = c_1 \langle (n - \phi)^2 \rangle$
5. Compares against reference Gamma from HDF5 files
6. Generates plots via `shared/plotting.py`

---

## Config Structure

FNO currently uses a **plain dict** loaded from YAML (not a dataclass). This should eventually be migrated to a dataclass for consistency with OpInf/DMD.

```yaml
model:
  n_modes: [64, 64]         # Fourier modes to keep per spatial dimension
  in_channels: 2             # density + potential
  out_channels: 2
  hidden_channels: 128       # Width of FNO layers
  n_layers: 4                # Number of Fourier layers

training:
  batch_size: 1              # Often 1 due to large spatial grids
  n_epochs: 100
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  grad_clip: 1.0             # Gradient clipping norm
  noise_std: 0.01            # Input noise for regularization
  gradient_checkpointing: true
  scheduler:
    type: "cosine"
    warmup_epochs: 5
  save_every: 20
  eval_every: 10
```

Config values are accessed via `get_config_value(cfg, 'training', 'batch_size', default=1)` with safe nested access.

---

## Key Functions in `utils.py`

### Datasets
- `SingleStepDataset(states)` — returns consecutive `(u_k, u_{k+1})` pairs. Memory-efficient: stores numpy, converts on-the-fly.
- `StateTrajectoryDataset` — for multi-step rollout training
- `TrajectoryWithDerivedDataset` — includes derived quantities (Gamma) as targets

### Training Loops
- `train_epoch_single_step(model, loader, optimizer, ...)` — standard supervised training for Step 1
- `train_epoch_rollout(model, loader, optimizer, rollout_len, tf_ratio, ...)` — curriculum rollout training for Step 2. Implements scheduled sampling and optional gradient checkpointing.

### Inference
- `rollout_state(model, initial_state, n_steps)` — fully autoregressive evaluation rollout
- `rollout_combined(model, initial_state, n_steps)` — rollout returning both states and derived quantities

### Checkpointing
- `save_model(model, optimizer, epoch, loss, path)` — saves model state dict, optimizer, epoch, loss
- `load_model(model, path)` — loads checkpoint, returns epoch and loss
- `list_checkpoints(run_dir)` — finds all saved checkpoint files

---

## FNO Architecture Notes

The FNO from `neuralop` operates on 2D fields with `n_modes` Fourier modes per spatial dimension. Key parameters:

- **`n_modes: [64, 64]`** — number of Fourier modes in each spatial dimension. Higher = more spectral resolution but more parameters and memory.
- **`hidden_channels: 128`** — width of the spectral convolution layers
- **`n_layers: 4`** — depth of the FNO (number of Fourier + pointwise layers)
- **Input/output**: 2-channel (density, potential) 2D fields. The FNO predicts the full next-step state, not a residual.

---

## Gamma Computation

FNO computes Gamma **directly from predicted 2D fields** (no reduced-space intermediary). In `step_3_evaluate.py`:

1. Predicted `(T, 2, H, W)` tensor is split: `density = pred[:, 0]`, `phi = pred[:, 1]`
2. `compute_gamma_n(density, phi, dx)` and `compute_gamma_c(density, phi, c1)` from `shared/physics.py`
3. Grid parameters from `get_hw2d_grid_params(k0, nx)`

This is the most straightforward Gamma path — no projection, no output model. FNO predictions are already in physical space.

---

## Differences from ROM Methods

| Aspect | OpInf / DMD | FNO |
|--------|-------------|-----|
| Precision | `float64` default | `float32` (PyTorch default) |
| State space | Reduced (POD) | Full physical |
| Gamma path | Output model or reconstruct + physics | Direct physics on predicted fields |
| Parallelism | MPI | GPU (CUDA) |
| Config format | Dataclass (`OpInfConfig`) | Plain dict |
| Step 1 purpose | Preprocessing + POD | Single-step training |

---

## Common Issues

- **GPU memory**: at 512×512 × 2 channels, a batch of 1 is already large. Gradient checkpointing is essential for Step 2 rollout training. If memory is still tight, reduce `hidden_channels` or `n_modes`.
- **Input noise**: `noise_std: 0.01` adds Gaussian noise to inputs during training for regularization. This helps with autoregressive stability by making the model robust to small perturbations (its own errors).
- **Curriculum matters**: skipping the curriculum (going directly to long rollout) usually diverges. The gradual increase in rollout length is what makes FNO stable at 80+ autoregressive steps.
- **Float precision**: FNO uses `float32`. When comparing Gamma values against reference (which is `float64`), upcast predictions to `float64` before computing metrics.
- **No MPI**: FNO does not use MPI. It's purely GPU-based. Do not add MPI imports to FNO files.
