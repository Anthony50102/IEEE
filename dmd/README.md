# DMD Pipeline for Hasegawa-Wakatani 2D Data

This module implements **Optimized DMD (opt-DMD)** using the BOPDMD algorithm from PyDMD for reduced-order modeling of Hasegawa-Wakatani 2D turbulence data.

## Overview

The DMD pipeline fits a linear dynamical system to snapshot data:

$$\frac{d\hat{x}}{dt} = A\hat{x}$$

DMD identifies eigenvalues and modes, enabling time-extrapolation (forecasting).

The pipeline computes a POD basis first, then fits DMD in the reduced space. This provides regularization and reduces computational cost for large datasets.

## Pipeline Structure

| Step | Script | Function |
|------|--------|----------|
| 1 | `step_1_preprocess.py` | Load data, compute POD, prepare training data |
| 2 | `step_2_train.py` | Fit BOPDMD model |
| 3 | `step_3_evaluate.py` | Compute predictions and metrics |

## Training Modes

Two training modes are supported via the `training_mode` config option:

### Multi-Trajectory (`training_mode: "multi_trajectory"`)
Train on complete trajectory(ies), test on trajectories with different initial conditions.

```yaml
dmd:
  training_mode: "multi_trajectory"

paths:
  training_files: [train1.h5]
  test_files: [test1.h5, test2.h5]
```

### Temporal Split (`training_mode: "temporal_split"`)
Train on the first n snapshots of a single trajectory, predict the remaining portion.

```yaml
dmd:
  training_mode: "temporal_split"
  train_start: 0
  train_end: 2000
  test_start: 2000
  test_end: 4000

paths:
  training_files: [trajectory.h5]
  test_files: []
```

## Physics-Based Gamma Computation

Transport fluxes are computed directly from reconstructed fields:

**Particle flux:**
$$\Gamma_n = -\int d^2x \, \tilde{n} \frac{\partial\tilde{\phi}}{\partial y}$$

**Conductive flux:**
$$\Gamma_c = c_1 \int d^2x \, (\tilde{n} - \tilde{\phi})^2$$

## Quick Start

```bash
python dmd/step_1_preprocess.py --config config/dmd_temporal_split.yaml
python dmd/step_2_train.py --config config/dmd_temporal_split.yaml --run-dir <run_dir>
python dmd/step_3_evaluate.py --config config/dmd_temporal_split.yaml --run-dir <run_dir>
```

## Configuration

```yaml
dmd:
  training_mode: "temporal_split"  # or "multi_trajectory"
  train_start: 0
  train_end: 2000
  test_start: 2000
  test_end: 4000
  
  rank: null                       # DMD rank (null = use POD r)
  num_trials: 0                    # Bagging trials (0 = no bagging)
  use_proj: true                   # Use projection in BOPDMD
  eig_sort: "real"                 # Eigenvalue sorting
  k0: 0.15                         # Wavenumber for grid spacing
  c1: 1.0                          # Adiabaticity parameter

reduction:
  r: 75                            # POD rank
```

## Module Structure

```
dmd/
├── step_1_preprocess.py  # Data loading + POD computation
├── step_2_train.py       # BOPDMD fitting
├── step_3_evaluate.py    # Prediction and metrics
├── utils.py              # Configuration, forecasting utilities
├── data.py               # Data I/O and POD computation
└── README.md
```

## Output Files

```
<run_dir>/
├── pod_basis.npz         # POD basis U_r and mean
├── POD.npz               # Singular values
├── Xhat_train.npy        # Training data (projected)
├── Xhat_test.npy         # Test data (projected)
├── dmd_model.npz         # DMD eigenvalues, modes, amplitudes
├── dmd_predictions.npz   # Forecasted Gamma values
└── dmd_evaluation_metrics.yaml
```

## Mathematical Details

### DMD Eigenvalues

Continuous-time eigenvalues $\alpha_j$ relate to discrete-time by: $\mu_j = e^{\alpha_j \Delta t}$

- $\text{Re}(\alpha_j) < 0$: stable (decaying)
- $\text{Re}(\alpha_j) > 0$: unstable (growing)

### Forecasting

The state at time $t$:
$$\hat{x}(t) = \sum_j b_j \phi_j e^{\alpha_j t}$$

### Full State Reconstruction

**POD mode:**
$$Q(t) = U_r \hat{x}(t) + \bar{Q}$$

**Raw mode:**
$$Q(t) = \hat{x}(t) + \bar{Q}$$

where $\hat{x}(t)$ is already in full state space.

## References

- Askham & Kutz (2018): "Variable Projection Methods for an Optimized DMD"
- PyDMD documentation: https://mathlab.github.io/PyDMD/
