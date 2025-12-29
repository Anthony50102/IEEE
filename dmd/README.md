# DMD Pipeline for Hasegawa-Wakatani 2D Data

This module implements **Optimized DMD (opt-DMD)** using the BOPDMD algorithm from PyDMD for reduced-order modeling of Hasegawa-Wakatani 2D turbulence data.

## Overview

The DMD pipeline fits a linear dynamical system in the POD-reduced space:

$$\frac{d\hat{x}}{dt} = A\hat{x}$$

where $\hat{x}$ is the POD-projected state. DMD identifies eigenvalues and modes of this system, enabling time-extrapolation (forecasting).

### Physics-Based Gamma Computation

Instead of learning an output operator, this implementation computes the transport fluxes **directly from the reconstructed density and potential fields** using the physics-based formulas:

**Particle flux:**
$$\Gamma_n = -\int d^2x \, \tilde{n} \frac{\partial\tilde{\phi}}{\partial y}$$

**Conductive flux:**
$$\Gamma_c = c_1 \int d^2x \, (\tilde{n} - \tilde{\phi})^2$$

where:
- $\tilde{n}$ is the density fluctuation
- $\tilde{\phi}$ is the potential fluctuation  
- $c_1$ is the adiabaticity parameter (typically 1.0)
- Grid spacing: $dx = 2\pi/k_0$ where $k_0 = 0.15$

## Pipeline Steps

### Step 1: Preprocessing (Shared with OpInf)
Uses the existing `opinf/step_1_preprocess.py` or `step_1_parallel_preprocess.py` to:
- Load and stack training/test data
- Compute POD basis via method of snapshots
- Project data to reduced coordinates
- Save initial conditions and boundaries

### Step 1.5: Save POD Basis
```bash
python dmd/save_pod_basis.py --config config/local_dmd_1train_5test.yaml --run-dir <run_dir>
```
Computes and saves the spatial POD basis U_r needed for full-state reconstruction. This is required for physics-based Gamma computation.

### Step 2: Fit DMD Model
```bash
python dmd/step_2_fit_dmd.py --config config/local_dmd_1train_5test.yaml --run-dir <run_dir>
```

Fits BOPDMD to the POD-projected training data:
1. Load projected training data from Step 1
2. Fit BOPDMD model (optionally with bagging)
3. Extract eigenvalues, modes, and amplitudes
4. Save DMD model

### Step 3: Evaluate Predictions
```bash
python dmd/step_3_evaluate_dmd.py --config config/local_dmd_1train_5test.yaml --run-dir <run_dir>
```

Evaluates DMD predictions:
1. Load DMD model and POD basis
2. Compute forecasts for training and test trajectories
3. **Reconstruct full state** from reduced state: $Q = U_r \hat{x} + \bar{Q}$
4. **Compute Gamma from physics formulas**
5. Compute error metrics
6. Generate diagnostic plots

## Configuration

The DMD configuration extends the OpInf config with a `dmd` section:

```yaml
dmd:
  rank: null                    # DMD rank (null = use POD r)
  num_trials: 0                 # Bagging trials (0 = no bagging)
  use_proj: true                # Use projection in BOPDMD
  eig_sort: "real"              # Eigenvalue sorting
  k0: 0.15                      # Wavenumber for grid spacing
  c1: 1.0                       # Adiabaticity parameter
```

## Full Pipeline Script

For local runs:
```bash
./scripts/local_dmd_full_pipeline.sh [run_name]
```

This runs all steps sequentially: preprocessing → save POD basis → fit DMD → evaluate.

## Output Files

After running the pipeline, the run directory contains:

```
<run_dir>/
├── POD.npz                     # POD eigenvalues/vectors from Step 1
├── Xhat_train.npy              # Projected training data
├── Xhat_test.npy               # Projected test data  
├── pod_basis.npz               # POD basis U_r and mean (Step 1.5)
├── dmd_model.npz               # DMD eigenvalues, modes, amplitudes
├── dmd_eigenvalues.npy         # Continuous-time eigenvalues
├── dmd_modes.npy               # Reduced-space DMD modes
├── dmd_amplitudes.npy          # DMD mode amplitudes
├── dmd_evaluation_metrics.yaml # Evaluation results
├── dmd_forecasts/              # Saved predictions
│   ├── train_traj_0_Xhat.npy
│   ├── train_traj_0_gamma.npz
│   ├── test_traj_0_Xhat.npy
│   └── test_traj_0_gamma.npz
└── figures/                    # Diagnostic plots
    ├── dmd_eigenvalues.png
    ├── train_traj_1_gamma.png
    └── test_traj_1_gamma.png
```

## Dependencies

- numpy
- scipy
- pydmd (for BOPDMD)
- xarray (for HDF5 loading)
- h5netcdf (HDF5 engine)
- matplotlib (optional, for plotting)
- yaml

## Mathematical Details

### DMD Eigenvalues

The continuous-time DMD eigenvalues $\alpha_j$ relate to the discrete-time eigenvalues $\mu_j$ by:
$$\mu_j = e^{\alpha_j \Delta t}$$

Eigenvalues with $\text{Re}(\alpha_j) < 0$ are stable (decaying modes).
Eigenvalues with $\text{Re}(\alpha_j) > 0$ are unstable (growing modes).

### Forecasting

The reduced state at time $t$ is:
$$\hat{x}(t) = \sum_j b_j \phi_j e^{\alpha_j t}$$

where:
- $\alpha_j$ are DMD eigenvalues
- $\phi_j$ are DMD modes (in reduced space)
- $b_j$ are amplitudes computed from initial condition

### Full State Reconstruction

To compute physics-based Gamma, we reconstruct:
$$Q(t) = U_r \hat{x}(t) + \bar{Q}$$

where $U_r$ is the truncated POD basis and $\bar{Q}$ is the temporal mean.

## References

- Askham & Kutz (2018): "Variable Projection Methods for an Optimized DMD"
- PyDMD documentation: https://mathlab.github.io/PyDMD/
