# DMD (Dynamic Mode Decomposition) ROM Pipeline

This module implements Optimized DMD (opt-DMD via BOPDMD) for reduced-order modeling of the Hasegawa-Wakatani plasma turbulence system.

## Overview

The DMD pipeline fits a linear dynamical system to the training data in reduced (POD) coordinates, then uses the learned eigenvalues and modes to forecast new trajectories from different initial conditions.

**Key features:**
- Uses BOPDMD (Bagging/Optimized DMD) from the `pydmd` library
- Reuses Step 1 preprocessing from the OpInf pipeline (POD computation)
- Learns a linear output operator for Gamma (particle flux, conductive flux) prediction
- Supports forecasting on multiple test trajectories

## Pipeline Structure

```
Step 1: Preprocessing (shared with OpInf)
    ↓
    opinf/step_1_parallel_preprocess.py
    - Computes POD basis
    - Projects training/test data
    - Saves initial conditions and boundaries
    ↓
Step 2: Fit DMD Model
    ↓
    dmd/step_2_fit_dmd.py
    - Loads POD basis and projected data
    - Fits BOPDMD to training trajectory
    - Learns output operator (optional)
    ↓
Step 3: Evaluate/Forecast
    ↓
    dmd/step_3_evaluate_dmd.py
    - Computes forecasts for training/test trajectories
    - Computes evaluation metrics
    - Generates diagnostic plots
```

## Usage

### 1. Run Preprocessing (Step 1)

First, run the shared preprocessing step to compute POD basis:

```bash
# Serial execution (for testing)
python opinf/step_1_preprocess.py --config config/local_dmd_1train_5test.yaml

# Parallel execution (for HPC)
mpirun -np 4 python opinf/step_1_parallel_preprocess.py --config config/dmd_1train_5test.yaml
```

Note the run directory that is created (e.g., `output/20250101_120000_dmd_1train_5test/`).

### 2. Fit DMD Model (Step 2)

```bash
python dmd/step_2_fit_dmd.py \
    --config config/local_dmd_1train_5test.yaml \
    --run-dir /path/to/run/directory
```

This will:
- Load the POD basis from Step 1
- Fit BOPDMD to the training trajectory
- Learn a linear output operator for Gamma prediction
- Save the model components (eigenvalues, modes, amplitudes)

### 3. Evaluate/Forecast (Step 3)

```bash
python dmd/step_3_evaluate_dmd.py \
    --config config/local_dmd_1train_5test.yaml \
    --run-dir /path/to/run/directory
```

This will:
- Load the fitted DMD model
- Compute forecasts for training and test trajectories
- Compute evaluation metrics (mean error, std error, RMSE)
- Generate diagnostic plots

## Configuration

The DMD pipeline uses the same configuration structure as OpInf, with an additional `dmd` section:

```yaml
# DMD-specific settings
dmd:
  rank: null                    # DMD rank (null = use same as POD r)
  num_trials: 0                 # Number of bagging trials (0 = standard BOPDMD)
  use_proj: true                # Use POD projection in BOPDMD
  eig_sort: "real"              # Eigenvalue sorting: "real", "imag", "abs"
  learn_output_operator: true   # Learn linear output operator for Gamma
  output_reg: 1.0e-6            # Regularization for output operator
```

### Configuration Options

| Parameter | Description |
|-----------|-------------|
| `dmd.rank` | Number of DMD modes. If `null`, uses the POD rank from `pod.r` |
| `dmd.num_trials` | Number of bagging trials for ensemble DMD. 0 = standard BOPDMD |
| `dmd.use_proj` | Use POD projection in BOPDMD fitting |
| `dmd.eig_sort` | How to sort eigenvalues: "real" (by growth rate), "imag" (by frequency), "abs" (by magnitude) |
| `dmd.learn_output_operator` | Learn a linear mapping from reduced state to Gamma |
| `dmd.output_reg` | Tikhonov regularization for output operator learning |

## Output Files

After running all steps, the run directory will contain:

```
run_directory/
├── POD.npz                     # POD decomposition (from Step 1)
├── X_hat_train.npy             # Projected training data
├── X_hat_test.npy              # Projected test data
├── data_boundaries.npz         # Trajectory boundaries
├── initial_conditions.npz      # ICs in full and reduced space
├── gamma_reference.npz         # Reference Gamma values
├── learning_matrices.npz       # Data matrices for operator learning
├── preprocessing_info.npz      # Preprocessing metadata
├── dmd_model.npz               # Full DMD model
├── dmd_eigenvalues.npy         # DMD eigenvalues (continuous-time)
├── dmd_modes.npy               # DMD modes in reduced space
├── dmd_amplitudes.npy          # DMD amplitudes
├── dmd_output_operator.npz     # Output operator (C, c)
├── dmd_forecasts/              # Saved forecasts
│   ├── train_traj_0_Xhat.npy
│   ├── train_traj_0_gamma.npz
│   ├── test_traj_0_Xhat.npy
│   └── ...
├── dmd_evaluation_metrics.yaml # Evaluation metrics
├── figures/                    # Diagnostic plots
│   ├── dmd_eigenvalues.png
│   ├── train_traj_1_gamma.png
│   └── test_traj_1_gamma.png
└── pipeline_status.yaml        # Step completion status
```

## Dependencies

Required Python packages:
- numpy
- scipy
- pydmd (for BOPDMD)
- xarray (for data loading)
- h5netcdf (HDF5 backend)
- matplotlib (for plotting)
- PyYAML (for configuration)

Install pydmd with:
```bash
pip install pydmd
```

## Mathematical Background

### DMD Formulation

DMD approximates dynamics as a linear system:
```
x_{k+1} = A x_k
```

The DMD algorithm finds eigenvalues λ and modes φ such that:
```
x(t) = Σ_j b_j φ_j exp(α_j t)
```

Where:
- α_j = log(λ_j) / dt are continuous-time eigenvalues
- φ_j are DMD modes
- b_j are amplitudes (initial conditions in mode space)

### BOPDMD

BOPDMD (Bagging, Optimized, and Projected DMD) improves on standard DMD by:
1. Optimizing eigenvalues to minimize reconstruction error
2. Projecting onto a pre-computed basis (POD) for noise reduction
3. Optional bagging for uncertainty quantification

### Output Operator

For predicting Gamma from reduced state, we learn a linear operator:
```
y = C @ (x_hat_scaled) + c
```

Where x_hat_scaled = (x_hat - mean) / scale.

## Comparison with OpInf

| Feature | DMD | OpInf |
|---------|-----|-------|
| Model form | Linear | Quadratic |
| Stability | No guarantee | Depends on regularization |
| Training | Single SVD | Regression with regularization sweep |
| Generalization | Limited (linear dynamics) | Better for nonlinear systems |
| Computational cost | Low | Higher (hyperparameter sweep) |

## References

1. Schmid, P. J. (2010). Dynamic mode decomposition of numerical and experimental data. JFM.
2. Askham, T. & Kutz, J. N. (2018). Variable projection methods for an optimized DMD. SIADS.
3. Peherstorfer, B. & Willcox, K. (2016). Data-driven operator inference for non-intrusive projection-based model reduction. CMAME.
