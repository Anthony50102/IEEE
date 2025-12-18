# OpInf ROM Pipeline

A modular, configurable pipeline for training Operator Inference (OpInf) 
reduced-order models for the Hasegawa-Wakatani equations.

## Overview

This pipeline replaces the monolithic Jupyter notebook workflow with a clean,
reproducible set of Python scripts. Each run creates a timestamped directory
containing all outputs and a copy of the configuration used.

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step_1_preprocess.py` | Load data, compute POD basis, project data |
| 2 | `step_2_train_rom.py` | Hyperparameter sweep to find best ROM operators |
| 3 | `step_3_evaluate.py` | Evaluate ensemble on train/test data |
| All | `run_pipeline.py` | Run all steps end-to-end |

## Quick Start

### 1. Create a configuration file

Copy and modify the example configuration:

```bash
cp config_example.yaml my_config.yaml
# Edit my_config.yaml with your paths and parameters
```

### 2. Run the full pipeline

**Serial execution (for testing):**
```bash
python run_pipeline.py --config my_config.yaml
```

**Parallel execution (HPC):**
```bash
mpirun -np 56 python run_pipeline.py --config my_config.yaml
```

### 3. Run individual steps

If you want to run steps separately:

```bash
# Step 1: Preprocessing (always serial)
python step_1_preprocess.py --config my_config.yaml

# Step 2: Training (can be parallel)
mpirun -np 56 python step_2_train_rom.py --config my_config.yaml \
    --run-dir /path/to/run_directory

# Step 3: Evaluation (always serial)
python step_3_evaluate.py --config my_config.yaml \
    --run-dir /path/to/run_directory
```

## Configuration

The pipeline is controlled by a YAML configuration file. Key sections:

### Paths
```yaml
paths:
  output_base: "/path/to/output/"  # Base directory for runs
  data_dir: "/path/to/data/"        # Raw simulation data
  training_files:
    - "file1.h5"
    - "file2.h5"
  test_files:
    - "test_file.h5"
```

### POD Parameters
```yaml
pod:
  r: 100                # Number of POD modes
  svd_save: 100         # Number of singular values to save
```

### Regularization Sweep
```yaml
regularization:
  state_lin:
    min: 1.0e-2
    max: 1.0e3
    num: 12
    scale: "linear"     # or "log"
  state_quad:
    min: 1.0e7
    max: 1.0e12
    num: 12
  output_lin:
    min: 1.0e-8
    max: 1.0e-2
    num: 12
  output_quad:
    min: 1.0e-10
    max: 1.0e-2
    num: 12
```

### Model Selection
```yaml
model_selection:
  method: "threshold"    # or "top_k"
  num_top_models: 20     # for top_k method
  threshold_mean: 0.05   # for threshold method
  threshold_std: 0.30
```

## Output Structure

Each run creates a directory with the following structure:

```
20241217_143022_hw_rom_experiment/
├── config.yaml              # Copy of configuration used
├── pipeline_status.yaml     # Step completion status
├── step_1.log              # Step 1 log
├── step_2.log              # Step 2 log
├── step_3.log              # Step 3 log
├── POD.npz                 # POD basis (U, S)
├── X_hat_train.npy         # Projected training data
├── X_hat_test.npy          # Projected test data
├── data_boundaries.npz     # Trajectory boundaries
├── initial_conditions.npz  # ICs in full and reduced space
├── learning_matrices.npz   # Matrices for ROM training
├── gamma_reference.npz     # Reference Gamma values
├── ensemble_models.npz     # Trained ROM ensemble
├── sweep_results.npz       # Sweep summary statistics
├── ensemble_predictions.npz # Predictions on all trajectories
├── evaluation_metrics.yaml # Evaluation results
└── figures/                # Diagnostic plots
    ├── train/
    └── test/
```

## HPC Job Script Example

For TACC Frontera:

```bash
#!/bin/bash
#SBATCH -J opinf_pipeline
#SBATCH -o opinf_%j.out
#SBATCH -e opinf_%j.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 56
#SBATCH -t 04:00:00

module load python3
source ~/venv/bin/activate

cd /path/to/pipeline

ibrun python run_pipeline.py --config config.yaml
```

## Dependencies

- numpy
- xarray
- h5py / h5netcdf
- pyyaml
- matplotlib (optional, for plots)
- mpi4py (optional, for parallel execution)

## Algorithm Overview

### Step 1: Data Preprocessing

1. Load raw HDF5 simulation data (density, phi fields)
2. Compute POD basis via truncated SVD
3. Project training and test data onto POD basis
4. Prepare learning matrices for ridge regression

### Step 2: ROM Training

For each combination of regularization parameters:

1. Solve regularized least-squares for state operators (A, F)
2. Integrate ROM forward from training IC
3. Solve regularized least-squares for output operators (C, G, c)
4. Compute error metrics vs reference Gamma values

Select models by:
- **Top-K**: Keep k models with lowest total error
- **Threshold**: Keep all models meeting error criteria

### Step 3: Evaluation

1. Load ensemble of trained models
2. For each trajectory (train and test):
   - Run all models from trajectory IC
   - Compute ensemble mean and spread
3. Compare to reference data
4. Generate diagnostic plots

## References

Peherstorfer, B., & Willcox, K. (2016). Data-driven operator inference
for nonintrusive projection-based model reduction. *Computer Methods in
Applied Mechanics and Engineering*, 306, 196-215.
