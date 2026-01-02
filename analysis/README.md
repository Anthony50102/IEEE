# Analysis Scripts

This folder contains standalone analysis scripts for investigating ROM performance 
and other post-hoc analyses. These scripts are not part of the main training/evaluation 
pipelines but are useful for specific investigations.

## Scripts

### `pod_reconstruction.py`

Analyzes POD reconstruction quality as a function of the number of modes.

**Purpose:**
- Verify that gamma (Γ_n, Γ_c) computations match saved values
- Compute POD basis via Gram matrix eigendecomposition
- Analyze reconstruction error vs number of modes (r)
- Compare gamma estimation accuracy from reconstructed states

**Outputs:**
- `pod_reconstruction_YYYYMMDD_HHMMSS.log`: Detailed log file
- `pod_error_vs_r.png`: Error metrics vs number of modes
- `pod_gamma_timeseries.png`: Time series comparison

**Configuration:**
Edit the script directly to change:
- `DATA_FILE`: Path to HW2D simulation HDF5 file
- `R_VALUES`: List of r values to test
- `NT_MAX`: Number of timesteps to use

**Usage:**
```bash
cd /path/to/IEEE/analysis
python pod_reconstruction.py
```