# DMD / BOPDMD Method Instructions

## Overview

The `dmd/` directory implements **Bagging Optimized Dynamic Mode Decomposition (BOPDMD)** for reduced-order modeling of HW2D plasma turbulence. It fits a continuous-time linear system in POD-reduced space:

$$\frac{d\hat{x}}{dt} = A\hat{x}$$

and forecasts via:

$$\hat{x}(t) = \sum_j b_j \phi_j e^{\alpha_j t}$$

where $\alpha_j$ are continuous-time eigenvalues, $\phi_j$ are modes, and $b_j$ are amplitudes.

**Key library:** `pydmd` (BOPDMD algorithm).

---

## File Layout

| File | Purpose |
|------|---------|
| `step_1_preprocess.py` | Load HDF5 data, compute POD basis (method of snapshots), project training/test data, save ICs |
| `step_2_train.py` | Fit BOPDMD in reduced space; optionally train a quadratic learned output model for Gamma |
| `step_3_evaluate.py` | Forecast, reconstruct full state, compute Gamma, metrics, plots |
| `data.py` | Trajectory loading, POD computation (Gram eigendecomposition), data projection, save/load helpers |
| `utils.py` | `DMDConfig` dataclass, forecasting functions, output model operators, Gamma from stacked states |
| `save_pod_basis.py` | Standalone script to export POD basis for analysis |
| `config/` | YAML configurations (example, temporal split variants, various rank settings) |
| `run_local.sh` | Local pipeline runner |

---

## Pipeline Details

### Step 1 — Preprocessing

1. Loads HDF5 via `shared/data_io.py` or `dmd/data.py` helpers
2. Centers data (training mean subtracted from both train and test)
3. Computes POD via the **method of snapshots** (Gram matrix eigendecomposition — cheaper when n_time << n_spatial)
4. Projects data: `Xhat = U_r^T @ Q_centered`
5. Saves: POD basis (`U_r`, `S`), projected data (`Xhat_train`, `Xhat_test`), ICs, boundaries, preprocessing info

### Step 2 — DMD Fitting

1. Loads projected data from Step 1
2. Computes initial eigenvalue guess via standard DMD: `init_alpha = log(eigs) / dt`
3. Fits BOPDMD (from `pydmd.BOPDMD`) with the initial guess
4. Optionally trains a **quadratic output model** `Y = CX + GX^(2) + c` mapping reduced states to Gamma
5. Saves: eigenvalues, modes, amplitudes, output model operators

**Memory note:** For large ranks (r > 500), the code uses `float32` throughout Step 2 (`DTYPE = np.float32`) and upcasts to `float64` only for the pydmd fitting call. This is critical — do not change without understanding the memory implications. The `use_proj=False` setting avoids creating a dense `r×r` identity matrix in float64.

### Step 3 — Evaluation

1. For each trajectory: compute amplitudes from IC via least squares (`np.linalg.lstsq(modes, x0)`)
2. Forecast reduced state via exponential formula
3. Either:
   - **Learned output:** Apply `C, G, c` operators to get Gamma directly from reduced state
   - **Physics-based (default):** Reconstruct full state `Q = U_r @ X_hat + mean`, then compute Gamma via `compute_gamma_from_state()`
4. Compute metrics (relative errors in mean and std of Gamma_n, Gamma_c)
5. Generate plots via `shared/plotting.py`

---

## Config Structure (`DMDConfig`)

`DMDConfig` extends `OpInfConfig` (inherits all base fields). Additional DMD-specific fields:

```yaml
dmd:
  rank: null            # DMD rank (null = use POD rank r)
  num_trials: 0         # BOPDMD bagging trials (0 = no bagging)
  use_proj: true        # Whether to use projection basis
  eig_sort: "real"      # Eigenvalue sorting criterion
  k0: 0.15              # HW2D wavenumber parameter
  c1: 1.0               # HW2D adiabaticity parameter
  use_learned_output: false  # If true, train C, G, c output operators
  output_alpha_lin: 1.0e-4   # Tikhonov reg for linear output term
  output_alpha_quad: 1.0e-6  # Tikhonov reg for quadratic output term
```

The config is loaded via `load_dmd_config()`, which first parses the base OpInf fields then overlays the `dmd:` section.

---

## Key Functions in `utils.py`

- `dmd_forecast_reduced(eigs, modes, amplitudes, t)` — core exponential forecast, returns `(r, n_time)` complex array
- `dmd_forecast(eigs, modes_full, amplitudes, t)` — full-space version (modes already lifted to full space)
- `fit_output_operators(X_train, Y_Gamma, alpha_lin, alpha_quad)` — fits `C, G, c` via Tikhonov; uses `scipy.sparse.linalg.lsqr` for large `r` to avoid forming full quadratic feature matrix
- `get_quadratic_terms(x)` — computes upper-triangular products for quadratic feature vector
- `predict_gamma_learned(X_hat, output_model)` — applies learned operators to predict Gamma
- `compute_gamma_from_state(Q, n_fields, n_y, n_x, k0, c1)` — physics-based Gamma from stacked state vector
- `reconstruct_full_state(X_hat, U_r, mean)` — lifts from reduced space: `Q = U_r @ X_hat + mean`
- `periodic_gradient(field, dx, axis)` — central differences with periodic boundaries

---

## Gamma Computation

DMD has **two paths** for computing Gamma:

1. **Physics-based** (default): reconstruct `Q_full`, split into density/phi, call `periodic_gradient` + spatial averaging. This duplicates logic from `shared/physics.py` but operates on stacked state vectors.
2. **Learned output model**: train `C, G, c` operators so `Gamma ≈ C·x_hat + G·x_hat^(2) + c`. Useful when full-state reconstruction is too expensive.

Both must agree with the canonical `shared/physics.py` implementation. If you modify the physics formulas, update **both** `shared/physics.py` and `dmd/utils.py`.

---

## Common Issues

- **NaN/Inf in forecasts**: unstable eigenvalues cause exponential blowup. The code checks for this and marks trajectories as invalid. Consider enabling `eig_constraints={"stable"}` in BOPDMD if this is frequent.
- **Memory at large rank**: POD basis `U_r` is `(2·ny·nx, r)` in float64. At 512×512 grid with r=1000, that's ~4GB. Step 2 uses `float32` to halve this.
- **pydmd import**: Always import `pydmd` inside functions, not at module level, so the rest of the code works without it installed.
- **Multiple training trajectories**: BOPDMD only fits on the first trajectory (warns if multiple are provided). The code takes `Xhat_train[:train_boundaries[1], :]`.
