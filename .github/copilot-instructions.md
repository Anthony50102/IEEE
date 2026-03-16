# Copilot Instructions — IEEE ROM/DL Benchmark

## Project Overview

This repo benchmarks **reduced-order models (ROMs)** and **deep-learning surrogates** for PDE forecasting, targeting an IEEE publication. The primary benchmark PDE is **Hasegawa-Wakatani 2D** (HW2D) plasma turbulence. **Kuramoto-Sivashinsky** (KS) is a planned second benchmark.

Methods currently implemented:

| Method | Directory | Type | Key Library |
|--------|-----------|------|-------------|
| Optimized DMD (BOPDMD) | `dmd/` | ROM — continuous-time linear | `pydmd` |
| Operator Inference (OpInf) | `opinf/` | ROM — discrete-time quadratic | custom (NumPy/SciPy + MPI) |
| Fourier Neural Operator (FNO) | `fno/` | DL surrogate | `neuralop` + PyTorch |

Planned future methods: DeepONet, generative PDE solvers.

The quantity of interest for **all** methods is prediction of transport coefficients **Γ_n** (particle flux) and **Γ_c** (conductive flux). Every method must produce comparable Gamma predictions and use the same plotting utilities for paper-quality figures.

---

## Repository Structure

```
    IEEE-CiSE-Special-Issue/ # Github clone of overleaf project for this paper
../
    IEEE/
    ├── shared/          # Common utilities: data I/O, physics, plotting, MPI helpers
    ├── dmd/             # DMD pipeline (3-step)
    ├── opinf/           # OpInf pipeline (3-step, MPI-parallel)
    ├── fno/             # FNO pipeline (3-step, PyTorch)
    ├── analysis/        # Post-hoc comparison scripts
    ├── data/
    │   ├── hw2d/        # Hasegawa-Wakatani HDF5 simulations + generation notebook
    │   └── ks/          # Kuramoto-Sivashinsky data + generation notebook
    ├── scripts/         # SLURM job scripts (Frontera) + local bash runners
    └── frontera_pip_constraints.txt
```

### The 3-Step Pipeline Pattern

**Every method follows the same 3-step structure.** This is a hard convention:

| Step | Purpose | CLI Pattern |
|------|---------|-------------|
| `step_1_preprocess.py` | Load data, compute basis/embeddings, project | `python step_1_preprocess.py --config config.yaml` |
| `step_2_train.py` | Fit model (operators, networks, etc.) | `python step_2_train.py --config config.yaml --run-dir <dir>` |
| `step_3_evaluate.py` | Forecast, reconstruct, compute Gamma, metrics, plots | `python step_3_evaluate.py --config config.yaml --run-dir <dir>` |

Each step writes a status entry to `pipeline_status.yaml` with timestamps. Steps check that prior steps completed before running (`check_step_completed()`).

---

## Critical Rules

### 1. Physics Consistency

**Γ_n and Γ_c must be computed identically across all methods.** The canonical implementation lives in `shared/physics.py`:

```
Γ_n = -⟨n · ∂φ/∂y⟩     (particle flux)
Γ_c = c₁⟨(n − φ)²⟩     (conductive flux)
```

Where `∂/∂y` is a periodic central-difference gradient. If a method needs its own Gamma computation (e.g., `dmd/utils.py` has one for computing from stacked state vectors), it **must** produce identical numerical results to `shared/physics.py`. When in doubt, delegate to the shared implementation.

Grid parameters come from `get_hw2d_grid_params(k0, nx)` → `{Lx, dx, k0, nx, ny}` with `Lx = 2π/k0`, `dx = Lx/nx`.

### 2. Never Import `mpi_utils` at Module Level

`shared/mpi_utils.py` triggers MPI initialization on import. **Always** use lazy/conditional imports:

```python
# WRONG — breaks serial scripts
from shared.mpi_utils import distribute_indices

# RIGHT
def my_parallel_function():
    from shared.mpi_utils import distribute_indices
    ...
```

The `shared/__init__.py` deliberately does NOT export `mpi_utils`.

### 3. Standardized Plotting for the Paper

All methods **must** use `shared/plotting.py` for figure generation. Key functions:

- `plot_gamma_timeseries()` — the primary comparison plot (handles single + ensemble with ±2σ)
- `plot_pod_energy()` — for ROMs that use POD
- `plot_state_snapshots()` — 6-column grid comparing reference vs predicted fields
- `plot_state_error_timeseries()` — relative L₂ error over time
- `generate_state_diagnostic_plots()` — unified wrapper for trajectory lists
- `setup_publication_style()` — call this to set matplotlib rcParams

Do **not** create one-off matplotlib code in method directories for figures that go in the paper. Extend `shared/plotting.py` instead so all methods benefit.

### 4. Configuration via YAML + Dataclass

- OpInf and DMD use **dataclass** configs (`OpInfConfig`, `DMDConfig` which extends `OpInfConfig`)
- FNO currently uses a plain dict — this should eventually be a dataclass too
- Config files live in `<method>/config/` as YAML
- The regularization grid is defined as `{min, max, num, scale}` dicts and expanded to arrays by `_build_reg_array()`
- Two training modes: `"multi_trajectory"` (train/test on different HDF5 files) and `"temporal_split"` (train on `[start, end)` of one file, test on later portion)

### 5. Linear, Simple, Readable Code

This is a scientific computing paper codebase. Prioritize:

- **Linear flow** — avoid deep abstractions, metaclasses, or framework-heavy patterns
- **Readability** — someone reviewing the paper should be able to trace the math through the code
- **NumPy-style docstrings** with Parameters/Returns sections
- **Sectioned code** using `# ===` banner comments for major blocks
- **Explicit memory management** — `del` large arrays, `gc.collect()` where needed, especially on HPC

### 6. Dual Execution: HPC (SLURM) and Local (bash)

All pipelines must run on both:

- **TACC Frontera** — via SLURM scripts in `scripts/` (Intel Cascade Lake, 56 cores/node, `ibrun` for MPI)
- **Local machine** — via bash scripts in `scripts/` or `<method>/run_local.sh`

The only difference should be the launch mechanism (SLURM vs bash). The Python code itself should not contain HPC-specific logic. MPI code should degrade gracefully when `mpi4py` is not available or running with 1 rank.

### 7. Data Format

- HW2D data is **HDF5** (NetCDF4-compatible), loaded via `xarray` with the `h5netcdf` engine
- Fields: `density` (T, H, W), `phi` (T, H, W), `gamma_n` (T,), `gamma_c` (T,)
- State vector for ROMs: `Q = [density_flat; phi_flat]` shape `(2·ny·nx, n_time)`
- File naming encodes simulation parameters (step size, domain, grid, physics constants)

### 8. Float Precision

- Default to `float64` (double precision)
- Use `float32` for large-rank DMD to manage memory
- FNO uses `float32` (PyTorch default)
- When comparing across precisions, upcast to `float64` first

---

## Dependencies (No pyproject.toml Yet)

There is currently no `pyproject.toml` or `requirements.txt`. This should be created. Current dependencies:

| Category | Packages |
|----------|----------|
| Core scientific | `numpy`, `scipy`, `h5py`, `xarray`, `h5netcdf` |
| MPI (optional) | `mpi4py` |
| DMD | `pydmd` |
| FNO | `torch`, `neuralop` |
| Visualization | `matplotlib` |
| Config | `pyyaml` |

Cross-module imports use `sys.path.insert(0, ...)` rather than package installation. This is intentional — the repo is not an installable package.

---

## Adding a New Method

When adding a new PDE surrogate method (e.g., DeepONet, diffusion model):

1. **Create a new top-level directory** named after the method (e.g., `deeponet/`)
2. **Follow the 3-step pipeline**:
   - `step_1_preprocess.py` — data loading and any method-specific preprocessing
   - `step_2_train.py` — model fitting/training
   - `step_3_evaluate.py` — forecasting, Gamma computation, metrics, plots
3. **Accept `--config` and `--run-dir` CLI arguments** matching existing methods
4. **Create `config/` subdirectory** with at least an `example.yaml`
5. **Create `utils.py`** with a config dataclass (inherit from `OpInfConfig` if fields overlap, or create standalone)
6. **Use `shared/physics.py`** for Gamma computation — do not reimplement
7. **Use `shared/plotting.py`** for all paper figures
8. **Use `shared/data_io.py`** for HDF5 loading — do not reimplement
9. **Add `run_local.sh`** for local execution
10. **Add SLURM script** in `scripts/` for HPC execution
11. **Write `pipeline_status.yaml`** entries with `save_step_status()` at each step
12. **Add a `README.md`** documenting the method, its math, and how to run it
13. **Support both HW2D and KS** datasets (or at least design for it)

### Adding a New Benchmark PDE

When extending to a new PDE (beyond HW2D and KS):

1. Add data generation notebook in `data/<pde_name>/`
2. Ensure data is stored as HDF5 with fields that `shared/data_io.py` can load (or extend it)
3. Define the physics-relevant QoI (analogous to Γ_n, Γ_c) in `shared/physics.py`
4. Add corresponding plotting support in `shared/plotting.py`
5. Update configs for each method to point to the new data

---

## File Conventions

- **Modules**: `snake_case.py`
- **Classes**: `PascalCase` (`DMDConfig`, `BasisData`, `SingleStepDataset`)
- **Functions/variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE` (`DTYPE`, `CDTYPE`, `DEFAULT_CURRICULUM`)
- **Author header**: `Author: Anthony Poole` at top of each module
- **Logging**: always log to both console and file via `setup_logging()`
- **Output directories**: timestamped, created by `create_run_directory()`

---

## Method-Specific Instructions

See the per-method instruction files for detailed guidance:

- [dmd.md](.github/dmd.md) — DMD/BOPDMD specifics
- [opinf.md](.github/opinf.md) — Operator Inference specifics
- [fno.md](.github/fno.md) — FNO specifics
