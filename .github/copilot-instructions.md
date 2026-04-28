# Copilot Instructions — Context-Conditioned Structured ROM Framework

## What this repo is

Code for the IEEE/CiSE-track methods paper *Context-Conditioned Structured
Reduced Operators: A Framework for Parametric Surrogates of Chaotic PDEs*.
Companion paper repo lives in `../IEEE-CiSE-Special-Issue/`.

The paper proposes a 5-primitive framework for parametric reduced-order
modeling that sits between classical operator inference (OpInf) and
unstructured hypernetwork operator-learning (DISCO). The headline
instantiation is on the parametric Hasegawa-Wakatani plasma turbulence
equations.

This is **scientific computing code**, not production software. There is
**no CI**. The priorities are:

1. **Clean, simple, readable code.** Someone reviewing the paper should be
   able to trace the math through the code in an afternoon.
2. **Linear flow.** Avoid deep abstractions, metaclasses, framework-heavy
   patterns, abstract base classes.
3. **Sanity checks live as `__main__` blocks**, not a `tests/` directory.
   Run `python -m rom.basis` to validate it on a fixture. No pytest, no
   pre-commit, no GitHub Actions.
4. **One file per concept; flat directories.** No `framework/operators/quadratic/triadic/...`
   nesting. If it's a 200-line file, it lives at one level.
5. **No `utils.py`.** Every function lives in a file named after its concept.

---

## Repository Layout

```
IEEE/
├── hw/                  # HW physics, DNS, dataset. PDE-specific.
│   ├── physics.py       # Gamma_n, Gamma_c, Poisson bracket, RHS
│   ├── dataset.py       # HDF5 loader; whole-trajectory + snippet samplers
│   ├── dns.py           # Wrapper around the upstream hw2d solver
│   └── reference_gammas.yaml   # canonical Gamma_n table from hw2d README
│
├── rom/                 # The framework. PDE-class-agnostic in intent.
│   ├── basis.py         # P1: Fourier projection / lift
│   ├── operator.py      # P2: linear + Poisson-bracket-sparse quadratic
│   ├── encoder.py       # P3: MLP / Transformer over snippets
│   ├── head.py          # P4: coefficient head with sign / sparsity priors
│   ├── integrator.py    # P5: differentiable RK4 / ETDRK4
│   ├── model.py         # composed end-to-end module
│   ├── losses.py        # one-step + rollout + QoI losses
│   └── train.py         # generic train loop, YAML-driven
│
├── disco_lite/          # Unstructured-operator ablation. One file differs.
│   └── unet_operator.py # the only file unique to this baseline
│
├── opinf/               # Classical OpInf baselines (B1, B2).
│   └── ...              # Existing 3-step pipeline; works today.
│
├── eval/                # Cross-method evaluation. Stateless.
│   ├── metrics.py       # trajectory / spectral / QoI / cost metrics
│   ├── plot.py          # paper figures from saved artifacts
│   └── compare.py       # B1/B2/B3'/B4 x G1/G2/G3 matrix
│
├── shared/              # Pre-refactor utilities still in use:
│                        #   physics.py, data_io.py, metrics.py,
│                        #   plotting.py, mpi_utils.py, evaluation_io.py
│                        # Being ported into hw/ and eval/ incrementally.
│
├── configs/             # ONE config tree; named by intent.
│   ├── data/            # DNS data generation
│   ├── opinf/           # baselines (b1_per_alpha, b2_affine_mu)
│   ├── rom/             # framework (g1_alpha1, g2_unseen_ic, g3_unseen_alpha)
│   └── disco_lite/      # ablation (matched-budget variants)
│
├── data/                # Symlinks to actual data; small notebooks in repo.
│   └── hw2d/            # HW2D HDF5 files (HPC symlink) -- h5 gitignored
│
├── scripts/             # SLURM templates only. No method-specific scripts.
│   ├── setup_data.sh
│   └── setup_data.slurm
│
├── analysis/            # Surviving HW analysis scripts (non-mani/non-ks).
│
├── notebooks/           # Exploration only; never imported from src.
│
└── archive/             # Preserved for git-log; do not edit.
    ├── dmd/, fno/                   # cut methods
    ├── data_ks/, data_ns/           # cut PDEs
    ├── analysis_old/                # mani/ks/dmd analysis
    ├── scripts_old/                 # KS/NS/DMD/mani launchers
    ├── configs_old/                 # OpInf KS/NS configs and old sweeps
    └── sweep_configs_old/           # old top-level sweep YAMLs
```

---

## The Framework: Five Primitives (`rom/`)

The whole framework is intentionally seven files and ~1k LOC. Read the
file order top to bottom and the math reads top to bottom too.

| Primitive | File | Role |
|-----------|------|------|
| P1 | `basis.py` | reduced representation: Fourier for HW |
| P2 | `operator.py` | structured operator: linear + k-sparse quadratic |
| P3 | `encoder.py` | maps snippet -> context vector |
| P4 | `head.py` | maps context -> structured-operator coefficients |
| P5 | `integrator.py` | differentiable RK4 / ETDRK4 in reduced space |

Composed in `model.py`, trained via `train.py` driven by
`configs/rom/<cell>.yaml`.

DISCO-lite (`disco_lite/`) reuses every primitive of `rom/` except P2,
which is replaced by an unstructured U-Net. This is the controlled
ablation for the paper's central claim.

### Key sanity checks (Phase 3)

Run each module's `__main__` to validate it:
- `python -m rom.basis`     -> projection round-trip < 1e-10
- `python -m rom.operator`  -> RHS matches hw.physics.hw_rhs to numerical precision
- `python -m rom.integrator`-> short-horizon rollout matches hw2d at fixed alpha
- *End-to-end with hand-set theta = a valid spectral solver.* This is the
  most important check in the project. If it fails, nothing else matters.

---

## Critical Conventions

### 1. Physics consistency

Gamma_n / Gamma_c are computed in **one place** (`hw/physics.py`, currently
mirrored from `shared/physics.py` during port). Every method that reports
QoIs routes through it. Discrepancies are bugs.

```
Gamma_n = -<n * dphi/dy>     (particle flux)
Gamma_c = c1 * <(n - phi)^2> (conductive flux)
```

### 2. MPI is opt-in

Do **not** import `shared/mpi_utils.py` at module top level. Use lazy imports
inside functions. The `shared/__init__.py` deliberately does not export it.
Single-rank execution must work without `mpi4py` installed.

### 3. Configs are typed

Configs are dataclasses. YAML deserializes into them. Code reads
`cfg.k_max`, not `cfg["k_max"]`. Naming policy: configs are named by what
they do (`b1_per_alpha.yaml`), not their hyperparameters
(`opinf_ks_r15_tighter.yaml` style is forbidden; lives in `archive/`).

### 4. Outputs go outside the repo

Run output goes to `$SCRATCH/IEEE/output/<timestamp>/` on HPC and to
`$IEEE_OUTPUT/<timestamp>/` (or `./local_output/`) locally. Never commit
checkpoints or per-run logs. Curate a small set of canonical results into
`results/` for the paper; everything else is ephemeral.

### 5. Plotting reads artifacts, never runs models

`eval/plot.py` and `eval/compare.py` consume `run_summary.yaml + npz` and
emit figures. They do not load checkpoints. Re-making a paper figure should
never require re-running training.

### 6. No `tests/` directory

Each module has a `if __name__ == "__main__":` sanity check. No pytest, no
mocking, no fixtures library. The fixtures are short HW snapshots.

---

## Git Policy

### No AI co-author trailers

Many journals prohibit AI co-authorship claims. Commits MUST NOT include
`Co-authored-by: Copilot ...` or similar. Plain commit messages only.

### Ask before git ops

Do not `commit`, `push`, or `pull` without explicit user approval. Stage and
show diffs only.

### Active branches

`main` and `refactor` only. Other branches were intentionally pruned during
the 2026 refactor; their code (where worth keeping) is in `archive/`.

---

## HPC Quick Reference (TACC)

`hw/dns.py` runs on Frontera, `rom/train.py` runs on Vista.

| System | Use |
|--------|-----|
| Frontera | CPU/MPI: OpInf baselines, HW DNS data generation |
| Vista (gh) | GPU: ROM / DISCO-lite training |

**Frontera specifics: see `.github/frontera.md`.** That file documents the
canonical Python module load (`unset PYTHONPATH; module reset; module load
python3/3.9.2 phdf5/1.10.4`), filesystem layout, queue choices, the
`PYTHONPATH` bashrc gotcha, and submit conventions. Read it before any new
Frontera session; the rules are non-obvious.

Vista specifics: TBD (will be a sibling `.github/vista.md`). MPLBACKEND=Agg
is required on both clusters for matplotlib in headless jobs.

---

## Where to look

- Paper draft: `../IEEE-CiSE-Special-Issue/main.tex`
- Paper repo PLAN: `../IEEE-CiSE-Special-Issue/PLAN.md`
- Knowledge notes: `../knowledge/{novelty_check,opinf_and_popinf,disco,hasegawa_wakatani}.md`
- Refactor plan + experimental workflow: session workspace
  (`~/.copilot/session-state/.../plan.md` if a Copilot session is active).

When in doubt about a refactor or experiment design choice, prefer the
simpler, more readable option. Add complexity only when the math demands it.
