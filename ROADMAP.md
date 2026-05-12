# Project Roadmap

**Paper:** *Context-Conditioned Structured Reduced Operators: A Framework for
Parametric Surrogates of Chaotic PDEs* (IEEE/CiSE special issue).

**This file is the single human-readable source of truth for what we're
building, what's done, and what's next.** It is loaded automatically by
GitHub Copilot via `.github/copilot-instructions.md`. Edit freely; agents
will re-read it on each session.

For the **why** (scope decisions, design rationale, gates) see the longer
plan in the session workspace (`~/.copilot/session-state/.../plan.md`).
This file is for **what** and **status**.

---

## North-star claim

> A **structured**, PDE-class-aware reduced operator, conditioned on the
> same context as an unstructured baseline, generalizes better across the
> parameter regime (G3) at a **substantially smaller parameter count**.

The control comparison is **structured (B4 / framework) vs unstructured
(B3 / DISCO-lite)**, both trained inside *our* framework on identical
data, encoders, integrators, and compute budgets.

---

## Methods we evaluate

| ID | Name | What it is | Role |
|----|------|------------|------|
| B1 | **Per-α OpInf** | Classical OpInf fit independently at each α | Sanity baseline; per-trajectory upper bound on what unstructured-linear-quadratic can do |
| B2 | **Affine-µ pOpInf** | Parametric OpInf with affine-µ operator stacking across {0.1, 1, 5} | Classical parametric baseline; expected to fail at the transition (α=1.5 G3) — that's a paper plot |
| B3 | **DISCO-lite** | Unstructured U-Net operator, same encoder / head / integrator / data as B4 | Apples-to-apples control: isolates the *structured vs unstructured operator* claim |
| B4 | **Framework** | Our structured operator + context encoder + head + diff. integrator | The headline method |

Generalization protocols:

- **G1**: same α, unseen IC.
- **G2**: same α, unseen IC, longer rollout.
- **G3**: held-out α=1.5 (between {1.0, 5.0}, near transition).

---

## Phase ladder (hard gates)

### Phase 0 — Scaffolding ✅ done
- [x] Archive cut methods (`dmd/`, `fno/`, KS/NS)
- [x] Create `hw/`, `rom/`, `disco_lite/`, `eval/` skeletons (NotImplementedError stubs)
- [x] Rewrite `.github/copilot-instructions.md` for the new layout
- [x] One-config-tree under `configs/` (`data/`, `opinf/`, `rom/`, `disco_lite/`)

### Phase 1 — HW DNS data ✅ done (T=3000 for {0.1, 1, 5}; α=1.5 held-out at T=500)
- [x] 4-α DNS at 256² (jobs 7683431–7683434, +salvage 7692505)
- [x] `hw.validate` Γₙ vs hw2d reference: all 4 α pass (z ≤ 1.22)
- [x] `hw.dataset` loader + snippet sampler, smoke verified
- [x] Extended α∈{0.1, 1.0} to T=1500 (jobs 7698655, 7698656)
- [x] Extended α∈{0.1, 1.0, 5.0} to T=3000 (jobs 7706867/8/9). α=0.1: 10700 frames. α=1: 12001 frames. α=5: 12001 frames. Gives ≥3000 post-burn-in rows per α → r=75 at ≥1.7× over-det. α=1.5 intentionally held-out.

### Phase 2 — Classical OpInf baselines (B1, B2) 🟡 in progress (B1 done; B2 next)
- [x] B1 α=1.0 smoke at r=20, T=500: ran end-to-end (job 7698116). σ(Γₙ) collapse 13× — diagnosed as rank-limit (r=20 captures only 66% POD energy).
- [x] Port reference-Γ loaders off `xarray` → `h5py` (Frontera-portable)
- [x] **B1 v1 at r=50 locked** as fair classical baseline (`results/b1_v1_alpha1/`, commit `d425804`). Test ⟨Γₙ⟩ err 6.6%, σ(Γₙ) err 75% (collapsed 4×), 1.51× over-det.
- [x] Reg-grid corner-pinning audit at r=50: interior on all 4 axes; `state_quad`/`out_quad` lean right (~40% at max) — widened upper bounds for r=75 sweep.
- [x] **B1 v1 at α=0.1, r=50** (`results/b1_v1_alpha0.1/`, job 7711030). Test ⟨Γₙ⟩ err 5.9%, σ collapse ~10× (worst — strongest-transport regime). 3.0× over-det.
- [x] **B1 v1 at α=5.0, r=50** (`results/b1_v1_alpha5.0/`, job 7711047). Test ⟨Γₙ⟩ err 12.7%, σ collapse ~3× (mildest — strongly damped). 3.8× over-det. Train→test gap suggests longer trajectory needed but window length already capped by walltime.
- [x] **B1 v2 at α=1, r=75** (`results/b1_v2_alpha1/`, job 7711029). Test ⟨Γₙ⟩ err **3.7%** (down from 6.6%), σ collapse **87%** (up from 75%). Doubling modes resolved the mean better but made σ collapse *worse* — confirms the σ-collapse is structural to vanilla unstructured OpInf, not a resolution issue. **This is the paper's headline B1 finding.**
- [x] r=75 reg-grid audit: `state_quad` widening succeeded (top-5 winners at interior 2.2e11). `out_quad` still pinned at new max 1e4; further widening abandoned (would only damp σ further, worsening the headline weakness).
- [ ] **B2 affine-µ pOpInf on G1+G3** — train across {0.1, 1, 5}; eval at trained αs and at held-out α=1.5. (`p2-b2-g1g3`)
  - [x] Math layer (`opinf/parametric_data.py`, `opinf/parametric_solve.py`) verified to machine precision on synthetic planted data (`opinf/test_parametric_smoke.py`, passes on Frontera too).
  - [x] Pipeline scaffolded: `step_1_preprocess_parametric.py` (multi-α load, per-α centering, pooled POD, α-blocked D/Y), `step_2_train_parametric.py` (MPI 2D Tikhonov sweep with sentinel disqualification), `step_3_evaluate_parametric.py` (G1+G3 eval). `parametric_config.py` is a dedicated multi-α YAML loader. `configs/opinf/b2_alpha_p015.yaml` + `_smoke.yaml` + `run_opinf_parametric.slurm` ready. Pushed to `refactor` and pulled on Frontera.
  - [x] Trajectory inventory verified on Frontera:
        α=0.1 has only 10700 frames (burn-in 5350 → post-burn 5350);
        α=1.0 and α=5.0 have 12001 (burn-in 6000 → post-burn 6001);
        α=1.5 sentinel has only 2001 (burn-in 1000 → post-burn 1001).
        B2 uses a uniform 4000-frame training window per α anchored at
        each α's own burn-in (so no α imbalance in D); at r=75 K_total=12000
        vs 5852 cols = 2.05× over-det.
  - [ ] Frontera dry-run (r=20 smoke; then r=75 production).
- [ ] **Phase 2A cleanup** — delete `step_*_serial.py` (~3500 LOC), fan out `opinf/utils.py`, consolidate `opinf/config/` into `configs/opinf/`. (`p2-cleanup`)

### Phase 3 — Framework B4, smoke at α=1
- [ ] `rom.basis.fourier` + projection round-trip < 1e-10 (`p3-basis`)
- [ ] `rom.operator.hw_quadratic` — triadic Poisson-bracket sparse quadratic; matches hw2d RHS to numerical precision (`p3-op`)
- [ ] `rom.integrator.etdrk4` (or RK4) — differentiable; matches hw2d short-horizon rollout from fixed IC at α=1 (`p3-int`)
- [ ] End-to-end solver with **hand-set coefficients** — sanity gate. If this fails we stop. (`p3-handset`)
- [ ] `rom.encoder.mlp` + `rom.head.symmetric` — start with small MLP, transformer later (`p3-encoder`)
- [ ] **B4 G1 train+eval at α=1**, gate within 2× B1 nrmse (`p3-b4-g1`)

### Phase 4 — Headline experiments (B4 vs DISCO-lite, all G's)
- [ ] **Generate 5–10 ICs per training α at 256²** (`p4-ic-gen`)
- [ ] **B4 G2** (unseen IC, trained α) (`p4-b4-g2`)
- [ ] **B4 G3** (held-out α=1.5) — *the headline result* (`p4-b4-g3`)
- [ ] **DISCO-lite** on G1/G2/G3, matched compute budget (`p4-disco-lite`)
- [ ] **Parameter-count vs accuracy figure** — rhetorical centerpiece (`p4-paramcount`)

### Phase 5 — Polish & paper
- [ ] 512² spot-check at α=1 (resolution scaling) (`p5-512`)
- [ ] Ablations: r / k_max, snippet length, sparsity prior, rollout horizon (`p5-ablate`)
- [ ] Fill `needsresult` placeholders in `main.tex` (`p5-fill-tex`)

### Infrastructure (cross-cutting)
- [x] MPI 2 GB pickle pitfall: fix `chunked_gather` gate, document in `shared/mpi_utils.py`
- [ ] Audit remaining pickle-mode MPI callsites for >2 GB risk (`infra-mpi-audit`)
- [ ] Unify MPI wrappers (`opinf/utils.py` → `shared/mpi_utils.py`) (`infra-mpi-consolidate`)
- [ ] Round-trip tests for `chunked_gather` / `chunked_bcast` (`infra-mpi-tests`)
- [ ] Pin Vista venv constraints (`p0-pyenv`)

---

## Locked results (paper-quotable)

| Run | What | Test ⟨Γₙ⟩ err | Test σ(Γₙ) err | Location |
|-----|------|--------------:|---------------:|----------|
| B1 v1 α=0.1 r=50 | Vanilla OpInf, T=3000, 3.0× over-det | 5.9% | 90% | `results/b1_v1_alpha0.1/` |
| B1 v1 α=1   r=50 | Vanilla OpInf, T=1500, 1.5× over-det (initial lock) | 6.6% | 75% | `results/b1_v1_alpha1/` |
| B1 v1 α=5   r=50 | Vanilla OpInf, T=3000, 3.8× over-det | 12.7% | 66% | `results/b1_v1_alpha5.0/` |
| **B1 v2 α=1 r=75** | Vanilla OpInf, T=3000, 1.7× over-det, widened reg | **3.7%** | 87% | `results/b1_v2_alpha1/` |

Headline B1 finding: more modes (r=50 → r=75) cut the mean error nearly
in half but *worsened* the σ collapse. The collapse is structural to
unstructured OpInf on a chaotic flux state — exactly the gap our
context-conditioned structured operator is designed to close.

---

## Working agreements

- **Asks before doing**: code commits, pushes, Frontera pulls, job submissions.
- **No AI co-author trailers** in commits. Repo policy beats runtime default.
- **Outputs go outside the repo** (`$SCRATCH/IEEE/output/<ts>/` on HPC, `local_output/` locally). Only *curated* results land in `results/`.
- **One file per concept**; no `utils.py` in new code (existing `opinf/utils.py` is grandfathered, to be fanned out).
- **MPI is opt-in**: lazy import inside functions; single-rank must work without mpi4py.
- **OpInf formulation is locked**: linear A + quadratic H, mean-centered, Tikhonov. No closure, no stability projection — both invite reviewer arguments.

---

## Known open questions

- ~~B1 σ(Γₙ) collapse at α=1: known rank-limit~~ **Resolved**: r=75 retry
  shows the collapse is structural to unstructured OpInf, not a rank
  issue. Reported as the B1 weakness that motivates the framework.
- **DISCO-lite architectural choices** (channels, depth, conditioning mechanism) deferred until B4's encoder/head are settled, so both methods can share design language.
- **B2 transition behavior**: we *expect* B2 to fail at α=1.5. That failure mode is a paper plot, not a bug to fix.
