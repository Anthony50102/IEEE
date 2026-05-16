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

## Methods we evaluate (2-week sprint scope)

| ID | Name | What it is | Role |
|----|------|------------|------|
| B1 | **Per-α OpInf** | Classical OpInf fit independently at each α | Classical baseline ✅ locked |
| DISCO | **DISCO (Morel et al. 2025)** | Hypernetwork → small U-Net operator → time integration, trained on multi-α context snippets | Headline method for this paper |

*Cut from this paper:* B2 (affine-µ pOpInf, out of budget), B3 ablation
naming (we just call the method **DISCO**), B4 structured operator
(future / journal version).

Generalization protocols:

- **G1**: trained α, future time (short-horizon NRMSE).
- **G3**: held-out α=1.5 (statistical observables + rollout boundedness).
- **G2**: optional stretch goal (trained α, fresh IC).

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
  - [x] Frontera dry-run (r=20 smoke, pipeline validation only — `20260512_132420_b2_alpha_p015_r20_smoke`):
        step 1 → POD orthog. 2.7e-14, D=(4497, 462), over-det 9.73×.
        step 2 → 6/16 candidates admitted; best λ₁²=1e6, λ₂²=1e14.
        step 3 → G1 α=0.1 MSE=3.5e4 (bounded), α=1.0 MSE=8.1e3 (bounded),
        **α=5.0 NaN** (diverges); G3 α=1.5 MSE=6.9e3 (bounded).
        Finding: sentinel-only disqualification is insufficient — best
        candidate is unstable at α=5.0 because sweep never checked it.
        Also: snapshot-norm rescaling not yet implemented → projected
        states have norm ~10²-10³, forcing reg grid up to λ₂²~1e14.
  - [ ] r=75 production run (after: add trained-α stability check + snapshot rescaling).
- [ ] **Phase 2A cleanup** — delete `step_*_serial.py` (~3500 LOC), fan out `opinf/utils.py`, consolidate `opinf/config/` into `configs/opinf/`. (`p2-cleanup`)

### Phase 2 — Classical OpInf baselines (B1) ✅ locked

**B2 cut** (2026-05-16): with 2 weeks of Frontera compute remaining,
B2 affine-µ pOpInf is out of scope. B1 stands as the sole classical
baseline. The B2 pipeline code (`opinf/parametric_*`, `step_*_parametric.py`,
`configs/opinf/b2_*`) is retained on `refactor` branch but not pursued.

- [x] B1 v1 at α∈{0.1, 1, 5}, r=50 — locked (see Locked results below)
- [x] B1 v2 at α=1, r=75 — headline B1 finding (3.7% mean / 87% σ collapse)
- [x] B1 reg-grid audits at r=50 and r=75
- ~~[ ] B2 affine-µ pOpInf~~ **cut** (out of compute budget)

### Phase 6 — DISCO sprint (2-week, hard deadline) 🟢 active

Cut from the original plan: **B2** (above), the **structured-operator B4**
contribution, and the **B3-vs-B4 head-to-head**. The single paper claim
becomes:

> *DISCO (Morel, Han, Oyallon 2025) — an architecture for
> multi-physics-agnostic prediction — also generalizes across parameter
> regimes within a single chaotic PDE (Hasegawa–Wakatani), without
> retraining at the unseen parameter.*

Training data: 3-α snippet pairs from existing HW2D DNS at α∈{0.1, 1, 5},
held-out α=1.5. **Each α is registered as a separate "dataset"** in
DISCO's framework (matching how PDEBench treats different physics) so
the hypernetwork infers α from the context snippet.

Compute: HW2D OpInf/B1 work used **Frontera CPU**; DISCO training and
evaluation moves to **Vista Grace Hopper (GH200)** GPU nodes.

Generalization protocols (kept):
- **G1**: trained α, future time (short-horizon MSE — meaningful inside τ_Lyap).
- **G3**: held-out α=1.5 (statistical observables: ⟨Γₙ⟩, σ(Γₙ), energy time-average; rollout boundedness).
- **G2**: only if Phase 6 finishes early.

Workflow:

- [x] **`p6-clone`** — Upstream `RudyMorel/DISCO` @ `ddd18f17` vendored to `IEEE/disco/upstream/` (models, attention, torchdiffeq, yparams, train_reference.py, config_reference/). License + SOURCE.md preserved. Old `IEEE/disco_lite/` B3 stub removed.
- [x] **`p6-data`** — Adapter from `hw.dataset` → DISCO context-snippet format `(T, C=2, H, W)`. `IEEE/disco/dataset_specs.py` registers the four αs as `hw2d_a01/a10/a50/a15` (a15=G3 held-out). `IEEE/disco/hw2d_dataset.py` provides `HW2DDataset` (one alpha, with `train`/`g1` tail-holdout split) and `HW2DMixedDataset` (multi-α concat with `field_labels` + `file_index`). Per-sample dict matches upstream's `MixedDataset` contract. CPU-smoke verified on synthetic HDF5.
- [x] **`p6-smoke`** — Single-α α=1.0 CPU smoke (`disco/smoke_train.py`) at 32², hidden_dim=96, 15.8M params, batch=2, 8 SGD steps: loss 0.31→0.088 (monotone). Validated `src.*` import shim, HW2D `DATASET_SPECS` injection, full forward+backward through hypernet→param-generator→ODE-integrated operator. `hidden_dim` must be ÷12 (upstream `RMSGroupNorm` hardcode) and ÷`num_heads`.
- [ ] **`p6-train`** — Full multi-α training on 3-α, 256², ~50 epochs on 1–2 Frontera GPU nodes (rtx queue). Target NRMSE comparable to DISCO's PDEBench numbers.
- [ ] **`p6-sweep`** — Light HP sweep: LR, snippet length T, hypernet capacity (~3–5 configs total).
- [ ] **`p6-eval-g1`** — G1 short-horizon rollout at each trained α; report NRMSE vs B1.
- [ ] **`p6-eval-g3`** — G3 rollout at α=1.5; report stability + ⟨Γₙ⟩, σ(Γₙ), energy time-average compared to ground-truth DNS.
- [ ] **`p6-figures`** — Rollout snapshots, error tables, possibly a parameter-space UMAP (analog of paper Fig. 4) over the 3 αs.
- [ ] **`p6-paper-update`** — Strip B4/structured language from `IEEE-CiSE-Special-Issue/`; refocus claim on parameter-regime generalization.

### Phases retired (~~~~ kept for record ~~~~)

- ~~Phase 3 — Framework B4, smoke at α=1~~ deferred (post-deadline / journal version)
- ~~Phase 4 — Headline B4 vs DISCO-lite~~ deferred
- ~~Phase 5 — 512² scaling + ablations~~ deferred; only the most critical ablations make it in

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

- ~~B1 σ(Γₙ) collapse at α=1~~ **Resolved**: structural to unstructured OpInf, reported as the B1 weakness motivating the framework.
- ~~B2 transition behavior~~ **Cut** (out of compute budget).
- **DISCO HW2D adaptation**: the original paper uses CNN encoder + attention processor; for HW2D's periodic BCs we need to verify the encoder respects translation equivariance. Resolve during `p6-clone`.
- **Multi-α conditioning** (resolved `p6-data`): we follow upstream — no α label fed to model. Each α registered as a separate dataset in `HW2D_DATASET_SPECS`; the hypernet must infer regime from the context snippet.
- **Statistical metric for G3**: ⟨Γₙ⟩ and σ(Γₙ) are obvious; energy time-average is cheap. Whether to also report kₓ–k_y spectra is a Phase 6 stretch.
