# B1 v1 — Classical OpInf baseline, alpha = 1.0, r = 50

Locked baseline for the parametric Hasegawa–Wakatani study. This is the
"fair classical" reference: textbook operator inference (OpInf) with **no
closure term** and **no stability projection**, fit on the alpha = 1.0
trajectory at r = 50.

## Run identity

| Field | Value |
|---|---|
| Method | B1 v1 — textbook OpInf (linear + Poisson-bracket quadratic) |
| Closure | none |
| Stability projection | none |
| alpha | 1.0 |
| Reduced dimension | r = 50 |
| Train trajectory length | 2000 rows (post burn-in) |
| Operator columns | p(r=50) = 1 + r + r(r+1)/2 = 1325 |
| Rows / cols | 2000 / 1325 ≈ **1.51× over-determined** |
| Frontera run dir | `/scratch2/10407/anthony50102/IEEE/output/20260511_124229_b1_alpha1_r50/` |

## Headline test metrics (single test trajectory, 1001 steps)

Extracted from `evaluation_metrics.yaml` (`test.trajectories[0]`):

### Gamma_n (particle flux)

| Quantity | ref | pred |
|---|---|---|
| mean | 0.6139 | 0.5731 |
| std  | 0.0423 | 0.0106 |

- `err_mean_Gamma_n = 0.0664` (**6.64 %** error in mean)
- `err_std_Gamma_n  = 0.7486` (**74.86 %** error in std)

### Gamma_c (conductive flux)

| Quantity | ref | pred |
|---|---|---|
| mean | 0.6094 | 0.5829 |
| std  | 0.0355 | 0.0086 |

- `err_mean_Gamma_c = 0.0435` (**4.35 %** error in mean)
- `err_std_Gamma_c  = 0.7581` (**75.81 %** error in std)

### Interpretation

Mean fluxes match the DNS reference to within ~5–7 %. Fluctuation
amplitudes (std) are systematically under-predicted by ~75 % — the model
captures the saturated transport level but not the turbulent envelope.
This is the expected qualitative behaviour for textbook OpInf without
closure on a chaotic regime, and is exactly the gap the framework's
context-conditioned closure / structured operator is designed to close.

## Artifacts

- `config_step_{1,2,3}.yaml` — the three-step OpInf pipeline configs
- `evaluation_metrics.yaml` — full metrics (train + test, per-trajectory + ensemble)
- `pipeline_status.yaml` — wallclock + step status
- `pod_energy.png` — POD energy spectrum (justifies r = 50)

Multi-GB artifacts (`POD_basis_Ur.npy`, `X_hat_{train,test}.npy`,
`ensemble_models.npz`, `sweep_results.npz`) intentionally not copied; see
the Frontera run dir.
