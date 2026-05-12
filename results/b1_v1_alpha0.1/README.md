# B1 v1 — Classical OpInf baseline, alpha = 0.1, r = 50

Locked B1 v1 recipe (textbook OpInf, no closure, no stability projection)
applied to the alpha = 0.1 trajectory. Same recipe definition as
`b1_v1_alpha1/`; differs only in the data window.

## Run identity

| Field | Value |
|---|---|
| Method | B1 v1 — textbook OpInf (linear + quadratic), mean-centered, Tikhonov |
| Closure | none |
| Stability projection | none |
| alpha | 0.1 |
| Reduced dimension | r = 50 |
| Train trajectory length | 4000 rows (post burn-in) |
| Operator columns | p(r=50) = 1 + r + r(r+1)/2 = 1326 |
| Rows / cols | 4000 / 1326 ≈ **3.02× over-determined** |
| DNS source | `data/IEEE/hw2d/alpha0.1_n512/`, T=3000, 10700 frames, burn-in 5350 |
| Frontera run dir | `/scratch2/10407/anthony50102/IEEE/output/20260512_111152_b1_alpha0.1_r50/` |
| Job ID | 7711030 |

## Headline test metrics (single test trajectory, 1350 steps)

### Gamma_n (particle flux)

| Quantity | ref | pred |
|---|---|---|
| mean | 1.5110 | 1.4218 |
| std  | 0.2178 | 0.0222 |

- `err_mean_Gamma_n = 0.0591` (**5.9 %** error in mean)
- `err_std_Gamma_n  = 0.8982` (**89.8 %** error in std)

### Gamma_c (conductive flux)

| Quantity | ref | pred |
|---|---|---|
| mean | 1.1703 | 1.0959 |
| std  | 0.1230 | 0.0149 |

- `err_mean_Gamma_c = 0.0636` (**6.4 %** error in mean)
- `err_std_Gamma_c  = 0.8786` (**87.9 %** error in std)

### Interpretation

Strongest-transport regime in the parameter sweep (alpha = 0.1, weak
adiabatic coupling → most fully-developed turbulence). The mean particle
flux is recovered to 5.9 %, but the fluctuation envelope collapses by
~10× — the worst sigma collapse across the three locked alphas.
Consistent with the v1 finding that vanilla unstructured OpInf saturates
the mean transport but cannot represent the chaotic envelope.

## Artifacts

- `config_step_{1,2,3}.yaml` — three-step pipeline configs
- `evaluation_metrics.yaml` — full metrics
- `pipeline_status.yaml` — wallclock + step status
- `pod_energy.png` — POD energy spectrum
