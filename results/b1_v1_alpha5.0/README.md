# B1 v1 — Classical OpInf baseline, alpha = 5.0, r = 50

Locked B1 v1 recipe (textbook OpInf, no closure, no stability projection)
applied to the alpha = 5.0 trajectory. Same recipe definition as
`b1_v1_alpha1/`; differs only in the data window.

## Run identity

| Field | Value |
|---|---|
| Method | B1 v1 — textbook OpInf (linear + quadratic), mean-centered, Tikhonov |
| Closure | none |
| Stability projection | none |
| alpha | 5.0 |
| Reduced dimension | r = 50 |
| Train trajectory length | 5000 rows (post burn-in) |
| Operator columns | p(r=50) = 1 + r + r(r+1)/2 = 1326 |
| Rows / cols | 5000 / 1326 ≈ **3.77× over-determined** |
| DNS source | `data/IEEE/hw2d/alpha5.0_n512/`, T=3000, 12001 frames, burn-in 6000 |
| Frontera run dir | `/scratch2/10407/anthony50102/IEEE/output/20260512_112146_b1_alpha5.0_r50/` |
| Job ID | 7711047 |

## Headline test metrics (single test trajectory, 1001 steps)

### Gamma_n (particle flux)

| Quantity | ref | pred |
|---|---|---|
| mean | 0.1338 | 0.1168 |
| std  | 0.0134 | 0.0045 |

- `err_mean_Gamma_n = 0.1270` (**12.7 %** error in mean)
- `err_std_Gamma_n  = 0.6610` (**66.1 %** error in std)

### Gamma_c (conductive flux)

| Quantity | ref | pred |
|---|---|---|
| mean | 0.1341 | 0.1187 |
| std  | 0.0173 | 0.0062 |

- `err_mean_Gamma_c = 0.1148` (**11.5 %** error in mean)
- `err_std_Gamma_c  = 0.6432` (**64.3 %** error in std)

### Interpretation

Strongly-damped regime (alpha = 5.0, near adiabatic limit). The
fluctuations are an order of magnitude smaller than at alpha = 0.1, the
turbulence is weak and slow, and correspondingly the sigma collapse is
the mildest of the three locked alphas (~3× rather than ~10×). However
the mean error is the largest (12.7 %), with a ~4× train→test gap (3.2 %
→ 12.7 %) suggesting the slow dynamics make the train and test windows
sample different statistical realizations even after a 6000-frame
burn-in. A longer trajectory would tighten this; flagged for B2.

## Artifacts

- `config_step_{1,2,3}.yaml` — three-step pipeline configs
- `evaluation_metrics.yaml` — full metrics
- `pipeline_status.yaml` — wallclock + step status
- `pod_energy.png` — POD energy spectrum
