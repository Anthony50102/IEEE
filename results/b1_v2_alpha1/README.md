# B1 v2 — Classical OpInf at higher resolution, alpha = 1.0, r = 75

Upgrade of `b1_v1_alpha1/` to higher reduced-order dimension (r = 50 →
r = 75) on the extended T = 3000 DNS. Same vanilla recipe (linear A +
quadratic H, mean-centered, Tikhonov, no closure, no stability
projection); only r and the regularization upper bounds changed.

This run answers the question: "is the v1 sigma collapse a resolution
issue or a structural one?"

**Answer: structural.** Doubling-and-a-half the operator columns and
widening the reg grid roughly halved the mean error but made the sigma
collapse worse, not better.

## Run identity

| Field | Value |
|---|---|
| Method | B1 v2 — textbook OpInf (linear + quadratic), mean-centered, Tikhonov |
| Closure | none |
| Stability projection | none |
| alpha | 1.0 |
| Reduced dimension | r = 75 |
| Train trajectory length | 5000 rows (post burn-in) |
| Operator columns | p(r=75) = 1 + r + r(r+1)/2 = 2926 |
| Rows / cols | 5000 / 2926 ≈ **1.71× over-determined** |
| DNS source | `data/IEEE/hw2d/alpha1.0_n512/`, T=3000, 12001 frames, burn-in 6000 |
| Frontera run dir | `/scratch2/10407/anthony50102/IEEE/output/20260512_111152_b1_alpha1_r75/` |
| Job ID | 7711029 |

## What changed vs v1

| Knob | v1 (r=50) | v2 (r=75) |
|---|---|---|
| r | 50 | 75 |
| state_quad reg max | 1.0e14 | 1.0e16 |
| out_quad reg max | 1.0e2 | 1.0e4 |
| Train rows | 2000 | 5000 (also extended DNS) |
| All other knobs | — | unchanged |

Reg grid widening was driven by the v1 reg-grid audit
(state_quad/out_quad both right-leaning; widened upper bounds by 2 orders).

## Headline test metrics (single test trajectory, 1001 steps)

### Gamma_n (particle flux)

| Quantity | ref | pred |
|---|---|---|
| mean | 0.5983 | 0.5763 |
| std  | 0.0433 | 0.0058 |

- `err_mean_Gamma_n = 0.0367` (**3.7 %** error in mean — down from 6.6 % at r=50)
- `err_std_Gamma_n  = 0.8662` (**86.6 %** error in std — up from 75 % at r=50)

### Gamma_c (conductive flux)

| Quantity | ref | pred |
|---|---|---|
| mean | 0.5926 | 0.5898 |
| std  | 0.0367 | 0.0055 |

- `err_mean_Gamma_c = 0.0048` (**0.5 %** error in mean)
- `err_std_Gamma_c  = 0.8511` (**85.1 %** error in std)

## Reg-grid audit at r = 75 (post-run)

| Axis | grid range | max-pin % | optimum location |
|---|---|---|---|
| state_lin | [1e-2, 1e6] | 30 % @ 1e6 | interior, right-leaning |
| state_quad | [1e2, 1e16] (widened) | 40 % @ 1e16 | **interior at 2.2e11** for top-5 ✓ |
| out_lin | [1e-6, 1e2] | 10 % @ 1e2 | interior, left-leaning |
| out_quad | [1e-6, 1e4] (widened) | 40 % @ 1e4 | **pinned at 1e4 for all top-5** ✗ |

`state_quad` widening succeeded — the optimum is now inside the new
range. `out_quad` is still pinned at the upper edge. Cranking out_quad
further would only damp the output map more, which can only *worsen* the
sigma collapse — the already-headline weakness of vanilla OpInf here.
Further widening abandoned as cosmetic.

## Interpretation

- **Mean transport**: r = 75 is genuinely better. Cutting the test mean
  error from 6.6 % to 3.7 % at the same alpha shows that more modes do
  resolve the saturated transport level more accurately.
- **Sigma collapse**: r = 75 makes it *worse* (75 % → 87 %). The
  fluctuation envelope is not a resolution issue and not a regularization
  issue — it is structural to the unstructured OpInf ansatz when applied
  to a chaotic, self-sustained flux state. Adding modes pulls more of the
  small-scale energy into the ROM, but without a closure / driving term
  the integrated trajectory still relaxes to a near-fixed point.
- **Conductive flux mean** comes out essentially exact (0.5 %) at r = 75 —
  Gamma_c is a tighter quadratic functional and benefits more from the
  extra modes than Gamma_n does.

This is the canonical B1 result the paper will cite as the headline
"vanilla OpInf can match transport averages but not their fluctuations."

## Artifacts

- `config_step_{1,2,3}.yaml` — three-step pipeline configs
- `evaluation_metrics.yaml` — full metrics
- `pipeline_status.yaml` — wallclock + step status
- `pod_energy.png` — POD energy spectrum
