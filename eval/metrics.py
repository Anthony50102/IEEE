"""Evaluation metrics.

Author: Anthony Poole

Four families:
  - trajectory:   relative L2 over time, valid-prediction-horizon
  - spectral:     time-averaged kinetic / enstrophy spectra; spectral slope
  - qoi:          time-averaged Gamma_n, Gamma_c
  - cost:         operator parameter count, train wall, inference latency

Implementations are being ported from `shared/metrics.py` as the refactor
progresses.
"""

raise NotImplementedError("eval.metrics: port from shared/metrics.py during refactor")
