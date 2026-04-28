"""Generic ROM training entry point.

Author: Anthony Poole

Usage:
    python -m rom.train --config configs/rom/g1_alpha1.yaml --run-dir <dir>

Reads a YAML config (parsed to a dataclass), instantiates the model, dataset,
optimizer, and loss, and runs training. Writes:
  - <run-dir>/run_summary.yaml  (machine-readable metrics; same schema as
                                 the existing shared/metrics.py output)
  - <run-dir>/checkpoints/      (Torch state dicts)
  - <run-dir>/log.txt           (training log)
"""

raise NotImplementedError("rom.train: implement Phase 3")
