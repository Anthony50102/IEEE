# configs/

One canonical location for all experiment configs. Layout mirrors the
method packages:

```
configs/
├── data/         # DNS data generation configs (one per training-alpha and per held-out alpha)
├── opinf/        # B1 (per-alpha) and B2 (affine-mu) baseline configs
├── rom/          # B4 (our framework) configs, one per protocol cell
└── disco_lite/   # B3' (unstructured-operator ablation) configs
```

## Naming policy

Configs are named by **what they do**, not by their hyperparameters. Old
soup-style names like `opinf_ks_r15_tighter.yaml` are forbidden; they live
in `archive/configs_old/` for reference.

Good names:
  - `b1_per_alpha.yaml`
  - `b2_affine_mu.yaml`
  - `g1_alpha1_single_ic.yaml`
  - `g3_unseen_alpha.yaml`
  - `hw_alpha1.0_n256.yaml`

The hyperparameters (rank `r`, k_max, regularization, learning rate, ...)
live **inside** the YAML. If two configs differ only in one hyperparameter,
make it a sweep over that field, not a second file.

## Schema

Configs are parsed into typed dataclasses. See:
  - `opinf/utils.py`        for OpInf config dataclass
  - `rom/train.py`          for ROM config dataclass
  - `disco_lite/__init__.py` for DISCO-lite config (extends ROM config)
