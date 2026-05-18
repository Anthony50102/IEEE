"""Typed config for ``disco.train_hw2d``.

Author: Anthony Poole

A frozen dataclass that the multi-α DISCO trainer reads. YAML files in
``configs/disco/`` deserialize into this. Keep the schema flat — one
field per knob — so the YAML reads like a parameter table.

The dataclass split into nested sub-configs (e.g. ``ModelCfg``,
``DataCfg``) is intentionally avoided: the trainer is ~300 lines and
flattening the config keeps the call sites short (``cfg.batch_size``
instead of ``cfg.train.batch_size``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class TrainConfig:
    # --- run identity ------------------------------------------------------
    run_name: str = "disco_hw2d"
    seed: int = 0

    # --- data --------------------------------------------------------------
    data_root: str = "/work2/10407/anthony50102/frontera/data/IEEE/hw2d"
    # Unified-dataset mode (2026-05-17): DISCO sees ONE dataset name. α is
    # discriminated purely by snippet content, never by label. The per-α
    # train/val αs choose which on-disk files are sampled.
    dataset_name: str = "hw2d"
    train_alphas: List[float] = field(
        default_factory=lambda: [0.1, 1.0, 5.0]
    )
    val_alphas: List[float] = field(
        default_factory=lambda: [0.1, 1.0, 5.0]
    )
    resolution: Tuple[int, int] = (256, 256)
    n_past: int = 16
    n_future: int = 1
    burn_in_frac: float = 0.5
    g1_tail_frac: float = 0.1
    num_workers: int = 4

    # --- model -------------------------------------------------------------
    hidden_dim: int = 384         # must be % 12 == 0 AND % num_heads == 0
    patch_size: int = 16          # must divide each spatial dim
    processor_blocks: int = 12
    num_heads: int = 6
    groups: int = 1
    bias_type: str = "space-time"
    drop_path: float = 0.1
    max_steps: int = 32
    rtol: float = 5e-6
    atol: float = 1e-9
    integration_library: str = "torchdiffeq"
    hpnn_head_hidden_dim: int = 384

    # --- optimization ------------------------------------------------------
    batch_size: int = 2
    accum_grad: int = 4
    max_epochs: int = 200
    epoch_size: int = 2000
    optimizer: str = "adamw"
    learning_rate: float = 5e-5
    weight_decay: float = 1e-3
    scheduler: str = "cosine"     # 'cosine' or 'none'
    warmup_epochs: int = 1
    gnorm: Optional[float] = 1.0
    amp: bool = True              # bf16 when supported

    # --- validation --------------------------------------------------------
    val_batches: int = 16         # batches per α per validation pass
    val_rollout_steps: Tuple[int, ...] = (1,)

    # --- checkpointing / logging ------------------------------------------
    checkpoint_save_interval: int = 10  # snapshot every N epochs (besides latest+best)
    log_interval: int = 50              # print every N iters
    wandb_enabled: bool = False
    wandb_project: str = "ieee-disco"
    wandb_entity: str = ""

    # ----------------------------------------------------------------------
    def validate(self) -> None:
        from disco.dataset_specs import HW2D_ALPHA_META  # noqa: PLC0415

        h, w = self.resolution
        assert self.hidden_dim % 12 == 0, (
            f"hidden_dim={self.hidden_dim} must be divisible by 12 "
            "(upstream RMSGroupNorm hardcode)"
        )
        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim={self.hidden_dim} must be divisible by num_heads={self.num_heads}"
        )
        assert h % self.patch_size == 0 and w % self.patch_size == 0, (
            f"patch_size={self.patch_size} must divide resolution={self.resolution}"
        )
        assert self.n_past > 0 and self.n_future > 0
        assert self.batch_size > 0 and self.accum_grad > 0
        assert self.max_epochs > 0 and self.epoch_size > 0
        assert 0.0 <= self.burn_in_frac < 1.0
        assert 0.0 < self.g1_tail_frac < 1.0
        assert self.optimizer in ("adam", "adamw"), self.optimizer
        assert self.scheduler in ("cosine", "none"), self.scheduler
        assert self.dataset_name and isinstance(self.dataset_name, str)
        assert len(self.train_alphas) > 0, "train_alphas must be non-empty"
        assert len(self.val_alphas) > 0, "val_alphas must be non-empty"
        for a in list(self.train_alphas) + list(self.val_alphas):
            assert a in HW2D_ALPHA_META, (
                f"alpha={a} not in HW2D_ALPHA_META; known αs: {sorted(HW2D_ALPHA_META)}"
            )


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> TrainConfig:
    """Load a YAML file into a ``TrainConfig``.

    Unknown keys raise. Missing keys keep dataclass defaults. ``overrides``
    is applied on top of the YAML (used by the CLI for ``--max-epochs``
    style flags).
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})

    allowed = {f.name for f in fields(TrainConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")

    if "resolution" in raw and not isinstance(raw["resolution"], tuple):
        raw["resolution"] = tuple(raw["resolution"])
    if "val_rollout_steps" in raw and not isinstance(raw["val_rollout_steps"], tuple):
        raw["val_rollout_steps"] = tuple(raw["val_rollout_steps"])
    # Coerce alpha lists to floats (YAML int 1 → float 1.0 for HW2D_ALPHA_META lookup).
    for key in ("train_alphas", "val_alphas"):
        if key in raw:
            raw[key] = [float(a) for a in raw[key]]

    cfg = TrainConfig(**raw)
    cfg.validate()
    return cfg


def dump_config(cfg: TrainConfig) -> Dict[str, Any]:
    """JSON-friendly dict of the config (tuples → lists)."""
    d = asdict(cfg)
    d["resolution"] = list(d["resolution"])
    d["val_rollout_steps"] = list(d["val_rollout_steps"])
    return d


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        cfg = TrainConfig()
        cfg.validate()
        print("default config:")
    else:
        cfg = load_config(sys.argv[1])
        print(f"loaded {sys.argv[1]}:")
    for k, v in dump_config(cfg).items():
        print(f"  {k:28s} {v!r}")
