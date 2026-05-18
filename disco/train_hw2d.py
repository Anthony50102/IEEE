"""Multi-α DISCO trainer on HW2D (production entry point).

Author: Anthony Poole

Single-GPU (or CPU smoke), no DDP. Reads a typed YAML config
(``configs/disco/hw2d_multi.yaml``), trains across α ∈ {0.1, 1, 5} on
the HW2D snippets, validates per-α on the held-out G1 tail of each
trajectory, and writes checkpoints + JSONL metrics under ``--out``.

See ``disco/smoke_train.py`` for a minimal CPU-only version of the
forward+backward loop; this module reuses the same model + dataset
plumbing and adds:

  - epoch + iteration scheduling (``epoch_size`` batches per epoch)
  - gradient accumulation
  - AMP (bf16 on H200; off on CPU and pre-Ampere GPUs)
  - cosine LR with linear warmup
  - per-α G1 validation each epoch
  - rotating + best + latest checkpoints with full opt/sched state
  - JSONL metric log and a final ``run_summary.yaml``
  - optional wandb (off by default; see ``--wandb``)

CLI::

    python -m disco.train_hw2d \\
        --config configs/disco/hw2d_multi.yaml \\
        --out $SCRATCH/IEEE/output/${ts}_disco_train

    # smoke (CPU-runnable against the synthetic HDF5 from p6-data)
    python -m disco.train_hw2d \\
        --config configs/disco/hw2d_multi_smoke.yaml \\
        --data-root /tmp/hw2d_synth \\
        --out ./local_output/train_smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

import disco  # registers `src` alias and dataset-specs shim
from disco.config_hw2d import TrainConfig, dump_config, load_config
from disco.dataset_specs import (
    HW2D_ALPHA_META,
    alpha_paths as _alpha_paths_for,
    make_unified_spec,
)
from disco.hw2d_dataset import HW2DDataset, HW2DMixedDataset


# ============================================================
# Utilities
# ============================================================

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _amp_dtype(device: torch.device, cfg_amp: bool) -> Optional[torch.dtype]:
    """Returns the AMP autocast dtype, or None if AMP is disabled."""
    if not cfg_amp:
        return None
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _alpha_key(alpha: float) -> str:
    """Stable string label for an α value (used as the val-loader dict key)."""
    return f"a{alpha:g}"


def _build_specs(cfg: TrainConfig, data_root: str) -> Dict[str, Dict]:
    """Build the single-entry unified spec table for the union of train+val αs.

    Registered into upstream's ``DATASET_SPECS`` so DISCO's introspection
    succeeds for ``cfg.dataset_name``.
    """
    alphas = sorted(set(cfg.train_alphas) | set(cfg.val_alphas))
    return make_unified_spec(alphas, root=data_root, dataset_name=cfg.dataset_name)


def _build_train_loader(cfg: TrainConfig, data_root: str) -> DataLoader:
    dset = HW2DMixedDataset(
        alpha_paths=_alpha_paths_for(cfg.train_alphas, root=data_root),
        n_past=cfg.n_past,
        n_future=cfg.n_future,
        resolution=tuple(cfg.resolution),
        split="train",
        burn_in_frac=cfg.burn_in_frac,
        g1_tail_frac=cfg.g1_tail_frac,
        name=cfg.dataset_name,
    )
    return DataLoader(
        dset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )


def _build_val_loaders(
    cfg: TrainConfig, data_root: str
) -> Dict[str, DataLoader]:
    """One loader per validation α so per-α NRMSE is straightforward.

    Loader keys are ``f"a{alpha:g}"`` strings; the per-sample
    ``name`` is always ``cfg.dataset_name`` (no α leakage to the model).
    """
    out: Dict[str, DataLoader] = {}
    for i, alpha in enumerate(cfg.val_alphas):
        sub = HW2DDataset(
            main_path=f"{data_root}/{HW2D_ALPHA_META[alpha]['dirname']}",
            resolution=tuple(cfg.resolution),
            n_past=cfg.n_past,
            n_future=cfg.n_future,
            burn_in_frac=cfg.burn_in_frac,
            split="g1",
            g1_tail_frac=cfg.g1_tail_frac,
            name=cfg.dataset_name,
            alpha=alpha,
            file_index=i,
        )
        out[_alpha_key(alpha)] = DataLoader(
            sub,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
    return out


def _build_model(cfg: TrainConfig, device: torch.device):
    disco.register_hw2d_specs()
    from src.models.disco import DISCO  # noqa: PLC0415

    model = DISCO(
        n_states=2,
        hidden_dim=cfg.hidden_dim,
        patch_size=cfg.patch_size,
        ndims=[2],
        groups=cfg.groups,
        processor_blocks=cfg.processor_blocks,
        drop_path=cfg.drop_path,
        num_heads=cfg.num_heads,
        bias_type=cfg.bias_type,
        hpnn_head_hidden_dim=cfg.hpnn_head_hidden_dim,
        dataset_names=[cfg.dataset_name],
        max_steps=cfg.max_steps,
        atol=cfg.atol,
        rtol=cfg.rtol,
        integration_library=cfg.integration_library,
    )
    return model.to(device)


def _param_groups_with_no_decay(model: nn.Module, weight_decay: float):
    """1D params (biases, norm scales) get weight_decay=0; mirrors upstream."""
    decay, no_decay = [], []
    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if len(p.squeeze().shape) <= 1:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def _build_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    groups = _param_groups_with_no_decay(model, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.learning_rate)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(groups, lr=cfg.learning_rate)
    raise ValueError(cfg.optimizer)


def _build_scheduler(
    cfg: TrainConfig, optimizer: torch.optim.Optimizer, last_step: int
):
    if cfg.scheduler == "none":
        return None
    total_steps = max(1, (cfg.max_epochs * cfg.epoch_size) // cfg.accum_grad)
    warmup_steps = max(0, (cfg.warmup_epochs * cfg.epoch_size) // cfg.accum_grad)
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        decay = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=cfg.learning_rate / 100.0,
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, decay], [warmup_steps]
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=cfg.learning_rate / 100.0
        )
    for _ in range(max(0, last_step)):
        sched.step()
    return sched


# ============================================================
# Loss / metrics
# ============================================================

def _gaussian_nll_normalized(y_ref: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match upstream's ``Trainer.mse_loss`` exactly.

    ``y, y_ref`` shape: ``(B, T, C, H, W)``. ``y_ref`` is already in the
    normalized space (target normalized with the model's returned
    ``mean``/``std``). The loss is gaussian-NLL with the per-channel
    spatial variance as the predicted variance; the raw NRMSE is
    computed only for logging.
    """
    spatial_dims = tuple(range(3, y.ndim))
    var = 1e-7 + y_ref.var(spatial_dims, keepdim=True)
    loss = F.gaussian_nll_loss(
        y, y_ref, torch.ones_like(y) * var, eps=1e-8, reduction="mean"
    )
    with torch.no_grad():
        residual = y - y_ref
        norm_ref = 1e-7 + y_ref.pow(2.0).mean(spatial_dims, keepdim=True)
        raw_nrmse2 = residual.pow(2.0).mean(spatial_dims, keepdim=True) / norm_ref
    return loss, raw_nrmse2


def _nrmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Per-(batch,time,channel) NRMSE; shape ``(B, T, C, 1, 1)``."""
    spatial_dims = tuple(range(3, y_true.ndim))
    residual = y_true - y_pred
    mse = residual.pow(2.0).mean(spatial_dims, keepdim=True)
    norm = 1e-7 + y_true.pow(2.0).mean(spatial_dims, keepdim=True)
    return (mse / norm).sqrt()


# ============================================================
# Train / validate
# ============================================================

def _train_one_epoch(
    cfg: TrainConfig,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    state_labels: torch.Tensor,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    epoch_size = cfg.epoch_size
    accum = cfg.accum_grad
    fp16 = amp_dtype == torch.float16

    logs = {"loss": 0.0, "nrmse": 0.0, "n": 0}
    # Per-α breakdown comes from the diagnostic ``alpha`` field on the sample
    # dict (the model never sees this). Loader is mixed-α; we bucket each
    # batch by its first sample's α.
    per_alpha: Dict[str, Dict[str, float]] = {
        _alpha_key(a): {"loss": 0.0, "nrmse": 0.0, "n": 0} for a in cfg.train_alphas
    }
    optimizer.zero_grad(set_to_none=True)

    t0 = time.time()
    it = iter(loader)
    for batch_idx in range(epoch_size):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        x = batch["input_fields"].to(device, non_blocking=True)
        y_true = batch["output_fields"].to(device, non_blocking=True)
        alpha = float(batch["alpha"][0])
        akey = _alpha_key(alpha)

        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=amp_dtype)
            if amp_dtype is not None
            else torch.amp.autocast("cuda", enabled=False)
        )
        with autocast_ctx:
            y_pred, meta = model(
                x, predict_normed=False,
                state_labels=state_labels, dset_name=cfg.dataset_name,
            )
            y_true_norm = (y_true - meta["mean"]) / meta["std"]
            loss, raw_nrmse2 = _gaussian_nll_normalized(y_true_norm, y_pred)
            loss = loss / accum

        if fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        with torch.no_grad():
            nrmse_v = raw_nrmse2.sqrt().mean().item()
            loss_v = (loss.item() * accum)
            logs["loss"] += loss_v
            logs["nrmse"] += nrmse_v
            logs["n"] += 1
            if akey in per_alpha:
                pa = per_alpha[akey]
                pa["loss"] += loss_v
                pa["nrmse"] += nrmse_v
                pa["n"] += 1

        is_step = ((batch_idx + 1) % accum == 0) or (batch_idx + 1 == epoch_size)
        if is_step:
            if fp16:
                scaler.unscale_(optimizer)
            if cfg.gnorm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.gnorm)
            if fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        if (batch_idx % cfg.log_interval == 0):
            print(
                f"  epoch {epoch:3d} batch {batch_idx:5d}/{epoch_size} "
                f"loss={loss_v:.4e} nrmse={nrmse_v:.4e} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} ({akey})",
                flush=True,
            )

    dt = time.time() - t0
    out = {
        "train_loss": logs["loss"] / max(1, logs["n"]),
        "train_nrmse": logs["nrmse"] / max(1, logs["n"]),
        "train_time_s": dt,
        "train_steps_per_s": logs["n"] / max(1e-9, dt),
        "lr": optimizer.param_groups[0]["lr"],
    }
    for akey, d in per_alpha.items():
        if d["n"] > 0:
            out[f"train_nrmse/{akey}"] = d["nrmse"] / d["n"]
            out[f"train_loss/{akey}"] = d["loss"] / d["n"]
    return out


@torch.no_grad()
def _validate(
    cfg: TrainConfig,
    model: nn.Module,
    val_loaders: Dict[str, DataLoader],
    device: torch.device,
    state_labels: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}
    mean_acc, mean_n = 0.0, 0
    for akey, loader in val_loaders.items():
        nrmse_sum = 0.0
        n_batches = 0
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= cfg.val_batches:
                break
            x = batch["input_fields"].to(device, non_blocking=True)
            y_true = batch["output_fields"].to(device, non_blocking=True)
            y_pred, _meta = model(
                x, predict_normed=True,
                state_labels=state_labels, dset_name=cfg.dataset_name,
            )
            nrmse_v = _nrmse(y_true, y_pred).mean().item()
            nrmse_sum += nrmse_v
            n_batches += 1
        v = nrmse_sum / max(1, n_batches)
        out[f"val_nrmse/{akey}"] = v
        mean_acc += v
        mean_n += 1
    out["val_nrmse"] = mean_acc / max(1, mean_n)
    return out


# ============================================================
# Checkpointing
# ============================================================

def _save_ckpt(
    path: Path,
    epoch: int,
    iters: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    best_val: float,
    cfg: TrainConfig,
) -> None:
    payload = {
        "epoch": epoch,
        "iters": iters,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict(),
        "best_val_nrmse": best_val,
        "config": dump_config(cfg),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_ckpt(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    map_location,
) -> Tuple[int, int, float]:
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck["model_state_dict"])
    optimizer.load_state_dict(ck["optimizer_state_dict"])
    if scheduler is not None and ck.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ck["scheduler_state_dict"])
    if "scaler_state_dict" in ck:
        scaler.load_state_dict(ck["scaler_state_dict"])
    return int(ck["epoch"]), int(ck["iters"]), float(ck.get("best_val_nrmse", float("inf")))


# ============================================================
# Entry point
# ============================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HW2D × DISCO multi-α trainer")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--out", required=True, type=str, help="Output directory")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume from")
    p.add_argument("--wandb", action="store_true",
                   help="Enable wandb logging (overrides YAML)")
    p.add_argument("--data-root", type=str, default=None,
                   help="Override data_root from the YAML")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Override max_epochs from the YAML")
    p.add_argument("--device", type=str, default=None,
                   help="'cuda' or 'cpu'; default auto-detect")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    overrides: Dict[str, object] = {}
    if args.data_root is not None:
        overrides["data_root"] = args.data_root
    if args.max_epochs is not None:
        overrides["max_epochs"] = args.max_epochs
    if args.wandb:
        overrides["wandb_enabled"] = True

    cfg = load_config(args.config, overrides=overrides)
    _set_seed(cfg.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    config_path = out_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(dump_config(cfg), f, sort_keys=False)

    device = torch.device(args.device) if args.device else _pick_device()
    amp_dtype = _amp_dtype(device, cfg.amp)
    print(f"[trainer] device={device} amp_dtype={amp_dtype} out={out_dir}", flush=True)

    specs = _build_specs(cfg, cfg.data_root)
    disco.register_hw2d_specs()  # ensure the unified spec is visible to upstream
    from src.utils.data_utils import DATASET_SPECS  # noqa: PLC0415
    DATASET_SPECS.update(specs)

    train_loader = _build_train_loader(cfg, cfg.data_root)
    val_loaders = _build_val_loaders(cfg, cfg.data_root)
    print(f"[trainer] dataset_name={cfg.dataset_name}", flush=True)
    print(f"[trainer] train αs: {cfg.train_alphas}  "
          f"({len(train_loader.dataset)} samples)", flush=True)
    for akey, vl in val_loaders.items():
        print(f"[trainer] val   {akey}: {len(vl.dataset)} samples", flush=True)

    model = _build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[trainer] model params: {n_params:,}", flush=True)

    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer, last_step=0)
    scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))
    state_labels = torch.tensor([0, 1], dtype=torch.long, device=device)

    start_epoch, iters, best_val = 0, 0, float("inf")
    if args.resume is not None:
        start_epoch, iters, best_val = _load_ckpt(
            Path(args.resume), model, optimizer, scheduler, scaler,
            map_location=device,
        )
        print(f"[trainer] resumed from {args.resume} at epoch {start_epoch} "
              f"(best_val={best_val:.4e})", flush=True)

    run = None
    if cfg.wandb_enabled:
        import wandb  # noqa: PLC0415
        run = wandb.init(
            project=cfg.wandb_project, entity=cfg.wandb_entity or None,
            name=cfg.run_name, dir=str(out_dir), config=dump_config(cfg),
            resume="allow",
        )

    t_start = time.time()
    for epoch in range(start_epoch, cfg.max_epochs):
        train_logs = _train_one_epoch(
            cfg, model, train_loader, optimizer, scheduler, scaler,
            device, amp_dtype, state_labels, epoch,
        )
        val_logs = _validate(cfg, model, val_loaders, device, state_labels)

        iters += cfg.epoch_size
        row = {"epoch": epoch, "iters": iters, **train_logs, **val_logs}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if run is not None:
            run.log(row)
        print(
            f"[trainer] epoch {epoch:3d} train_nrmse={train_logs['train_nrmse']:.4e} "
            f"val_nrmse={val_logs['val_nrmse']:.4e} "
            f"({train_logs['train_time_s']:.1f}s)",
            flush=True,
        )

        # latest
        _save_ckpt(ckpt_dir / "latest.pt", epoch + 1, iters,
                   model, optimizer, scheduler, scaler, best_val, cfg)
        # rotating epoch snapshot
        if (epoch + 1) % cfg.checkpoint_save_interval == 0:
            _save_ckpt(ckpt_dir / f"epoch_{epoch + 1:04d}.pt", epoch + 1, iters,
                       model, optimizer, scheduler, scaler, best_val, cfg)
        # best
        if val_logs["val_nrmse"] < best_val:
            best_val = val_logs["val_nrmse"]
            _save_ckpt(ckpt_dir / "best.pt", epoch + 1, iters,
                       model, optimizer, scheduler, scaler, best_val, cfg)

    t_total = time.time() - t_start
    summary = {
        "run_name": cfg.run_name,
        "n_params": n_params,
        "epochs_run": cfg.max_epochs - start_epoch,
        "best_val_nrmse": best_val,
        "wall_clock_s": t_total,
        "device": str(device),
        "amp_dtype": str(amp_dtype),
    }
    with open(out_dir / "run_summary.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"[trainer] done. summary: {summary}", flush=True)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
