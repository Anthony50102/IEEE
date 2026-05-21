"""Long-horizon autoregressive rollout for trained DISCO models.

Author: Anthony Poole

This is the *field-saving* sibling of ``disco/eval_g1.py``. Where
``eval_g1.py`` writes only per-step NRMSE / VPT metrics, this module
writes the **full predicted and reference fields** to HDF5 so that
animation + QoI plots can be regenerated locally without re-running
inference.

For each requested α we:

  1. Open the HW2D trajectory (G1 tail split) and grab one
     deterministic-seed starting window from the post-burn-in region.
  2. Autoregressively roll out ``rollout_steps`` 1-step predictions.
  3. Pull the matching GT frames + ``gamma_n`` / ``gamma_c`` 1D arrays
     directly from the underlying HDF5.
  4. Compute predicted Γₙ(t) and Γc(t) from the predicted fields via
     :mod:`shared.physics`.
  5. Write ``<out>/rollout_a{α}.h5`` containing:

         pred_n, pred_phi, ref_n, ref_phi          (T, H, W) float32
         pred_gamma_n, pred_gamma_c                (T,)     float64
         ref_gamma_n,  ref_gamma_c                 (T,)     float64

     plus attrs ``alpha``, ``dt``, ``dx``, ``n_past``, ``start_frame``,
     ``checkpoint_path``, ``checkpoint_epoch``.

CLI::

    python -m disco.rollout \\
        --config configs/disco/hw2d_multi.yaml \\
        --checkpoint $SCRATCH/IEEE/output/<run>/checkpoints/best.pt \\
        --data-root /work2/.../IEEE/hw2d \\
        --alphas 0.1 1.0 5.0 \\
        --rollout-steps 2000 \\
        --out $SCRATCH/IEEE/output/rollout/<run>
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from disco.config_hw2d import TrainConfig, load_config
from disco.dataset_specs import HW2D_ALPHA_META
from disco.hw2d_dataset import HW2DDataset
from disco import train_hw2d as th  # _build_model, _alpha_key


# ---------------------------------------------------------------------------
# QoI computations
# ---------------------------------------------------------------------------
# Inlined from shared.physics to avoid pulling shared/__init__'s heavier
# deps (xarray) into the DISCO inference venv. Mirrors the canonical
# definitions exactly.

def compute_gamma_n(n: np.ndarray, phi: np.ndarray, dx: float) -> float:
    """Particle flux Γₙ = -<n · ∂φ/∂y> with periodic central differences."""
    dphi_dy = (np.roll(phi, -1, axis=-2) - np.roll(phi, 1, axis=-2)) / (2.0 * dx)
    return float(-np.mean(n * dphi_dy))


def compute_gamma_c(n: np.ndarray, phi: np.ndarray, c1: float = 1.0) -> float:
    """Conductive flux Γc = c1 · <(n − φ)²>."""
    return float(c1 * np.mean((n - phi) ** 2))


# ---------------------------------------------------------------------------
# checkpoint loading (mirrors eval_g1._load_model_from_checkpoint)
# ---------------------------------------------------------------------------

def _load_model_from_checkpoint(
    cfg: TrainConfig, checkpoint_path: Path, device: torch.device,
    data_root: str,
):
    # Mirror train_hw2d.main(): register both the legacy per-α specs
    # AND the unified ``cfg.dataset_name`` spec into upstream's
    # DATASET_SPECS *before* building the model.
    import disco as _disco  # noqa: PLC0415
    _disco.register_hw2d_specs()
    specs = th._build_specs(cfg, data_root)
    from src.utils.data_utils import DATASET_SPECS  # noqa: PLC0415
    DATASET_SPECS.update(specs)

    model = th._build_model(cfg, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[rollout] state_dict mismatch: missing={len(missing)} "
              f"unexpected={len(unexpected)}")
        if missing:
            print(f"  e.g. missing[:3] = {missing[:3]}")
        if unexpected:
            print(f"  e.g. unexpected[:3] = {unexpected[:3]}")
    model.eval()
    return model, int(ckpt.get("epoch", -1))


# ---------------------------------------------------------------------------
# rollout core
# ---------------------------------------------------------------------------

@torch.no_grad()
def _autoregressive_rollout(
    model,
    initial_window: torch.Tensor,
    n_steps: int,
    state_labels: torch.Tensor,
    dset_name: str,
) -> torch.Tensor:
    """Roll out ``n_steps`` 1-step predictions.

    Parameters
    ----------
    initial_window : (1, n_past, C, H, W) tensor on the model's device.

    Returns
    -------
    (n_steps, C, H, W) float32 CPU tensor of predicted frames.
    """
    window = initial_window.clone()
    out = torch.empty(
        (n_steps, window.shape[2], window.shape[3], window.shape[4]),
        dtype=torch.float32,
    )
    for t in range(n_steps):
        y_pred, _meta = model(
            window, predict_normed=False,
            state_labels=state_labels, dset_name=dset_name,
        )
        next_frame = y_pred[:, 0:1]  # (1, 1, C, H, W)
        out[t] = next_frame[0, 0].detach().to("cpu", dtype=torch.float32)
        window = torch.cat([window[:, 1:], next_frame], dim=1)
    return out


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _read_h5_meta(h5_path: str) -> Tuple[float, float, int]:
    """Return ``(dt, dx, n_x)`` from a trajectory HDF5.

    ``dx`` is derived from ``2π/k0 / n_x`` (HW2D unit-box convention)
    if not stored as an attribute.
    """
    import h5py  # noqa: PLC0415

    with h5py.File(h5_path, "r") as f:
        dt = float(f.attrs.get("dt", float("nan")))
        n_t, n_y, n_x = f["density"].shape
        if "dx" in f.attrs:
            dx = float(f.attrs["dx"])
        else:
            k0 = float(f.attrs.get("k0", 0.15))
            L = 2.0 * np.pi / k0
            dx = L / n_x
    return dt, dx, n_x


def _read_gt_frames(
    h5_path: str,
    start_abs: int,
    n_frames: int,
    stride_y: int,
    stride_x: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read GT density, phi, gamma_n, gamma_c for an absolute frame range."""
    import h5py  # noqa: PLC0415

    with h5py.File(h5_path, "r") as f:
        n_t = f["density"].shape[0]
        stop = min(start_abs + n_frames, n_t)
        ref_n = np.asarray(
            f["density"][start_abs:stop, ::stride_y, ::stride_x],
            dtype=np.float32,
        )
        ref_phi = np.asarray(
            f["phi"][start_abs:stop, ::stride_y, ::stride_x],
            dtype=np.float32,
        )
        gn = (
            np.asarray(f["gamma_n"][start_abs:stop], dtype=np.float64)
            if "gamma_n" in f else np.full(stop - start_abs, np.nan)
        )
        gc = (
            np.asarray(f["gamma_c"][start_abs:stop], dtype=np.float64)
            if "gamma_c" in f else np.full(stop - start_abs, np.nan)
        )
    return ref_n, ref_phi, gn, gc


# ---------------------------------------------------------------------------
# per-α rollout
# ---------------------------------------------------------------------------

def rollout_alpha(
    model,
    cfg: TrainConfig,
    data_root: str,
    alpha: float,
    rollout_steps: int,
    state_labels: torch.Tensor,
    device: torch.device,
    out_dir: Path,
    checkpoint_path: Path,
    checkpoint_epoch: int,
    seed: int = 0,
) -> dict:
    """Single (α, deterministic seed) rollout. Writes HDF5; returns summary."""
    if alpha not in HW2D_ALPHA_META:
        raise ValueError(f"alpha={alpha} not in HW2D_ALPHA_META")

    dset = HW2DDataset(
        main_path=f"{data_root}/{HW2D_ALPHA_META[alpha]['dirname']}",
        resolution=tuple(cfg.resolution),
        n_past=cfg.n_past,
        n_future=cfg.n_future,
        burn_in_frac=cfg.burn_in_frac,
        split="g1",
        g1_tail_frac=cfg.g1_tail_frac,
        name=cfg.dataset_name,
        alpha=alpha,
        file_index=0,
    )
    n_avail = len(dset)
    if n_avail < rollout_steps + 1:
        raise RuntimeError(
            f"α={alpha}: G1 tail too short ({n_avail} samples) for "
            f"{rollout_steps}-step rollout"
        )

    rng = np.random.default_rng(seed)
    last_valid = max(1, n_avail - rollout_steps - 1)
    start = int(rng.integers(0, last_valid))
    print(f"[rollout] α={alpha}: n_avail={n_avail} start={start} "
          f"rollout_steps={rollout_steps}")

    seed_sample = dset[start]
    x0 = seed_sample["input_fields"].unsqueeze(0).to(device)  # (1, P, C, H, W)

    t0 = time.time()
    pred = _autoregressive_rollout(
        model, x0, rollout_steps, state_labels, cfg.dataset_name
    )  # (T, C, H, W) cpu float32
    elapsed = time.time() - t0
    print(f"[rollout] α={alpha}: {rollout_steps} steps in {elapsed:.1f}s "
          f"({rollout_steps/max(elapsed,1e-9):.2f} step/s)")

    pred_np = pred.numpy()
    pred_n = pred_np[:, 0]
    pred_phi = pred_np[:, 1]

    h5_path = dset._file_path
    sy, sx = dset._stride
    # First predicted frame absolute index in HDF5 = t_burn + g1.t_lo + start + n_past
    g1 = dset._g1_split
    start_frame_abs = dset._t_burn + g1.t_lo + start + cfg.n_past

    ref_n, ref_phi, ref_gn_full, ref_gc_full = _read_gt_frames(
        h5_path, start_frame_abs, rollout_steps, sy, sx,
    )
    T_ref = ref_n.shape[0]
    if T_ref < rollout_steps:
        print(f"[rollout] WARNING: only {T_ref}/{rollout_steps} GT frames "
              f"available (trajectory ran out); truncating arrays.")
        pred_n = pred_n[:T_ref]
        pred_phi = pred_phi[:T_ref]

    dt, dx, _ = _read_h5_meta(h5_path)
    # dx is over the *native* grid; correct for spatial stride.
    dx_eff = dx * sx

    pred_gamma_n = np.empty(pred_n.shape[0], dtype=np.float64)
    pred_gamma_c = np.empty(pred_n.shape[0], dtype=np.float64)
    for t in range(pred_n.shape[0]):
        pred_gamma_n[t] = float(compute_gamma_n(pred_n[t], pred_phi[t], dx_eff))
        pred_gamma_c[t] = float(compute_gamma_c(pred_n[t], pred_phi[t], c1=alpha))

    ref_gamma_n = np.empty(ref_n.shape[0], dtype=np.float64)
    ref_gamma_c = np.empty(ref_n.shape[0], dtype=np.float64)
    if np.isnan(ref_gn_full).all():
        for t in range(ref_n.shape[0]):
            ref_gamma_n[t] = float(compute_gamma_n(ref_n[t], ref_phi[t], dx_eff))
            ref_gamma_c[t] = float(compute_gamma_c(ref_n[t], ref_phi[t], c1=alpha))
    else:
        ref_gamma_n[:] = ref_gn_full[:ref_n.shape[0]]
        ref_gamma_c[:] = ref_gc_full[:ref_n.shape[0]]

    # per-step NRMSE (field-averaged)
    diff = pred_np[:ref_n.shape[0]] - np.stack([ref_n, ref_phi], axis=1)
    rmse = np.sqrt(np.mean(diff ** 2, axis=(1, 2, 3)))
    ref_std = np.std(
        np.stack([ref_n, ref_phi], axis=1), axis=(1, 2, 3), ddof=1
    )
    nrmse_per_step = rmse / np.where(ref_std > 1e-15, ref_std, np.nan)

    import h5py  # noqa: PLC0415
    out_h5 = out_dir / f"rollout_a{th._alpha_key(alpha)}.h5"
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("pred_n", data=pred_n, compression="gzip", compression_opts=4)
        f.create_dataset("pred_phi", data=pred_phi, compression="gzip", compression_opts=4)
        f.create_dataset("ref_n", data=ref_n, compression="gzip", compression_opts=4)
        f.create_dataset("ref_phi", data=ref_phi, compression="gzip", compression_opts=4)
        f.create_dataset("pred_gamma_n", data=pred_gamma_n)
        f.create_dataset("pred_gamma_c", data=pred_gamma_c)
        f.create_dataset("ref_gamma_n", data=ref_gamma_n)
        f.create_dataset("ref_gamma_c", data=ref_gamma_c)
        f.create_dataset("nrmse_per_step", data=nrmse_per_step)
        f.attrs["alpha"] = float(alpha)
        f.attrs["dt"] = float(dt)
        f.attrs["dx"] = float(dx_eff)
        f.attrs["n_past"] = int(cfg.n_past)
        f.attrs["rollout_steps"] = int(rollout_steps)
        f.attrs["start_frame"] = int(start_frame_abs)
        f.attrs["start_idx"] = int(start)
        f.attrs["seed"] = int(seed)
        f.attrs["checkpoint_path"] = str(checkpoint_path)
        f.attrs["checkpoint_epoch"] = int(checkpoint_epoch)
        f.attrs["dataset_name"] = cfg.dataset_name
        f.attrs["source_h5"] = h5_path

    summary = {
        "alpha": float(alpha),
        "rollout_steps": int(pred_n.shape[0]),
        "start_frame_abs": int(start_frame_abs),
        "nrmse_per_step_mean": float(np.nanmean(nrmse_per_step)),
        "nrmse_per_step_final": float(nrmse_per_step[-1]),
        "pred_gamma_n_mean": float(np.mean(pred_gamma_n)),
        "ref_gamma_n_mean": float(np.mean(ref_gamma_n)),
        "pred_gamma_c_mean": float(np.mean(pred_gamma_c)),
        "ref_gamma_c_mean": float(np.mean(ref_gamma_c)),
        "wall_s": float(elapsed),
        "out_h5": str(out_h5),
    }
    print(f"  → wrote {out_h5}")
    print(f"  → NRMSE mean={summary['nrmse_per_step_mean']:.4f} "
          f"final={summary['nrmse_per_step_final']:.4f}")
    print(f"  → Γₙ pred/ref mean = {summary['pred_gamma_n_mean']:.4f} / "
          f"{summary['ref_gamma_n_mean']:.4f}")
    print(f"  → Γc pred/ref mean = {summary['pred_gamma_c_mean']:.4f} / "
          f"{summary['ref_gamma_c_mean']:.4f}")
    return summary


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="DISCO long-horizon rollout")
    ap.add_argument("--config", required=True, type=Path,
                    help="Training YAML used to fit the checkpoint")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=str)
    ap.add_argument("--alphas", nargs="+", type=float, required=True)
    ap.add_argument("--rollout-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[rollout] loading {args.checkpoint}")
    model, ckpt_epoch = _load_model_from_checkpoint(
        cfg, args.checkpoint, device, args.data_root,
    )
    print(f"[rollout] model built; checkpoint epoch={ckpt_epoch}")

    state_labels = torch.tensor([0, 1], dtype=torch.long, device=device)

    import yaml  # noqa: PLC0415
    summaries = []
    for alpha in args.alphas:
        try:
            summ = rollout_alpha(
                model, cfg, args.data_root, alpha, args.rollout_steps,
                state_labels, device, args.out, args.checkpoint, ckpt_epoch,
                seed=args.seed,
            )
            summaries.append(summ)
        except Exception as e:  # noqa: BLE001
            print(f"[rollout] α={alpha}: FAILED with {type(e).__name__}: {e}")
            summaries.append({"alpha": float(alpha), "error": str(e)})

    with (args.out / "summary.yaml").open("w") as fh:
        yaml.safe_dump({
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(ckpt_epoch),
            "config": str(args.config),
            "rollout_steps": args.rollout_steps,
            "seed": args.seed,
            "alphas": list(args.alphas),
            "per_alpha": summaries,
        }, fh, sort_keys=False)
    print(f"[rollout] wrote summary to {args.out}/summary.yaml")


if __name__ == "__main__":
    main()
