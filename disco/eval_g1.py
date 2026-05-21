"""G1 short-horizon rollout evaluation for trained DISCO models.

Author: Anthony Poole

G1 = "short-horizon, in-distribution" per the project evaluation taxonomy.
We take a trained checkpoint (1-step ahead, ``n_future=1``), and do
autoregressive rollout for ``rollout_steps`` frames at each requested
α, comparing against the held-out G1 tail of that α's trajectory.

For each (α, starting-position) pair we report

  - per-step NRMSE (averaged over the field dim)
  - integrated NRMSE (mean over rollout_steps)
  - valid prediction time (Pearson corr against ref drops below 0.8)

Aggregated across starting positions we report per-α mean + std and a
top-level mean across αs.

CLI::

    python -m disco.eval_g1 \\
        --config configs/disco/hw2d_multi.yaml \\
        --checkpoint $SCRATCH/IEEE/output/<run>/checkpoints/best.pt \\
        --data-root /work2/.../IEEE/hw2d \\
        --alphas 0.1 1.0 5.0 \\
        --rollout-steps 64 \\
        --n-starts 16 \\
        --out $SCRATCH/IEEE/output/eval/<run>_g1

Outputs:

  - ``rollout.jsonl`` — one record per (α, start) with per-step NRMSE arrays
  - ``summary.yaml`` — per-α aggregates + headline mean
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from disco.config_hw2d import TrainConfig, load_config
from disco.dataset_specs import HW2D_ALPHA_META
from disco.hw2d_dataset import HW2DDataset
from disco import train_hw2d as th  # _build_model, _alpha_key


# ---------------------------------------------------------------------------
# rollout core
# ---------------------------------------------------------------------------

@torch.no_grad()
def _rollout_one(
    model,
    initial_window: torch.Tensor,
    n_steps: int,
    state_labels: torch.Tensor,
    dset_name: str,
) -> torch.Tensor:
    """Autoregressive 1-step rollout.

    Parameters
    ----------
    initial_window : (1, n_past, C, H, W) tensor on the model's device.
    n_steps : number of future frames to predict.

    Returns
    -------
    (n_steps, C, H, W) tensor of predicted frames.
    """
    window = initial_window.clone()
    n_past = window.shape[1]
    preds = []
    for _ in range(n_steps):
        y_pred, _meta = model(
            window, predict_normed=False,
            state_labels=state_labels, dset_name=dset_name,
        )
        # y_pred: (1, n_future=1, C, H, W) -> take frame 0
        next_frame = y_pred[:, 0:1]  # (1, 1, C, H, W)
        preds.append(next_frame[0, 0])
        window = torch.cat([window[:, 1:], next_frame], dim=1)
    return torch.stack(preds, dim=0)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _per_step_nrmse(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """NRMSE at each rollout step, averaged across fields.

    pred, ref : (T, C, H, W)
    Returns (T,) array.
    """
    out = np.empty(pred.shape[0], dtype=np.float64)
    for t in range(pred.shape[0]):
        diff = pred[t] - ref[t]
        rmse = np.sqrt(np.mean(diff ** 2))
        denom = np.std(ref[t], ddof=1)
        out[t] = float(rmse / denom) if denom > 1e-15 else float("nan")
    return out


def _valid_prediction_steps(
    pred: np.ndarray, ref: np.ndarray, threshold: float = 0.8
) -> int:
    """First step at which Pearson corr (over flattened field) < threshold.

    Returns the number of *valid* steps (i.e. T if it never drops).
    """
    T = pred.shape[0]
    for t in range(T):
        p = pred[t].ravel()
        r = ref[t].ravel()
        if np.std(p) < 1e-15 or np.std(r) < 1e-15:
            return t
        c = float(np.corrcoef(p, r)[0, 1])
        if c < threshold:
            return t
    return T


# ---------------------------------------------------------------------------
# checkpoint loading
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
        print(f"[eval_g1] state_dict mismatch: missing={len(missing)} "
              f"unexpected={len(unexpected)}")
        if missing:
            print(f"  e.g. missing[:3] = {missing[:3]}")
        if unexpected:
            print(f"  e.g. unexpected[:3] = {unexpected[:3]}")
    model.eval()
    return model, ckpt.get("epoch", -1)


# ---------------------------------------------------------------------------
# per-α evaluation
# ---------------------------------------------------------------------------

def evaluate_alpha(
    model,
    cfg: TrainConfig,
    data_root: str,
    alpha: float,
    rollout_steps: int,
    n_starts: int,
    state_labels: torch.Tensor,
    device: torch.device,
    seed: int = 0,
) -> Tuple[List[dict], dict]:
    """Run ``n_starts`` rollouts over the G1 tail of one α.

    Returns (per_start_records, per_alpha_summary).
    """
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
    # We need rollout_steps consecutive future frames, so the *starting*
    # snippet must leave room. The dataset returns (input, output) windows
    # at index i; we use input as the seed window and pull the ref frames
    # directly from the underlying trajectory cache.
    n_avail = len(dset)
    if n_avail < rollout_steps + 1:
        raise RuntimeError(
            f"α={alpha}: G1 tail too short ({n_avail} samples) for "
            f"{rollout_steps}-step rollout"
        )
    rng = np.random.default_rng(seed)
    # Pick `n_starts` starting samples whose rollout horizon fits.
    last_valid = n_avail - rollout_steps - 1
    if last_valid <= 0:
        starts = [0]
    else:
        starts = sorted(rng.choice(last_valid, size=min(n_starts, last_valid),
                                   replace=False).tolist())

    records: List[dict] = []
    all_nrmse = []
    all_vps = []
    for s in starts:
        seed_sample = dset[s]
        x0 = seed_sample["input_fields"].unsqueeze(0).to(device)  # (1, P, C, H, W)
        # Reference: pull rollout_steps frames after the seed window.
        # dset[s] uses frames [s+offset .. s+offset+n_past+n_future). The
        # *first predicted* frame corresponds to dset[s+n_past-... ] —
        # easier to assemble from consecutive output_fields:
        ref = []
        for k in range(rollout_steps):
            samp = dset[s + k]
            ref.append(samp["output_fields"][0])  # (C, H, W), n_future=1
        ref_np = torch.stack(ref, dim=0).numpy()

        t0 = time.time()
        pred = _rollout_one(model, x0, rollout_steps, state_labels,
                            cfg.dataset_name)
        elapsed = time.time() - t0
        pred_np = pred.cpu().float().numpy()

        per_step = _per_step_nrmse(pred_np, ref_np)
        vps = _valid_prediction_steps(pred_np, ref_np, threshold=0.8)
        records.append({
            "alpha": alpha,
            "start": int(s),
            "rollout_steps": rollout_steps,
            "nrmse_per_step": per_step.tolist(),
            "nrmse_integrated": float(np.nanmean(per_step)),
            "valid_pred_steps": int(vps),
            "wall_s": float(elapsed),
        })
        all_nrmse.append(per_step)
        all_vps.append(vps)

    nrmse_arr = np.stack(all_nrmse, axis=0)  # (n_starts, rollout_steps)
    summary = {
        "alpha": alpha,
        "n_starts": len(starts),
        "rollout_steps": rollout_steps,
        "nrmse_integrated_mean": float(np.nanmean(nrmse_arr)),
        "nrmse_integrated_std": float(np.nanstd(np.nanmean(nrmse_arr, axis=1))),
        "nrmse_per_step_mean": np.nanmean(nrmse_arr, axis=0).tolist(),
        "nrmse_final_step_mean": float(np.nanmean(nrmse_arr[:, -1])),
        "valid_pred_steps_mean": float(np.mean(all_vps)),
        "valid_pred_steps_min": int(np.min(all_vps)),
    }
    return records, summary


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="DISCO G1 short-horizon eval")
    ap.add_argument("--config", required=True, type=Path,
                    help="Training YAML used to fit the checkpoint")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=str)
    ap.add_argument("--alphas", nargs="+", type=float, required=True)
    ap.add_argument("--rollout-steps", type=int, default=64)
    ap.add_argument("--n-starts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[eval_g1] loading {args.checkpoint}")
    model, ckpt_epoch = _load_model_from_checkpoint(
        cfg, args.checkpoint, device, args.data_root,
    )
    print(f"[eval_g1] model built; checkpoint epoch={ckpt_epoch}")

    state_labels = torch.tensor([0, 1], dtype=torch.long, device=device)

    all_records: List[dict] = []
    per_alpha: Dict[str, dict] = {}
    for alpha in args.alphas:
        print(f"[eval_g1] α={alpha}: rolling out {args.n_starts} starts × "
              f"{args.rollout_steps} steps")
        recs, summ = evaluate_alpha(
            model, cfg, args.data_root, alpha, args.rollout_steps,
            args.n_starts, state_labels, device, seed=args.seed,
        )
        all_records.extend(recs)
        per_alpha[th._alpha_key(alpha)] = summ
        print(f"  → mean NRMSE = {summ['nrmse_integrated_mean']:.4f} "
              f"(final={summ['nrmse_final_step_mean']:.4f}, "
              f"VPS={summ['valid_pred_steps_mean']:.1f}/"
              f"{args.rollout_steps})")

    with (args.out / "rollout.jsonl").open("w") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")

    headline = float(np.mean([per_alpha[k]["nrmse_integrated_mean"]
                              for k in per_alpha]))
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(ckpt_epoch) if ckpt_epoch >= 0 else None,
        "config": str(args.config),
        "rollout_steps": args.rollout_steps,
        "n_starts": args.n_starts,
        "seed": args.seed,
        "alphas": list(args.alphas),
        "per_alpha": per_alpha,
        "nrmse_integrated_mean_across_alphas": headline,
    }
    with (args.out / "summary.yaml").open("w") as fh:
        yaml.safe_dump(summary, fh, sort_keys=False)
    print(f"[eval_g1] headline mean NRMSE across αs = {headline:.4f}")
    print(f"[eval_g1] wrote {args.out}/rollout.jsonl + summary.yaml")


if __name__ == "__main__":
    main()
