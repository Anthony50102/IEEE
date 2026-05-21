"""Plot a rollout HDF5: state MP4 animation + Γₙ/Γc time-series + NRMSE-vs-t.

Author: Anthony Poole

Reads an HDF5 produced by :mod:`disco.rollout` and emits:

  - ``state_{a}.mp4``  : 2×2 grid (GT n / pred n / GT φ / pred φ)
  - ``gamma_n_{a}.png`` : Γₙ(t) pred vs GT
  - ``gamma_c_{a}.png`` : Γc(t) pred vs GT
  - ``qoi_panel_{a}.png``: stacked Γₙ/Γc with NRMSE twin axis
  - ``nrmse_{a}.png``   : per-step NRMSE (standalone)

Designed to run locally (no GPU, no model imports). Just matplotlib +
h5py + numpy. Requires ffmpeg on PATH for the MP4 writer.

CLI::

    python -m disco.plot_rollout --h5 rollout_a10.h5 --out plots/multi/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def _load(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        d = {k: np.asarray(f[k]) for k in (
            "pred_n", "pred_phi", "ref_n", "ref_phi",
            "pred_gamma_n", "pred_gamma_c",
            "ref_gamma_n", "ref_gamma_c",
            "nrmse_per_step",
        )}
        attrs = dict(f.attrs)
    d["attrs"] = attrs
    return d


def _time_axis(attrs: dict, T: int) -> Tuple[np.ndarray, str]:
    dt = float(attrs.get("dt", float("nan")))
    if np.isfinite(dt):
        return dt * np.arange(T), "t  (sim units)"
    return np.arange(T, dtype=float), "step"


# ---------------------------------------------------------------------------
# animation
# ---------------------------------------------------------------------------

def make_state_animation(
    data: dict, out_mp4: Path, fps: int = 20, max_frames: int = 600,
) -> None:
    """2x2 state animation (GT n, pred n, GT φ, pred φ).

    To keep MP4 size reasonable we subsample to at most ``max_frames``.
    """
    ref_n = data["ref_n"]; pred_n = data["pred_n"]
    ref_phi = data["ref_phi"]; pred_phi = data["pred_phi"]
    T = min(ref_n.shape[0], pred_n.shape[0])
    if T > max_frames:
        stride = int(np.ceil(T / max_frames))
        idx = np.arange(0, T, stride)
    else:
        idx = np.arange(T)
    alpha = float(data["attrs"].get("alpha", float("nan")))
    dt = float(data["attrs"].get("dt", float("nan")))

    # shared symmetric color scale per row (5/95 pct of GT)
    n_vmax = float(np.percentile(np.abs(ref_n), 99))
    p_vmax = float(np.percentile(np.abs(ref_phi), 99))
    n_kw = dict(vmin=-n_vmax, vmax=n_vmax, cmap="RdBu_r", origin="lower")
    p_kw = dict(vmin=-p_vmax, vmax=p_vmax, cmap="RdBu_r", origin="lower")

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    im00 = axes[0, 0].imshow(ref_n[idx[0]], **n_kw);  axes[0, 0].set_title("GT density")
    im01 = axes[0, 1].imshow(pred_n[idx[0]], **n_kw); axes[0, 1].set_title("pred density")
    im10 = axes[1, 0].imshow(ref_phi[idx[0]], **p_kw);  axes[1, 0].set_title("GT φ")
    im11 = axes[1, 1].imshow(pred_phi[idx[0]], **p_kw); axes[1, 1].set_title("pred φ")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im00, ax=axes[0, :].tolist(), shrink=0.7, label="n")
    fig.colorbar(im10, ax=axes[1, :].tolist(), shrink=0.7, label="φ")
    title = fig.suptitle("", fontsize=12)

    def _fmt_t(k):
        if np.isfinite(dt):
            return f"α={alpha:g}   t = {dt*k:7.3f}   (step {k:5d})"
        return f"α={alpha:g}   step {k:5d}"

    def update(i):
        k = int(idx[i])
        im00.set_data(ref_n[k]); im01.set_data(pred_n[k])
        im10.set_data(ref_phi[k]); im11.set_data(pred_phi[k])
        title.set_text(_fmt_t(k))
        return im00, im01, im10, im11, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(idx), interval=1000 // fps, blit=False,
    )
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(str(out_mp4), writer=writer, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_mp4}  ({len(idx)} frames @ {fps} fps)")


# ---------------------------------------------------------------------------
# QoI plots
# ---------------------------------------------------------------------------

def _plot_qoi_pair(
    t: np.ndarray, t_label: str,
    pred: np.ndarray, ref: np.ndarray,
    ylabel: str, out_png: Path, title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    ax.plot(t, ref, color="black", lw=1.5, label="GT")
    ax.plot(t, pred, color="C1", lw=1.5, ls="--", label="pred")
    ax.axhline(np.mean(ref), color="black", lw=0.8, alpha=0.4,
               label=f"GT mean = {np.mean(ref):.3g}")
    ax.axhline(np.mean(pred), color="C1", lw=0.8, alpha=0.4, ls="--",
               label=f"pred mean = {np.mean(pred):.3g}")
    ax.set_xlabel(t_label); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_png}")


def make_qoi_plots(data: dict, out_dir: Path, tag: str) -> None:
    T = min(data["pred_gamma_n"].shape[0], data["ref_gamma_n"].shape[0])
    t, t_label = _time_axis(data["attrs"], T)
    alpha = float(data["attrs"].get("alpha", float("nan")))

    _plot_qoi_pair(
        t, t_label, data["pred_gamma_n"][:T], data["ref_gamma_n"][:T],
        r"$\Gamma_n$", out_dir / f"gamma_n_{tag}.png",
        rf"Γₙ(t)   α={alpha:g}",
    )
    _plot_qoi_pair(
        t, t_label, data["pred_gamma_c"][:T], data["ref_gamma_c"][:T],
        r"$\Gamma_c$", out_dir / f"gamma_c_{tag}.png",
        rf"Γc(t)   α={alpha:g}",
    )


def make_qoi_panel(data: dict, out_png: Path) -> None:
    T = min(
        data["pred_gamma_n"].shape[0], data["ref_gamma_n"].shape[0],
        data["nrmse_per_step"].shape[0],
    )
    t, t_label = _time_axis(data["attrs"], T)
    alpha = float(data["attrs"].get("alpha", float("nan")))

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True,
                             constrained_layout=True)
    ax_n, ax_c = axes

    ax_n.plot(t, data["ref_gamma_n"][:T], "k-", lw=1.4, label="GT")
    ax_n.plot(t, data["pred_gamma_n"][:T], "C1--", lw=1.4, label="pred")
    ax_n.axhline(np.mean(data["ref_gamma_n"][:T]), color="k", lw=0.6, alpha=0.4)
    ax_n.axhline(np.mean(data["pred_gamma_n"][:T]), color="C1", lw=0.6,
                 alpha=0.4, ls="--")
    ax_n.set_ylabel(r"$\Gamma_n$"); ax_n.legend(loc="best", fontsize=8)
    ax_n.grid(True, alpha=0.3)
    ax_n.set_title(rf"DISCO rollout QoIs   α={alpha:g}")

    ax_c.plot(t, data["ref_gamma_c"][:T], "k-", lw=1.4, label="GT")
    ax_c.plot(t, data["pred_gamma_c"][:T], "C1--", lw=1.4, label="pred")
    ax_c.set_ylabel(r"$\Gamma_c$"); ax_c.grid(True, alpha=0.3)

    ax_t = ax_c.twinx()
    ax_t.plot(t, data["nrmse_per_step"][:T], color="C3", lw=1.0, alpha=0.6,
              label="NRMSE")
    ax_t.set_ylabel("NRMSE", color="C3")
    ax_t.tick_params(axis="y", labelcolor="C3")
    ax_c.set_xlabel(t_label)
    ax_c.legend(loc="upper left", fontsize=8)

    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_png}")


def make_nrmse_plot(data: dict, out_png: Path) -> None:
    arr = data["nrmse_per_step"]
    T = arr.shape[0]
    t, t_label = _time_axis(data["attrs"], T)
    alpha = float(data["attrs"].get("alpha", float("nan")))

    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    ax.plot(t, arr, color="C3", lw=1.2)
    ax.axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.5,
               label="NRMSE = 1 (climatology)")
    ax.set_xlabel(t_label); ax.set_ylabel("NRMSE")
    ax.set_title(rf"Per-step NRMSE   α={alpha:g}")
    ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_png}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def _tag_from_attrs(attrs: dict) -> str:
    alpha = float(attrs.get("alpha", float("nan")))
    return f"a{int(round(alpha * 10)):02d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot a DISCO rollout HDF5")
    ap.add_argument("--h5", required=True, type=Path,
                    help="Path to rollout_a*.h5 produced by disco.rollout")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory (created if missing)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=600,
                    help="Cap on MP4 frames (subsamples if rollout is longer)")
    ap.add_argument("--no-anim", action="store_true",
                    help="Skip MP4 animation (still produces PNGs)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[plot_rollout] loading {args.h5}")
    data = _load(args.h5)
    tag = _tag_from_attrs(data["attrs"])

    make_qoi_plots(data, args.out, tag)
    make_qoi_panel(data, args.out / f"qoi_panel_{tag}.png")
    make_nrmse_plot(data, args.out / f"nrmse_{tag}.png")
    if not args.no_anim:
        make_state_animation(
            data, args.out / f"state_{tag}.mp4",
            fps=args.fps, max_frames=args.max_frames,
        )
    print(f"[plot_rollout] done; outputs in {args.out}")


if __name__ == "__main__":
    main()
