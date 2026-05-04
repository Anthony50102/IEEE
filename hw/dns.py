"""HW DNS runner: thin wrapper around upstream `hw2d` (the-rccg/hw2d).

Author: Anthony Poole

Reads our YAML config (a `DnsConfig` dataclass, parsed in :func:`DnsConfig.from_yaml`)
and dispatches to ``hw2d.run.run`` to generate one DNS trajectory at the
requested adiabaticity. Writes:

  <out_dir>/trajectory.h5      # hw2d output: density, omega, phi, properties
  <out_dir>/data_card.yaml     # provenance: config + computed Gamma_n/Gamma_c

Usage
-----

    python -m hw.dns --config configs/data/hw_alpha1.0_n256.yaml \\
                     --out-dir $SCRATCH/IEEE/data/hw2d/alpha1.0_n256

Notes
-----

* We simulate AND save at ``grid_pts`` (default 512). The fields are
  stored at full resolution because Gamma_n and Gamma_c are spatial
  averages over the full grid; saving downsampled fields would corrupt
  any later field-level recomputation or diagnostic. Downstream training
  can spatially stride at load time (see ``hw.dataset.iter_snippets``).
  ``downsample_factor`` is exposed for completeness but defaults to 1.
* This module imports `hw2d` lazily so the rest of `hw/` can be imported
  without the package installed.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

import numpy as np
import yaml


# === config ===

@dataclasses.dataclass
class DnsConfig:
    """One HW DNS run.

    Field names match hw2d.run.run kwargs where possible.
    """

    # Physics
    alpha: float                       # adiabaticity (= c1 in hw2d)
    kappa: float = 1.0                 # density-gradient drive (kappa_coeff)
    k0: float = 0.15
    N: int = 3                         # hyperdiffusion order (nabla^{2N})
    nu: float = 5.0e-9

    # Numerics
    grid_pts: int = 512                # simulation AND save resolution
    downsample_factor: int = 1         # >1 lossy: corrupts QoIs, avoid
    step_size: float = 0.025
    end_time: float = 1000.0
    snaps: int = 1                     # save every snaps steps

    # Initialization
    init_type: str = "normal"
    init_scale: float = 0.01
    seed: int | None = 0

    # Sampling
    recording_start_time: float = 0.0
    buffer_length: int = 100
    chunk_size: int = 100

    # Bookkeeping
    label: str = "hw_run"

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "DnsConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)


# === runner ===

def run(cfg: DnsConfig, out_dir: str | os.PathLike) -> Path:
    """Run one HW DNS trajectory and write the data_card sidecar.

    Returns the path to the generated HDF5 file.
    """
    from hw2d.run import run as hw2d_run  # noqa: PLC0415  (lazy import)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "trajectory.h5"

    hw2d_run(
        c1=cfg.alpha,
        kappa_coeff=cfg.kappa,
        k0=cfg.k0,
        N=cfg.N,
        nu=cfg.nu,
        grid_pts=cfg.grid_pts,
        step_size=cfg.step_size,
        end_time=cfg.end_time,
        snaps=cfg.snaps,
        downsample_factor=cfg.downsample_factor,
        init_type=cfg.init_type,
        init_scale=cfg.init_scale,
        seed=cfg.seed,
        output_path=str(h5_path),
        recording_start_time=cfg.recording_start_time,
        buffer_length=cfg.buffer_length,
        chunk_size=cfg.chunk_size,
        movie=False,
        debug=False,
    )

    write_data_card(cfg, h5_path, out_dir / "data_card.yaml")
    return h5_path


def write_data_card(cfg: DnsConfig, h5_path: Path, card_path: Path) -> None:
    """Write a sidecar YAML recording the config + computed Gamma stats.

    `hw/validate.py` reads this to check whether the DNS reproduced the
    hw2d reference table.
    """
    import h5py  # noqa: PLC0415

    from hw.physics import gamma_timeseries, grid

    with h5py.File(h5_path, "r") as hf:
        if "gamma_n" in hf:
            g_n = np.asarray(hf["gamma_n"])
            g_c = np.asarray(hf["gamma_c"]) if "gamma_c" in hf else None
            source = "hw2d-native"
        else:
            density = np.asarray(hf["density"])
            phi = np.asarray(hf["phi"])
            g = grid(k0=cfg.k0, nx=density.shape[-1])
            g_n, g_c = gamma_timeseries(density, phi, g["dx"], c1=cfg.alpha)
            source = "recomputed-from-fields"

        n_frames = g_n.shape[0]

    burn_in_frames = int(0.5 * n_frames)
    g_n_steady = g_n[burn_in_frames:]
    card = {
        "config": dataclasses.asdict(cfg),
        "trajectory_h5": str(h5_path.name),
        "n_frames": int(n_frames),
        "burn_in_frames_for_average": burn_in_frames,
        "gamma_n_source": source,
        "gamma_n_mean": float(np.mean(g_n_steady)),
        "gamma_n_std": float(np.std(g_n_steady)),
    }
    if g_c is not None:
        g_c_steady = g_c[burn_in_frames:]
        card["gamma_c_mean"] = float(np.mean(g_c_steady))
        card["gamma_c_std"] = float(np.std(g_c_steady))

    with open(card_path, "w") as f:
        yaml.safe_dump(card, f, sort_keys=False)


# === CLI ===

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run one HW DNS trajectory.")
    p.add_argument("--config", required=True, help="Path to DNS config YAML.")
    p.add_argument("--out-dir", required=True, help="Directory for trajectory.h5 + data_card.yaml.")
    p.add_argument(
        "--card-only",
        action="store_true",
        help=(
            "Skip simulation; recompute Gamma stats from an existing "
            "trajectory.h5 in --out-dir and (re)write data_card.yaml. "
            "Use this to salvage a run that hit a SLURM walltime before "
            "post-processing finished."
        ),
    )
    args = p.parse_args(argv)
    cfg = DnsConfig.from_yaml(args.config)
    out_dir = Path(args.out_dir)
    if args.card_only:
        h5 = out_dir / "trajectory.h5"
        if not h5.exists():
            raise SystemExit(f"--card-only: {h5} does not exist")
        write_data_card(cfg, h5, out_dir / "data_card.yaml")
        print(f"wrote {out_dir / 'data_card.yaml'}")
    else:
        h5 = run(cfg, args.out_dir)
        print(f"wrote {h5}")


if __name__ == "__main__":
    main()
