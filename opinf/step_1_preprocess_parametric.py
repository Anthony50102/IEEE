"""
Step 1 (parametric): preprocess multi-alpha data for affine-mu pOpInf (B2).

Pipeline:
  1. For each training alpha, distributed-load its trajectory covering the
     union of train and test ranges.
  2. Compute per-alpha temporal mean over the training range; center both
     train and test by it.
  3. Concatenate centered training snapshots across all alphas along time
     (pooled training matrix).
  4. Pooled POD via the existing distributed Gram-matrix path.
  5. Project each alpha's centered train and test onto the pooled basis.
  6. Assemble the alpha-blocked parametric data matrix D and target Y on
     rank 0 via parametric_data.build_parametric_data_matrix.
  7. If a sentinel alpha is configured, load + project its trajectory too
     (for stability check IC in step_2 and G3 evaluation in step_3).
  8. Save all artifacts under run_dir.

Theorem 2.3 sanity (Theta condition number) is logged on rank 0.

Usage:
    mpirun -np 4 python step_1_preprocess_parametric.py \
        --config ../configs/opinf/b2_alpha_p015.yaml

Author: Anthony Poole
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import numpy as np
from mpi4py import MPI

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))  # for shared/

import h5py

from utils import (
    distribute_indices, setup_logging, DummyLogger, print_header,
    save_step_status,
)
from data import (
    get_file_metadata, load_distributed_snapshots,
)
from pod import compute_pod_distributed

from parametric_config import (
    load_parametric_config, ParametricConfig, AlphaSpec,
    sentinel_trajectory_path,
)
from parametric_data import (
    check_affine_feature_rank,
    build_parametric_data_matrix,
    parametric_data_matrix_row_widths,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ParamCfgAdapter:
    """Minimal shim so legacy data.get_file_metadata works on a ParametricConfig."""
    def __init__(self, pcfg: ParametricConfig):
        self.pde = pcfg.pde
        self.n_fields = pcfg.n_fields
        self.engine = pcfg.engine
        self.dt = pcfg.dt
        self.truncation_enabled = False
        self.truncation_snapshots = None
        self.truncation_time = None


def _load_alpha_trajectory(spec: AlphaSpec, cfg: ParametricConfig,
                           comm, rank, size, logger):
    """Load a single alpha's distributed snapshots covering both train and test."""
    adapter = _ParamCfgAdapter(cfg)
    fp = spec.trajectory_path
    if rank == 0:
        n_spatial, n_time_total, _ = get_file_metadata(adapter, fp)
        if max(spec.train_end, spec.test_end) > n_time_total:
            raise ValueError(
                f"alpha={spec.alpha}: train/test ranges exceed trajectory "
                f"length ({n_time_total})"
            )
        info = {"n_spatial": n_spatial, "n_time_total": n_time_total}
    else:
        info = None
    info = comm.bcast(info, root=0)
    n_spatial = info["n_spatial"]

    start_idx, end_idx, n_local = distribute_indices(rank, n_spatial, size)
    max_snap = max(spec.train_end, spec.test_end)
    Q_local = load_distributed_snapshots(
        fp, start_idx, end_idx, cfg.engine, max_snap, pde=cfg.pde,
    )
    # Cast to float64 (Frontera B1 runs already use float64 at this stage).
    Q_local = np.asarray(Q_local, dtype=np.float64)

    Q_train_local = Q_local[:, spec.train_start:spec.train_end].copy()
    Q_test_local = Q_local[:, spec.test_start:spec.test_end].copy()
    del Q_local
    gc.collect()

    if rank == 0:
        logger.info(
            f"  alpha={spec.alpha}: n_spatial={n_spatial:,}, "
            f"train=[{spec.train_start}, {spec.train_end}) ({Q_train_local.shape[1]}), "
            f"test=[{spec.test_start}, {spec.test_end}) ({Q_test_local.shape[1]})"
        )

    return Q_train_local, Q_test_local, n_spatial, n_local, start_idx, end_idx


def _gather_full_xhat(Xhat_local, comm, rank):
    """Helper: Xhat at this point is already (n_time, r) replicated everywhere
    because project_data_distributed broadcasts. We don't actually use this
    in our flow — Xhat from manual projection here is already full. Kept for
    clarity / future use."""
    return Xhat_local


def _project_onto_basis(Q_local: np.ndarray, Ur_local: np.ndarray,
                        comm) -> np.ndarray:
    """Compute Xhat = Ur^T @ Q, distributed-reducing the spatial sum.

    Q_local: (n_local_spatial, n_time)
    Ur_local: (n_local_spatial, r)
    Returns: (n_time, r) — replicated on every rank.
    """
    local_dot = Ur_local.T @ Q_local           # (r, n_time)
    r, n_time = local_dot.shape
    Xhat = np.zeros((r, n_time), dtype=np.float64)
    comm.Allreduce(local_dot, Xhat, op=MPI.SUM)
    return Xhat.T                              # (n_time, r)


def _gather_full_basis(Ur_local: np.ndarray, n_spatial: int,
                       comm, rank, size) -> np.ndarray:
    """Gather the distributed POD basis to rank 0."""
    counts = comm.gather(Ur_local.shape[0], root=0)
    if rank == 0:
        r = Ur_local.shape[1]
        Ur_full = np.zeros((n_spatial, r), dtype=np.float64)
        recvbuf = [Ur_full,
                   ([c * r for c in counts],
                    [int(np.sum(counts[:i]) * r) for i in range(size)]),
                   MPI.DOUBLE]
    else:
        Ur_full = None
        recvbuf = None
    sendbuf = np.ascontiguousarray(Ur_local, dtype=np.float64)
    comm.Gatherv(sendbuf, recvbuf, root=0)
    return Ur_full


def _load_gamma_reference(fp: str, train_start: int, train_end: int,
                          test_start: int, test_end: int) -> dict:
    """Load gamma_n, gamma_c for one alpha (rank-0 only path)."""
    with h5py.File(fp, "r") as fh:
        gn = np.asarray(fh["gamma_n"][:])
        gc_ = np.asarray(fh["gamma_c"][:])
    return {
        "gamma_n_train": gn[train_start:train_end],
        "gamma_c_train": gc_[train_start:train_end],
        "gamma_n_test": gn[test_start:test_end],
        "gamma_c_test": gc_[test_start:test_end],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None,
                        help="Existing run dir; if omitted, one is created")
    args = parser.parse_args()

    cfg = load_parametric_config(args.config)

    # Set up run dir on rank 0
    if rank == 0:
        if args.run_dir and os.path.isdir(args.run_dir):
            run_dir = args.run_dir
        else:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(cfg.output_base,
                                   f"{ts}_{cfg.run_name}" if cfg.run_name else ts)
            os.makedirs(run_dir, exist_ok=True)
        cfg.run_dir = run_dir
    else:
        run_dir = None
    run_dir = comm.bcast(run_dir, root=0)
    cfg.run_dir = run_dir

    logger = (setup_logging("step_1_param", run_dir, cfg.log_level, rank)
              if rank == 0 else DummyLogger())

    if rank == 0:
        print_header("STEP 1 (PARAMETRIC): MULTI-ALPHA PREPROCESSING + POOLED POD")
        logger.info(f"  Run directory: {run_dir}")
        logger.info(f"  MPI ranks: {size}")
        logger.info(f"  Training alphas: {cfg.training_alpha_values}")
        logger.info(f"  Sentinel alpha: {cfg.sentinel_alpha}")
        logger.info(f"  r = {cfg.r}, target_energy = {cfg.target_energy}")
        save_step_status(run_dir, "step_1_param", "running")

    t_start = MPI.Wtime()

    try:
        # =================================================================
        # Theorem 2.3 well-posedness sanity (rank 0)
        # =================================================================
        if rank == 0:
            rep = check_affine_feature_rank(cfg.training_alpha_values, q=2)
            logger.info(
                f"  Theta cond (q=2, s={rep['s']}): {rep['condition_number']:.3e}, "
                f"full_col_rank={rep['full_column_rank']}"
            )
            if not rep["full_column_rank"]:
                raise RuntimeError(
                    "Affine feature matrix Theta is rank-deficient — "
                    "the regression is ill-posed. Check training alphas."
                )
            if rep["condition_number"] > 1e3:
                logger.warning(
                    f"  Theta is poorly conditioned (cond={rep['condition_number']:.2e}); "
                    "regression will need careful regularization."
                )

        # =================================================================
        # Load each alpha, per-alpha center, store local centered arrays
        # =================================================================
        Q_train_local_per_alpha = {}
        Q_test_local_per_alpha = {}
        train_means_local = {}      # per-alpha temporal mean over training range, local slab
        K_train_per_alpha = {}
        K_test_per_alpha = {}
        n_spatial_ref = None
        n_local_ref = None

        for spec in cfg.training_alphas:
            Q_tr, Q_te, n_spatial, n_local, _, _ = _load_alpha_trajectory(
                spec, cfg, comm, rank, size, logger
            )
            if n_spatial_ref is None:
                n_spatial_ref = n_spatial
                n_local_ref = n_local
            elif n_spatial != n_spatial_ref:
                raise RuntimeError(
                    f"n_spatial mismatch across alphas: "
                    f"alpha={spec.alpha} has {n_spatial}, expected {n_spatial_ref}"
                )

            # Per-alpha centering
            if cfg.centering_enabled:
                mean_local = np.mean(Q_tr, axis=1, keepdims=True)
                Q_tr_c = Q_tr - mean_local
                Q_te_c = Q_te - mean_local
                train_means_local[spec.alpha] = mean_local.squeeze()
            else:
                Q_tr_c = Q_tr
                Q_te_c = Q_te
                train_means_local[spec.alpha] = np.zeros(Q_tr.shape[0])

            Q_train_local_per_alpha[spec.alpha] = Q_tr_c
            Q_test_local_per_alpha[spec.alpha] = Q_te_c
            K_train_per_alpha[spec.alpha] = Q_tr_c.shape[1]
            K_test_per_alpha[spec.alpha] = Q_te_c.shape[1]

        # =================================================================
        # Pool centered training snapshots across alphas
        # =================================================================
        pooled_train_local = np.concatenate(
            [Q_train_local_per_alpha[a.alpha] for a in cfg.training_alphas],
            axis=1,
        )  # (n_local, sum_K_train)
        if rank == 0:
            logger.info(
                f"  Pooled training matrix: shape ({n_local_ref}, "
                f"{pooled_train_local.shape[1]}) across {len(cfg.training_alphas)} alphas"
            )

        # =================================================================
        # Pooled POD via distributed Gram matrix
        # =================================================================
        eigs, eigv, D_global, r_energy = compute_pod_distributed(
            pooled_train_local, comm, rank, size, logger, cfg.target_energy
        )
        r_actual = min(cfg.r, r_energy) if cfg.r > 0 else r_energy
        if rank == 0:
            logger.info(
                f"  r_actual={r_actual} (config r={cfg.r}, "
                f"energy-based r={r_energy} at {cfg.target_energy*100:.1f}%)"
            )

        # Build Ur_local from pooled eigvecs
        eigs_safe = np.where(eigs[:r_actual] > 1e-14, eigs[:r_actual], 1e-14)
        Tr = eigv[:, :r_actual] @ np.diag(eigs_safe ** (-0.5))
        Ur_local = pooled_train_local @ Tr        # (n_local, r)

        # Sanity: Ur^T Ur ≈ I (Allreduce over spatial)
        UtU_local = Ur_local.T @ Ur_local
        UtU_global = np.zeros_like(UtU_local)
        comm.Allreduce(UtU_local, UtU_global, op=MPI.SUM)
        if rank == 0:
            err = np.linalg.norm(UtU_global - np.eye(r_actual))
            logger.info(f"  ||Ur^T Ur - I|| = {err:.3e}")

        # Free pooled (not needed past basis build)
        del pooled_train_local, D_global
        gc.collect()

        # =================================================================
        # Project each alpha's train and test onto pooled basis
        # =================================================================
        Xhat_train_per_alpha = {}      # (K_train, r), rank 0 and replicated
        Xhat_test_per_alpha = {}

        for spec in cfg.training_alphas:
            Xtr = _project_onto_basis(
                Q_train_local_per_alpha[spec.alpha], Ur_local, comm
            )
            Xte = _project_onto_basis(
                Q_test_local_per_alpha[spec.alpha], Ur_local, comm
            )
            Xhat_train_per_alpha[spec.alpha] = Xtr
            Xhat_test_per_alpha[spec.alpha] = Xte
            if rank == 0:
                logger.info(
                    f"  alpha={spec.alpha}: Xhat_train shape {Xtr.shape}, "
                    f"Xhat_test shape {Xte.shape}"
                )

        # =================================================================
        # Build parametric data matrix on rank 0
        # =================================================================
        if rank == 0:
            X_pred = {a.alpha: Xhat_train_per_alpha[a.alpha][:-1]
                      for a in cfg.training_alphas}
            Y_targ = {a.alpha: Xhat_train_per_alpha[a.alpha][1:]
                      for a in cfg.training_alphas}
            bundle = build_parametric_data_matrix(X_pred, Y_targ)
            D = bundle["D"]
            Y = bundle["Y"]
            layout = bundle["layout"]
            K_per_alpha = bundle["K_per_alpha"]
            alpha_labels = bundle["alphas"]
            logger.info(
                f"  Parametric D shape: {D.shape}, Y shape: {Y.shape}, "
                f"total cols: {layout['total']}"
            )
            # Theorem 2.5 check: K_total vs minimum required
            K_total = sum(K_per_alpha.values())
            min_required = layout["total"]
            ratio = K_total / max(1, min_required)
            logger.info(
                f"  K_total={K_total}, min required (cols)={min_required}, "
                f"over-det ratio={ratio:.2f}x"
            )
            if ratio < 1.0:
                logger.warning(
                    "  Under-determined regression (K_total < cols); "
                    "regularization is mandatory."
                )

        # =================================================================
        # Sentinel alpha: load + project (no centering vs training basis)
        # =================================================================
        Xhat_sentinel = None
        sentinel_mean_local = None
        sentinel_path = sentinel_trajectory_path(cfg)
        if sentinel_path is not None:
            if rank == 0:
                logger.info(f"  Loading sentinel alpha={cfg.sentinel_alpha} "
                            f"from {sentinel_path}")
            spec_s = AlphaSpec(
                alpha=cfg.sentinel_alpha,
                data_dir=cfg.sentinel_data_dir,
                training_file=cfg.sentinel_training_file,
                train_start=cfg.sentinel_eval_start,
                train_end=cfg.sentinel_eval_end,
                test_start=cfg.sentinel_eval_start,
                test_end=cfg.sentinel_eval_end,
            )
            Q_s_local, _, _, _, _, _ = _load_alpha_trajectory(
                spec_s, cfg, comm, rank, size, logger
            )
            if cfg.centering_enabled:
                sentinel_mean_local = np.mean(Q_s_local, axis=1, keepdims=True)
                Q_s_c = Q_s_local - sentinel_mean_local
                sentinel_mean_local = sentinel_mean_local.squeeze()
            else:
                Q_s_c = Q_s_local
                sentinel_mean_local = np.zeros(Q_s_local.shape[0])
            Xhat_sentinel = _project_onto_basis(Q_s_c, Ur_local, comm)
            del Q_s_local, Q_s_c
            gc.collect()
            if rank == 0:
                logger.info(f"  Xhat_sentinel shape: {Xhat_sentinel.shape}")

        # =================================================================
        # Gather full basis + means to rank 0; save artifacts
        # =================================================================
        Ur_full = _gather_full_basis(Ur_local, n_spatial_ref, comm, rank, size)

        train_means_full = {}
        for alpha, mean_local in train_means_local.items():
            mean_gathered = comm.gather(mean_local, root=0)
            if rank == 0:
                train_means_full[alpha] = np.concatenate(mean_gathered)

        if sentinel_mean_local is not None:
            s_mean_gathered = comm.gather(sentinel_mean_local, root=0)
            sentinel_mean_full = (np.concatenate(s_mean_gathered)
                                  if rank == 0 else None)
        else:
            sentinel_mean_full = None

        if rank == 0:
            out = os.path.join(run_dir, "preprocess_parametric.npz")
            payload = {
                "D": D,
                "Y": Y,
                "layout_total": np.int64(layout["total"]),
                "layout_widths": np.array(
                    [layout["widths"][k] for k in
                     ("A0", "A1", "F0", "F1", "c0", "c1")],
                    dtype=np.int64,
                ),
                "Ur": Ur_full,
                "eigs": eigs,
                "r": np.int64(r_actual),
                "training_alphas": np.array(cfg.training_alpha_values),
                "alpha_labels": alpha_labels,
                "n_spatial": np.int64(n_spatial_ref),
            }
            # Per-alpha trajectories (in reduced coords) — used by step_2 sweep
            # and step_3 evaluation
            for spec in cfg.training_alphas:
                a = spec.alpha
                payload[f"Xhat_train_alpha_{a:g}"] = Xhat_train_per_alpha[a]
                payload[f"Xhat_test_alpha_{a:g}"] = Xhat_test_per_alpha[a]
                payload[f"train_mean_alpha_{a:g}"] = train_means_full[a]
            if cfg.sentinel_alpha is not None and Xhat_sentinel is not None:
                payload[f"Xhat_sentinel_alpha_{cfg.sentinel_alpha:g}"] = (
                    Xhat_sentinel
                )
                if sentinel_mean_full is not None:
                    payload[f"sentinel_mean_alpha_{cfg.sentinel_alpha:g}"] = (
                        sentinel_mean_full
                    )
                payload["sentinel_alpha"] = np.float64(cfg.sentinel_alpha)
            np.savez_compressed(out, **payload)
            logger.info(f"  Saved preprocess bundle: {out}")

            # Per-alpha gamma references
            gamma_payload = {}
            for spec in cfg.training_alphas:
                g = _load_gamma_reference(
                    spec.trajectory_path,
                    spec.train_start, spec.train_end,
                    spec.test_start, spec.test_end,
                )
                for k, v in g.items():
                    gamma_payload[f"{k}_alpha_{spec.alpha:g}"] = v
            if sentinel_path is not None:
                g = _load_gamma_reference(
                    sentinel_path,
                    cfg.sentinel_eval_start, cfg.sentinel_eval_end,
                    cfg.sentinel_eval_start, cfg.sentinel_eval_end,
                )
                for k, v in g.items():
                    gamma_payload[f"{k}_sentinel_alpha_{cfg.sentinel_alpha:g}"] = v
            gpath = os.path.join(run_dir, "gamma_reference_parametric.npz")
            np.savez_compressed(gpath, **gamma_payload)
            logger.info(f"  Saved gamma reference: {gpath}")

        total_time = MPI.Wtime() - t_start
        if rank == 0:
            save_step_status(run_dir, "step_1_param", "completed", {
                "r": int(r_actual),
                "n_spatial": int(n_spatial_ref),
                "mpi_ranks": size,
                "total_time_seconds": float(total_time),
                "training_alphas": list(cfg.training_alpha_values),
                "sentinel_alpha": (None if cfg.sentinel_alpha is None
                                   else float(cfg.sentinel_alpha)),
            })
            print_header("STEP 1 (PARAMETRIC) COMPLETE")
            logger.info(f"  Runtime: {total_time:.1f}s")

    except Exception as e:
        if rank == 0:
            logger.error(f"Step 1 (parametric) failed: {e}", exc_info=True)
            save_step_status(run_dir, "step_1_param", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
