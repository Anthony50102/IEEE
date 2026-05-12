"""
Step 2 (parametric): MPI-parallel 2D Tikhonov sweep for affine-mu pOpInf (B2).

Loads preprocess bundle from step_1_preprocess_parametric.py, then sweeps
the 2D grid of (lambda1_sq, lambda2_sq) by slicing it across MPI ranks.
Each rank solves a subset of (D^T D + Lambda) O^T = D^T Y systems and
evaluates each candidate's score (mean per-alpha rollout MSE on training
trajectories) plus stability at the sentinel alpha.

Candidates with NaN rollouts or sentinel blowup are disqualified.

Outputs:
  - sweep_results_parametric.npz : all candidates (lambda1, lambda2,
    per-alpha MSE, sentinel status, disqualified flag)
  - operators_parametric.npz : the best operator blocks (A0, A1, F0, F1,
    c0, c1) plus chosen hyperparameters.

Usage:
    mpirun -np 4 python step_2_train_parametric.py \
        --config ../configs/opinf/b2_alpha_p015.yaml \
        --run-dir <RUN_DIR>

Author: Anthony Poole
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
from mpi4py import MPI

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from utils import setup_logging, DummyLogger, print_header, save_step_status
from parametric_config import load_parametric_config
from parametric_data import parametric_data_matrix_row_widths
from parametric_solve import evaluate_parametric_candidate


def _slice_grid(grid_pairs, rank, size):
    return [(i, p) for i, p in enumerate(grid_pairs) if i % size == rank]


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-rollout-steps", type=int, default=200,
                        help="Steps to roll out at each training alpha for scoring")
    args = parser.parse_args()

    cfg = load_parametric_config(args.config)
    run_dir = args.run_dir

    logger = (setup_logging("step_2_param", run_dir, cfg.log_level, rank)
              if rank == 0 else DummyLogger())

    if rank == 0:
        print_header("STEP 2 (PARAMETRIC): 2D TIKHONOV SWEEP")
        logger.info(f"  Run dir: {run_dir}")
        logger.info(f"  MPI ranks: {size}")
        logger.info(f"  lambda1_sq grid: {cfg.lambda1_sq}")
        logger.info(f"  lambda2_sq grid: {cfg.lambda2_sq}")
        save_step_status(run_dir, "step_2_param", "running")

    t_start = MPI.Wtime()

    try:
        # =================================================================
        # Load preprocess bundle (rank 0 reads, broadcasts the parts we need)
        # =================================================================
        bundle_path = os.path.join(run_dir, "preprocess_parametric.npz")
        if rank == 0:
            data = np.load(bundle_path, allow_pickle=False)
            D = data["D"].astype(np.float64)
            Y = data["Y"].astype(np.float64)
            r = int(data["r"])
            training_alphas = list(map(float, data["training_alphas"]))
            sentinel_alpha = (float(data["sentinel_alpha"])
                              if "sentinel_alpha" in data.files else None)
            # Per-alpha rollout ICs and targets (just first n_rollout_steps)
            n_steps = args.n_rollout_steps
            training_ics = {}
            training_targets = {}
            for a in training_alphas:
                Xtr = data[f"Xhat_train_alpha_{a:g}"]   # (K, r)
                training_ics[a] = Xtr[0]
                # Predict steps 1..n_steps, compare to true steps 1..n_steps
                end = min(1 + n_steps, Xtr.shape[0])
                training_targets[a] = Xtr[1:end]
                n_steps = end - 1
            sentinel_ic = None
            if sentinel_alpha is not None:
                Xsent = data[f"Xhat_sentinel_alpha_{sentinel_alpha:g}"]
                sentinel_ic = Xsent[0]
            layout = parametric_data_matrix_row_widths(r)
            bcast_payload = {
                "D": D, "Y": Y, "r": r, "layout": layout,
                "training_ics": training_ics,
                "training_targets": training_targets,
                "training_alphas": training_alphas,
                "sentinel_alpha": sentinel_alpha,
                "sentinel_ic": sentinel_ic,
                "n_steps": n_steps,
            }
            logger.info(
                f"  Loaded D{tuple(D.shape)} Y{tuple(Y.shape)} r={r} "
                f"training_alphas={training_alphas} sentinel={sentinel_alpha} "
                f"n_steps={n_steps}"
            )
        else:
            bcast_payload = None
        bcast_payload = comm.bcast(bcast_payload, root=0)

        D = bcast_payload["D"]
        Y = bcast_payload["Y"]
        r = bcast_payload["r"]
        layout = bcast_payload["layout"]
        training_ics = bcast_payload["training_ics"]
        training_targets = bcast_payload["training_targets"]
        sentinel_alpha = bcast_payload["sentinel_alpha"]
        sentinel_ic = bcast_payload["sentinel_ic"]
        n_steps = bcast_payload["n_steps"]

        # =================================================================
        # Build grid and slice across ranks
        # =================================================================
        grid_pairs = [(float(l1), float(l2))
                      for l1 in cfg.lambda1_sq for l2 in cfg.lambda2_sq]
        n_total = len(grid_pairs)
        my_slice = _slice_grid(grid_pairs, rank, size)
        if rank == 0:
            logger.info(f"  Total candidates: {n_total} "
                        f"(rank 0 takes {len(my_slice)})")

        # =================================================================
        # Sweep
        # =================================================================
        local_results = []
        for i, (l1, l2) in my_slice:
            res = evaluate_parametric_candidate(
                D=D, Y=Y, layout=layout,
                lambda1_sq=l1, lambda2_sq=l2,
                training_ics=training_ics,
                training_targets=training_targets,
                n_steps=n_steps,
                sentinel_alpha=sentinel_alpha,
                sentinel_ic=sentinel_ic,
                sentinel_steps=cfg.sentinel_steps,
                sentinel_max_norm=cfg.sentinel_max_norm,
            )
            res["grid_index"] = i
            local_results.append(res)
            if rank == 0:
                logger.info(
                    f"  [rank0 {len(local_results)}/{len(my_slice)}] "
                    f"l1_sq={l1:.2e} l2_sq={l2:.2e} "
                    f"score={res['score']:.4e} disq={res['is_disqualified']} "
                    f"sentinel_ok={res['sentinel_bounded']}"
                )

        # =================================================================
        # Gather + select best
        # =================================================================
        all_results_per_rank = comm.gather(local_results, root=0)

        if rank == 0:
            all_results = [r for sub in all_results_per_rank for r in sub]
            all_results.sort(key=lambda r: r["grid_index"])

            admitted = [r for r in all_results if not r["is_disqualified"]]
            n_disq = n_total - len(admitted)
            logger.info(
                f"  Sweep complete: {len(admitted)}/{n_total} admitted "
                f"({n_disq} disqualified)"
            )

            if not admitted:
                raise RuntimeError(
                    "All sweep candidates disqualified — increase regularization "
                    "or relax sentinel threshold."
                )

            admitted.sort(key=lambda r: r["score"])
            best = admitted[0]
            logger.info(
                f"  Best: lambda1_sq={best['lambda1_sq']:.4e}, "
                f"lambda2_sq={best['lambda2_sq']:.4e}, "
                f"score={best['score']:.4e}"
            )
            for a, mse in best["per_alpha_mse"].items():
                logger.info(f"    alpha={a}: MSE={mse:.4e}")

            # Save sweep results
            sweep_out = os.path.join(run_dir, "sweep_results_parametric.npz")
            l1_arr = np.array([r["lambda1_sq"] for r in all_results])
            l2_arr = np.array([r["lambda2_sq"] for r in all_results])
            scores = np.array([r["score"] for r in all_results])
            disq = np.array([r["is_disqualified"] for r in all_results])
            sent_ok = np.array(
                [True if r["sentinel_bounded"] is None
                 else bool(r["sentinel_bounded"])
                 for r in all_results]
            )
            # Per-alpha MSE matrix in column order matching training_alphas
            training_alphas = sorted(best["per_alpha_mse"].keys())
            per_alpha = np.array(
                [[r["per_alpha_mse"].get(a, np.inf)
                  for a in training_alphas]
                 for r in all_results]
            )
            np.savez_compressed(
                sweep_out,
                lambda1_sq=l1_arr, lambda2_sq=l2_arr,
                score=scores, disqualified=disq,
                sentinel_bounded=sent_ok,
                per_alpha_mse=per_alpha,
                training_alphas=np.array(training_alphas),
            )
            logger.info(f"  Sweep results: {sweep_out}")

            # Re-solve best to save operator blocks
            from parametric_solve import solve_parametric_opinf
            blocks = solve_parametric_opinf(
                D, Y, layout,
                lambda1_sq=best["lambda1_sq"],
                lambda2_sq=best["lambda2_sq"],
            )
            ops_out = os.path.join(run_dir, "operators_parametric.npz")
            np.savez_compressed(
                ops_out,
                **blocks,
                lambda1_sq=np.float64(best["lambda1_sq"]),
                lambda2_sq=np.float64(best["lambda2_sq"]),
                best_score=np.float64(best["score"]),
                r=np.int64(r),
            )
            logger.info(f"  Best operators: {ops_out}")

            total_time = MPI.Wtime() - t_start
            save_step_status(run_dir, "step_2_param", "completed", {
                "n_total": n_total,
                "n_admitted": len(admitted),
                "n_disqualified": n_disq,
                "lambda1_sq": float(best["lambda1_sq"]),
                "lambda2_sq": float(best["lambda2_sq"]),
                "best_score": float(best["score"]),
                "total_time_seconds": float(total_time),
            })
            print_header("STEP 2 (PARAMETRIC) COMPLETE")
            logger.info(f"  Runtime: {total_time:.1f}s")

    except Exception as e:
        if rank == 0:
            logger.error(f"Step 2 (parametric) failed: {e}", exc_info=True)
            save_step_status(run_dir, "step_2_param", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
