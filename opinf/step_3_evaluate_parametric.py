"""
Step 3 (parametric): evaluate B2 operators at G1 (trained alphas) and G3
(sentinel alpha).

For each training alpha:
  - roll out the parametric ROM from the test-window IC for the full test
    length and save the reduced-coordinate trajectory.
  - compute per-alpha rollout MSE in reduced coords against the projected
    test trajectory.

For the sentinel alpha (G3):
  - roll out from the first sentinel snapshot for the full sentinel range;
    save the trajectory; flag NaN/blowup.

This step is serial (rank 0). Reconstruction to full physical space and
QoI (gamma_n, gamma_c) metrics are deferred to a downstream analysis
script; the artifacts saved here contain enough information.

Usage:
    python step_3_evaluate_parametric.py \
        --config ../configs/opinf/b2_alpha_p015.yaml \
        --run-dir <RUN_DIR>

Author: Anthony Poole
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from utils import setup_logging, print_header, save_step_status
from parametric_config import load_parametric_config
from parametric_solve import rollout_at_alpha, rollout_is_bounded
from parametric_data import parametric_data_matrix_row_widths


def _rollout_and_score(blocks, alpha, x0, n_steps, target=None,
                       max_norm=1e6):
    is_nan, X = rollout_at_alpha(blocks, alpha, x0, n_steps)
    bounded = rollout_is_bounded(X, is_nan, max_state_norm=max_norm)
    mse = None
    if target is not None and bounded:
        n = min(X.shape[1], target.shape[0])
        diff = X[:, :n].T - target[:n]
        mse = float(np.mean(diff ** 2))
    return {
        "X": X,                  # (r, n_steps)
        "is_nan": bool(is_nan),
        "bounded": bool(bounded),
        "mse": mse,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-state-norm", type=float, default=1.0e6)
    args = parser.parse_args()

    cfg = load_parametric_config(args.config)
    run_dir = args.run_dir
    logger = setup_logging("step_3_param", run_dir, cfg.log_level, rank=0)

    print_header("STEP 3 (PARAMETRIC): G1 + G3 EVALUATION")
    save_step_status(run_dir, "step_3_param", "running")

    try:
        bundle = np.load(
            os.path.join(run_dir, "preprocess_parametric.npz"),
            allow_pickle=False,
        )
        ops = np.load(
            os.path.join(run_dir, "operators_parametric.npz"),
            allow_pickle=False,
        )
        r = int(ops["r"])
        blocks = {name: ops[name] for name in
                  ("A0", "A1", "F0", "F1", "c0", "c1")}
        logger.info(f"  r={r}, lambda1_sq={float(ops['lambda1_sq']):.3e}, "
                    f"lambda2_sq={float(ops['lambda2_sq']):.3e}")

        training_alphas = list(map(float, bundle["training_alphas"]))
        results = {}

        # ---- G1: trained alphas ------------------------------------------
        g1_summary = {}
        for a in training_alphas:
            Xtest = bundle[f"Xhat_test_alpha_{a:g}"]   # (K_test, r)
            if Xtest.shape[0] < 2:
                logger.warning(f"  alpha={a}: test window too short, skipping")
                continue
            x0 = Xtest[0]
            target = Xtest[1:]
            n_steps = target.shape[0]
            res = _rollout_and_score(
                blocks, a, x0, n_steps, target=target,
                max_norm=args.max_state_norm,
            )
            results[f"rollout_alpha_{a:g}"] = res["X"]
            mse_str = (f"{res['mse']:.4e}" if res["mse"] is not None
                       else "n/a")
            logger.info(
                f"  G1 alpha={a}: n_steps={n_steps} "
                f"bounded={res['bounded']} is_nan={res['is_nan']} "
                f"MSE={mse_str}"
            )
            g1_summary[a] = {
                "bounded": res["bounded"],
                "is_nan": res["is_nan"],
                "mse": res["mse"],
                "n_steps": int(n_steps),
            }

        # ---- G3: sentinel alpha -----------------------------------------
        g3_summary = None
        if cfg.sentinel_alpha is not None:
            sa = cfg.sentinel_alpha
            Xsent_key = f"Xhat_sentinel_alpha_{sa:g}"
            if Xsent_key in bundle.files:
                Xsent = bundle[Xsent_key]
                if Xsent.shape[0] >= 2:
                    x0 = Xsent[0]
                    target = Xsent[1:]
                    n_steps = target.shape[0]
                    res = _rollout_and_score(
                        blocks, sa, x0, n_steps, target=target,
                        max_norm=args.max_state_norm,
                    )
                    results[f"rollout_sentinel_alpha_{sa:g}"] = res["X"]
                    mse_str = (f"{res['mse']:.4e}" if res["mse"] is not None
                               else "n/a")
                    logger.info(
                        f"  G3 alpha={sa} (sentinel): n_steps={n_steps} "
                        f"bounded={res['bounded']} is_nan={res['is_nan']} "
                        f"MSE={mse_str}"
                    )
                    g3_summary = {
                        "alpha": sa,
                        "bounded": res["bounded"],
                        "is_nan": res["is_nan"],
                        "mse": res["mse"],
                        "n_steps": int(n_steps),
                    }
                else:
                    logger.warning(
                        f"  Sentinel alpha={sa}: trajectory too short, "
                        "skipping"
                    )
            else:
                logger.info("  No sentinel trajectory in bundle; skipping G3")

        # ---- Save ---------------------------------------------------------
        out = os.path.join(run_dir, "evaluation_parametric.npz")
        np.savez_compressed(out, **results)
        logger.info(f"  Saved rollouts: {out}")

        # Summary YAML for human inspection
        import yaml
        summary_payload = {
            "r": r,
            "lambda1_sq": float(ops["lambda1_sq"]),
            "lambda2_sq": float(ops["lambda2_sq"]),
            "best_sweep_score": float(ops["best_score"]),
            "G1": {float(a): {
                k: (v if not isinstance(v, np.generic) else float(v))
                for k, v in g.items()
            } for a, g in g1_summary.items()},
        }
        if g3_summary is not None:
            summary_payload["G3"] = {
                k: (v if not isinstance(v, np.generic) else float(v))
                for k, v in g3_summary.items()
            }
        with open(os.path.join(run_dir, "evaluation_parametric_summary.yaml"),
                  "w") as f:
            yaml.safe_dump(summary_payload, f, default_flow_style=False,
                           sort_keys=False)

        save_step_status(run_dir, "step_3_param", "completed", summary_payload)
        print_header("STEP 3 (PARAMETRIC) COMPLETE")

    except Exception as e:
        logger.error(f"Step 3 (parametric) failed: {e}", exc_info=True)
        save_step_status(run_dir, "step_3_param", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
