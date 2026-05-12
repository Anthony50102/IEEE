"""
Smoke test for parametric_data.py and parametric_solve.py.

Strategy: plant a known set of ground-truth blocks (A0, A1, F0, F1, c0, c1)
at small r, generate synthetic trajectories at 3 alpha values by rolling
out the ground-truth ROM, then call solve_parametric_opinf and check
that we recover the blocks to numerical precision (with tiny
regularization). Also verify:

- check_affine_feature_rank for s=3, q=2 reports full column rank.
- check_affine_feature_rank for s=3, q=3 (boundary) reports condition
  number > 1 but still full rank with non-collinear alphas.
- build_parametric_data_matrix returns the expected row/column shapes.
- assemble_operators_at_alpha matches solver output at trained alphas.
- A sweep with a "bad" lambda1=lambda2=0 vs sensible lambdas behaves.

Run:
    cd opinf/ && python test_parametric_smoke.py
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core import get_quadratic_terms, solve_difference_model  # noqa: E402
from parametric_data import (  # noqa: E402
    check_affine_feature_rank,
    parametric_data_matrix_row_widths,
    build_parametric_data_matrix,
    split_operator_block,
    assemble_operators_at_alpha,
)
from parametric_solve import (  # noqa: E402
    solve_parametric_opinf,
    rollout_at_alpha,
    evaluate_parametric_candidate,
    parametric_hyperparam_sweep,
)


def test_feature_rank():
    print("=== check_affine_feature_rank ===")
    rep = check_affine_feature_rank([0.1, 1.0, 5.0], q=2)
    print(f"  s=3, q=2: cond={rep['condition_number']:.3e}, "
          f"full_rank={rep['full_column_rank']}, sv={rep['singular_values']}")
    assert rep["full_column_rank"], "q=2 with 3 distinct alphas must be full rank"
    assert rep["condition_number"] < 1e3, "cond too high for safe regression"

    rep3 = check_affine_feature_rank([0.1, 1.0, 5.0], q=3)
    print(f"  s=3, q=3: cond={rep3['condition_number']:.3e}, "
          f"full_rank={rep3['full_column_rank']}")
    assert rep3["condition_number"] > rep["condition_number"], \
        "q=3 should be more ill-conditioned than q=2"

    rep_dup = check_affine_feature_rank([0.1, 0.1, 0.1], q=2)
    print(f"  all-equal alphas: full_rank={rep_dup['full_column_rank']}, "
          f"cond={rep_dup['condition_number']:.3e}")
    assert not rep_dup["full_column_rank"], "all-equal alphas must be rank-deficient"
    print("  PASS\n")


def test_layout():
    print("=== parametric_data_matrix_row_widths ===")
    for r in (3, 5, 10):
        L = parametric_data_matrix_row_widths(r)
        s = r * (r + 1) // 2
        expected = 2 * (r + s + 1)
        assert L["total"] == expected, f"r={r}: total {L['total']} != {expected}"
        assert L["s"] == s
        last = 0
        for name in ("A0", "A1", "F0", "F1", "c0", "c1"):
            lo, hi = L["offsets"][name]
            assert lo == last, f"gap before {name}"
            last = hi
        assert last == L["total"]
        print(f"  r={r}: total cols={L['total']}, s={s}  OK")
    print("  PASS\n")


def synth_ground_truth(r=4, seed=0):
    rng = np.random.default_rng(seed)
    s = r * (r + 1) // 2
    A0 = rng.standard_normal((r, r)) * 0.1
    A1 = rng.standard_normal((r, r)) * 0.02
    eigs = np.abs(np.linalg.eigvals(A0))
    if eigs.max() > 0:
        A0 *= 0.5 / eigs.max()
    # F has to be large enough that the quadratic feature columns
    # contribute non-trivially to the targets; otherwise the regression
    # has insufficient signal to identify F0 vs F1.
    F0 = rng.standard_normal((r, s)) * 0.05
    F1 = rng.standard_normal((r, s)) * 0.02
    c0 = rng.standard_normal(r) * 1e-2
    c1 = rng.standard_normal(r) * 5e-3
    return {"A0": A0, "A1": A1, "F0": F0, "F1": F1, "c0": c0, "c1": c1}


def rollout_ground_truth(blocks, alpha, x0, n_steps):
    A = blocks["A0"] + alpha * blocks["A1"]
    F = blocks["F0"] + alpha * blocks["F1"]
    c = blocks["c0"] + alpha * blocks["c1"]
    r = x0.size
    X = np.zeros((n_steps, r))
    X[0] = x0
    for k in range(n_steps - 1):
        X[k + 1] = A @ X[k] + F @ get_quadratic_terms(X[k]) + c
    return X


def test_regression_recovery():
    print("=== regression recovery on synthetic data ===")
    r = 4
    alphas_train = [0.1, 1.0, 5.0]
    K_per_alpha = 400
    blocks_true = synth_ground_truth(r=r, seed=42)
    rng = np.random.default_rng(123)

    Xhat = {}
    Yhat = {}
    for a in alphas_train:
        # Use random snapshots (not a rollout) so the predictor states
        # span r-dim space and the quadratic features carry real signal.
        # This still tests identifiability of the affine blocks.
        X = rng.standard_normal((K_per_alpha, r))
        A = blocks_true["A0"] + a * blocks_true["A1"]
        F = blocks_true["F0"] + a * blocks_true["F1"]
        c = blocks_true["c0"] + a * blocks_true["c1"]
        Y = np.zeros_like(X)
        for k in range(K_per_alpha):
            Y[k] = A @ X[k] + F @ get_quadratic_terms(X[k]) + c
        Xhat[a] = X
        Yhat[a] = Y
        assert np.isfinite(Y).all()

    bundle = build_parametric_data_matrix(Xhat, Yhat)
    D, Y, layout = bundle["D"], bundle["Y"], bundle["layout"]
    print(f"  D shape: {D.shape}, expected ({3 * K_per_alpha}, "
          f"{layout['total']})")
    assert D.shape == (3 * K_per_alpha, layout["total"])
    assert Y.shape == (3 * K_per_alpha, r)

    blocks_solved = solve_parametric_opinf(D, Y, layout, 1e-12, 1e-12)

    max_err = 0.0
    for name in ("A0", "A1", "F0", "F1", "c0", "c1"):
        err = np.max(np.abs(blocks_solved[name] - blocks_true[name]))
        rel = err / max(1e-30, np.max(np.abs(blocks_true[name])))
        print(f"  {name}: max_abs_err={err:.2e}, rel={rel:.2e}")
        max_err = max(max_err, rel)
    assert max_err < 1e-6, f"Recovery error too large: {max_err}"
    print("  PASS\n")
    return blocks_true, blocks_solved, Xhat, Yhat


def test_assemble_at_alpha(blocks_true, blocks_solved):
    print("=== assemble_operators_at_alpha consistency ===")
    for a in (0.1, 1.0, 1.5, 5.0):
        ops_true = {
            "A": blocks_true["A0"] + a * blocks_true["A1"],
            "F": blocks_true["F0"] + a * blocks_true["F1"],
            "c": blocks_true["c0"] + a * blocks_true["c1"],
        }
        ops_solved = assemble_operators_at_alpha(blocks_solved, a)
        for k in ("A", "F", "c"):
            err = np.max(np.abs(ops_true[k] - ops_solved[k]))
            assert err < 1e-6, f"alpha={a} {k} err {err}"
        print(f"  alpha={a}: consistent  OK")
    print("  PASS\n")


def test_rollout_match(blocks_true, blocks_solved, Xhat):
    print("=== rollout match at trained alphas + held-out alpha ===")
    for a in (0.1, 1.0, 5.0):
        x0 = Xhat[a][0]
        is_nan, X_solved = rollout_at_alpha(blocks_solved, a, x0, 50)
        assert not is_nan
        X_true = rollout_ground_truth(blocks_true, a, x0, 50).T
        err = np.max(np.abs(X_solved - X_true))
        print(f"  alpha={a} (trained): rollout max abs err = {err:.2e}")
        assert err < 1e-4

    a = 1.5
    x0 = np.random.default_rng(7).standard_normal(blocks_solved["A0"].shape[0]) * 0.3
    is_nan, X_solved = rollout_at_alpha(blocks_solved, a, x0, 50)
    X_true = rollout_ground_truth(blocks_true, a, x0, 50).T
    err = np.max(np.abs(X_solved - X_true))
    print(f"  alpha={a} (held-out): rollout max abs err = {err:.2e}")
    assert not is_nan
    assert err < 1e-4
    print("  PASS\n")


def test_sweep_picks_sensible(blocks_true, Xhat, Yhat):
    print("=== sweep ranking sanity ===")
    bundle = build_parametric_data_matrix(Xhat, Yhat)
    D, Y, layout = bundle["D"], bundle["Y"], bundle["layout"]
    r = next(iter(Xhat.values())).shape[1]

    # Build coherent rollout targets (the Yhat in the bundle is from
    # random snapshots, fine for identifiability but not for scoring
    # multi-step rollouts).
    n_steps = 60
    rng = np.random.default_rng(99)
    training_ics = {}
    training_targets = {}
    for a in Xhat:
        x0 = rng.standard_normal(r) * 0.3
        traj = rollout_ground_truth(blocks_true, a, x0, n_steps)
        training_ics[a] = x0
        training_targets[a] = traj  # (n_steps, r)

    sentinel_ic = rng.standard_normal(r) * 0.3

    results = parametric_hyperparam_sweep(
        D, Y, layout,
        lambda1_grid=[1e-12, 1e-2, 1e2, 1e6],
        lambda2_grid=[1e-12, 1e-2, 1e2, 1e6],
        training_ics=training_ics,
        training_targets=training_targets,
        n_steps=n_steps,
        sentinel_alpha=1.5,
        sentinel_ic=sentinel_ic,
        sentinel_steps=50,
        sentinel_max_norm=1e3,
    )
    best = results[0]
    print(f"  best: lambda1_sq={best['lambda1_sq']:.2e}, "
          f"lambda2_sq={best['lambda2_sq']:.2e}, score={best['score']:.3e}, "
          f"sentinel_ok={best['sentinel_bounded']}")
    worst_admitted = next((r for r in reversed(results)
                           if not r["is_disqualified"]), None)
    if worst_admitted is not None:
        print(f"  worst admitted: l1_sq={worst_admitted['lambda1_sq']:.2e}, "
              f"l2_sq={worst_admitted['lambda2_sq']:.2e}, "
              f"score={worst_admitted['score']:.3e}")
    assert best["lambda1_sq"] <= 1e-2 and best["lambda2_sq"] <= 1e-2, (
        "On clean data, sweep should pick small lambdas"
    )
    print("  PASS\n")


if __name__ == "__main__":
    test_feature_rank()
    test_layout()
    blocks_true, blocks_solved, Xhat, Yhat = test_regression_recovery()
    test_assemble_at_alpha(blocks_true, blocks_solved)
    test_rollout_match(blocks_true, blocks_solved, Xhat)
    test_sweep_picks_sensible(blocks_true, Xhat, Yhat)
    print("All parametric math smoke tests passed.")
