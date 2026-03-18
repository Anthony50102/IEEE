#!/usr/bin/env python3
"""
DMD hyperparameter sweep for KS equation.

Sweeps output-model regularization, POD rank, and BOPDMD bagging
to minimize test-set energy/enstrophy errors.
"""

import copy
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent
BASE_CONFIG = BASE_DIR / "dmd/config/dmd_ks_temporal_split_local.yaml"
SWEEP_DIR = BASE_DIR / "sweep_configs"
OUTPUT_BASE = BASE_DIR / "local_output"

STEPS = [
    "dmd/step_1_preprocess.py",
    "dmd/step_2_train.py",
    "dmd/step_3_evaluate.py",
]


def load_base_config():
    with open(BASE_CONFIG) as f:
        return yaml.safe_load(f)


def write_config(cfg, path):
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def make_config(r, alpha_lin, alpha_quad, num_trials):
    """Return a modified config dict."""
    cfg = load_base_config()
    cfg["reduction"]["r"] = r
    cfg["output_model"]["alpha_lin"] = float(alpha_lin)
    cfg["output_model"]["alpha_quad"] = float(alpha_quad)
    cfg["dmd"]["num_trials"] = num_trials
    # Disable plots and saving predictions to speed up sweep
    cfg["evaluation"]["generate_plots"] = False
    cfg["evaluation"]["save_predictions"] = False
    cfg["evaluation"]["plot_state_error"] = False
    cfg["evaluation"]["plot_state_snapshots"] = False
    cfg["execution"]["verbose"] = False
    return cfg


def find_latest_run_dir(before_ts=None):
    """Find the most recently created dmd_ks run directory."""
    pattern = str(OUTPUT_BASE / "*dmd_ks_temporal_split")
    dirs = sorted(glob.glob(pattern), reverse=True)
    for d in dirs:
        if before_ts is None or os.path.basename(d) > (before_ts or ""):
            return d
    return dirs[0] if dirs else None


def run_pipeline(config_path, label):
    """Run the 3-step DMD pipeline. Returns (run_dir, success)."""
    # Record dirs before step 1
    existing = set(glob.glob(str(OUTPUT_BASE / "*dmd_ks_temporal_split")))

    for i, step in enumerate(STEPS):
        cmd = [sys.executable, str(BASE_DIR / step), "--config", str(config_path)]
        if i > 0:
            cmd.extend(["--run-dir", run_dir])

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(BASE_DIR), timeout=120
        )

        if result.returncode != 0:
            print(f"  [FAIL] Step {i+1} failed for {label}")
            print(f"  stderr: {result.stderr[-500:]}")
            return None, False

        # After step 1, find the new run directory
        if i == 0:
            new_dirs = set(glob.glob(str(OUTPUT_BASE / "*dmd_ks_temporal_split"))) - existing
            if new_dirs:
                run_dir = sorted(new_dirs)[-1]
            else:
                # Fallback: find most recent
                all_dirs = sorted(glob.glob(str(OUTPUT_BASE / "*dmd_ks_temporal_split")))
                run_dir = all_dirs[-1] if all_dirs else None
            if not run_dir:
                print(f"  [FAIL] Could not find run directory for {label}")
                return None, False

    return run_dir, True


def read_metrics(run_dir):
    """Read evaluation metrics from a run directory."""
    metrics_path = os.path.join(run_dir, "dmd_evaluation_metrics.yaml")
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path) as f:
        return yaml.safe_load(f)


def extract_test_metrics(metrics):
    """Extract key test metrics as a dict."""
    if metrics is None:
        return None
    s = metrics.get("test", {}).get("summary", {})
    return {
        "energy_mean": s.get("mean_err_Gamma_n", float("nan")),
        "energy_std": s.get("std_err_Gamma_n", float("nan")),
        "enstrophy_mean": s.get("mean_err_Gamma_c", float("nan")),
        "enstrophy_std": s.get("std_err_Gamma_c", float("nan")),
    }


def print_table(results):
    """Print a formatted results table."""
    header = (
        f"{'Phase':<8} {'r':>3} {'α_lin':>8} {'α_quad':>8} {'trials':>6} │ "
        f"{'E_mean%':>8} {'E_std%':>8} {'C_mean%':>8} {'C_std%':>8} │ {'Status':<6}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    best_score = float("inf")
    best_idx = -1

    for i, res in enumerate(results):
        m = res.get("metrics")
        if m:
            score = m["energy_mean"] + m["enstrophy_mean"]
            if score < best_score:
                best_score = score
                best_idx = i

    for i, res in enumerate(results):
        p = res["params"]
        m = res.get("metrics")
        marker = " ★" if i == best_idx else ""
        if m:
            line = (
                f"{res['phase']:<8} {p['r']:>3} {p['alpha_lin']:>8.2g} "
                f"{p['alpha_quad']:>8.2g} {p['num_trials']:>6} │ "
                f"{m['energy_mean']*100:>7.2f}% {m['energy_std']*100:>7.2f}% "
                f"{m['enstrophy_mean']*100:>7.2f}% {m['enstrophy_std']*100:>7.2f}% │ "
                f"{'OK':<6}{marker}"
            )
        else:
            line = (
                f"{res['phase']:<8} {p['r']:>3} {p['alpha_lin']:>8.2g} "
                f"{p['alpha_quad']:>8.2g} {p['num_trials']:>6} │ "
                f"{'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} │ {'FAIL':<6}"
            )
        print(line)
    print(sep)

    if best_idx >= 0:
        b = results[best_idx]
        bm = b["metrics"]
        bp = b["params"]
        print(f"\n★ Best config: r={bp['r']}, α_lin={bp['alpha_lin']:.2g}, "
              f"α_quad={bp['alpha_quad']:.2g}, trials={bp['num_trials']}")
        print(f"  Energy  err: {bm['energy_mean']*100:.2f}% mean, "
              f"{bm['energy_std']*100:.2f}% std")
        print(f"  Enstrophy err: {bm['enstrophy_mean']*100:.2f}% mean, "
              f"{bm['enstrophy_std']*100:.2f}% std")
        print(f"  Run dir: {b.get('run_dir', 'N/A')}")

    return best_idx


def run_sweep_item(phase, params, label):
    """Run a single sweep configuration."""
    cfg = make_config(**params)
    config_path = SWEEP_DIR / f"sweep_{label}.yaml"
    write_config(cfg, config_path)

    print(f"\n[{label}] r={params['r']}, α_lin={params['alpha_lin']:.2g}, "
          f"α_quad={params['alpha_quad']:.2g}, trials={params['num_trials']}")

    t0 = time.time()
    run_dir, success = run_pipeline(config_path, label)
    elapsed = time.time() - t0

    result = {"phase": phase, "params": params, "label": label, "run_dir": run_dir}

    if success:
        raw_metrics = read_metrics(run_dir)
        result["metrics"] = extract_test_metrics(raw_metrics)
        m = result["metrics"]
        if m:
            print(f"  Energy: {m['energy_mean']*100:.2f}%, "
                  f"Enstrophy: {m['enstrophy_mean']*100:.2f}% "
                  f"({elapsed:.1f}s)")
        else:
            print(f"  No metrics found ({elapsed:.1f}s)")
    else:
        result["metrics"] = None
        print(f"  FAILED ({elapsed:.1f}s)")

    return result


def main():
    SWEEP_DIR.mkdir(exist_ok=True)

    all_results = []

    # ── Phase 1: Output regularization sweep (r=20) ──
    print("=" * 70)
    print("PHASE 1: Output regularization sweep (r=20, num_trials=0)")
    print("=" * 70)

    alpha_lins = [0.01, 0.1, 1.0, 10.0]
    alpha_quads = [1.0, 10.0, 100.0, 1000.0]

    for al in alpha_lins:
        for aq in alpha_quads:
            params = {"r": 20, "alpha_lin": al, "alpha_quad": aq, "num_trials": 0}
            label = f"p1_al{al}_aq{aq}"
            result = run_sweep_item("Phase1", params, label)
            all_results.append(result)

    # Find best from Phase 1
    phase1_scores = []
    for i, r in enumerate(all_results):
        if r["metrics"]:
            score = r["metrics"]["energy_mean"] + r["metrics"]["enstrophy_mean"]
            phase1_scores.append((score, i))

    phase1_scores.sort()
    if phase1_scores:
        best_p1_idx = phase1_scores[0][1]
        best_p1 = all_results[best_p1_idx]["params"]
        best_al = best_p1["alpha_lin"]
        best_aq = best_p1["alpha_quad"]
        print(f"\nPhase 1 best: α_lin={best_al:.2g}, α_quad={best_aq:.2g}")
    else:
        print("Phase 1 produced no valid results!")
        best_al, best_aq = 0.01, 1.0

    # ── Phase 2: Rank sweep ──
    print("\n" + "=" * 70)
    print(f"PHASE 2: Rank sweep (α_lin={best_al:.2g}, α_quad={best_aq:.2g})")
    print("=" * 70)

    for r in [10, 15, 30]:
        params = {"r": r, "alpha_lin": best_al, "alpha_quad": best_aq, "num_trials": 0}
        label = f"p2_r{r}"
        result = run_sweep_item("Phase2", params, label)
        all_results.append(result)

    # Find overall best so far
    all_scores = []
    for i, r in enumerate(all_results):
        if r["metrics"]:
            score = r["metrics"]["energy_mean"] + r["metrics"]["enstrophy_mean"]
            all_scores.append((score, i))

    all_scores.sort()
    if all_scores:
        best_idx = all_scores[0][1]
        best_params = all_results[best_idx]["params"]
        best_r = best_params["r"]
        best_al = best_params["alpha_lin"]
        best_aq = best_params["alpha_quad"]
        print(f"\nBest so far: r={best_r}, α_lin={best_al:.2g}, α_quad={best_aq:.2g}")

    # ── Phase 3: Bagging ──
    print("\n" + "=" * 70)
    print(f"PHASE 3: BOPDMD bagging (r={best_r}, α_lin={best_al:.2g}, α_quad={best_aq:.2g})")
    print("=" * 70)

    params = {"r": best_r, "alpha_lin": best_al, "alpha_quad": best_aq, "num_trials": 10}
    label = "p3_bagging"
    result = run_sweep_item("Phase3", params, label)
    all_results.append(result)

    # ── Final Summary ──
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    best_idx = print_table(all_results)

    # Cleanup sweep configs
    # (keep them for reference)

    return all_results, best_idx


if __name__ == "__main__":
    all_results, best_idx = main()
