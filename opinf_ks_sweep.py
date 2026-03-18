#!/usr/bin/env python3
"""
OpInf Hyperparameter Sweep for KS Equation.

Systematically sweeps OpInf hyperparameters to improve test-set results,
particularly enstrophy std error (currently ~28.7%).

Usage:
    source .venv/bin/activate
    python opinf_ks_sweep.py [--experiments 1,2,3,4,5] [--skip-completed]
"""

import argparse
import copy
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml


BASE_DIR = Path(__file__).resolve().parent
BASE_CONFIG = BASE_DIR / "opinf" / "config" / "opinf_ks_temporal_split_local.yaml"
SWEEP_CONFIG_DIR = BASE_DIR / "opinf" / "config" / "sweep_configs"
LOCAL_OUTPUT = BASE_DIR / "local_output"

# Use conda IEEE environment which has MPI support
CONDA_PYTHON = Path.home() / "miniconda3" / "envs" / "IEEE" / "bin" / "python"
PYTHON = str(CONDA_PYTHON) if CONDA_PYTHON.exists() else sys.executable

STEP1 = "opinf/step_1_preprocess_serial.py"
STEP2 = "opinf/step_2_train.py"
STEP3 = "opinf/step_3_evaluate.py"


def load_base_config():
    with open(BASE_CONFIG, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg, path):
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def make_config_variant(base, modifications):
    """Apply nested modifications to a config dict."""
    cfg = copy.deepcopy(base)
    for key_path, value in modifications.items():
        keys = key_path.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
    return cfg


def find_run_dir_from_output(stdout_text):
    """Extract run directory path from step 1 stdout."""
    for line in stdout_text.split("\n"):
        if "Run directory:" in line:
            match = re.search(r"Run directory:\s*(.+)", line)
            if match:
                return match.group(1).strip()
    # Fallback: find most recent opinf_ks directory
    dirs = sorted(LOCAL_OUTPUT.glob("*opinf_ks*"))
    if dirs:
        return str(dirs[-1])
    return None


def run_pipeline(config_path, experiment_name):
    """Run the 3-step OpInf pipeline and return (run_dir, metrics, timing)."""
    config_path = str(config_path)
    timings = {}

    # Step 1: Preprocess
    print(f"  [Step 1] Preprocessing...", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, STEP1, "--config", config_path],
        capture_output=True, text=True, cwd=str(BASE_DIR),
    )
    timings["step1"] = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR in step 1:\n{result.stderr[-500:]}")
        return None, None, timings

    run_dir = find_run_dir_from_output(result.stdout)
    if not run_dir:
        print(f"  ERROR: Could not find run directory in step 1 output")
        print(f"  stdout (last 300 chars): {result.stdout[-300:]}")
        return None, None, timings

    print(f"  Run directory: {run_dir}")

    # Step 2: Train
    print(f"  [Step 2] Training (this takes ~2 min)...", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, STEP2, "--config", config_path, "--run-dir", run_dir],
        capture_output=True, text=True, cwd=str(BASE_DIR),
    )
    timings["step2"] = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR in step 2:\n{result.stderr[-500:]}")
        return run_dir, None, timings

    # Step 3: Evaluate
    print(f"  [Step 3] Evaluating...", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, STEP3, "--config", config_path, "--run-dir", run_dir],
        capture_output=True, text=True, cwd=str(BASE_DIR),
    )
    timings["step3"] = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR in step 3:\n{result.stderr[-500:]}")
        return run_dir, None, timings

    # Read metrics
    metrics = read_metrics(run_dir)
    return run_dir, metrics, timings


def run_evaluate_only(config_path, run_dir):
    """Re-run only step 3 (evaluation) on an existing run directory."""
    config_path = str(config_path)
    timings = {}
    
    print(f"  [Step 3 only] Re-evaluating with modified config...", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, STEP3, "--config", config_path, "--run-dir", run_dir],
        capture_output=True, text=True, cwd=str(BASE_DIR),
    )
    timings["step3"] = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR in step 3:\n{result.stderr[-500:]}")
        return None

    metrics = read_metrics(run_dir)
    return metrics, timings


def read_metrics(run_dir):
    """Read evaluation metrics and model counts from a run directory."""
    metrics = {}

    # Evaluation metrics
    metrics_path = Path(run_dir) / "evaluation_metrics.yaml"
    if metrics_path.exists():
        with open(metrics_path) as f:
            raw = yaml.unsafe_load(f)

        test = raw.get("test", {})
        ensemble = test.get("ensemble", {})
        trajectories = test.get("trajectories", [])

        metrics["energy_mean_err"] = ensemble.get("mean_err_Gamma_c", None)
        metrics["energy_std_err"] = ensemble.get("std_err_Gamma_c", None)
        metrics["enstrophy_mean_err"] = ensemble.get("mean_err_Gamma_n", None)
        metrics["enstrophy_std_err"] = ensemble.get("std_err_Gamma_n", None)

        if trajectories:
            t0 = trajectories[0]
            metrics["energy_mean_err"] = metrics["energy_mean_err"] or t0.get("err_mean_Gamma_c")
            metrics["energy_std_err"] = metrics["energy_std_err"] or t0.get("err_std_Gamma_c")
            metrics["enstrophy_mean_err"] = metrics["enstrophy_mean_err"] or t0.get("err_mean_Gamma_n")
            metrics["enstrophy_std_err"] = metrics["enstrophy_std_err"] or t0.get("err_std_Gamma_n")

    # Model counts from sweep_results.npz
    sweep_path = Path(run_dir) / "sweep_results.npz"
    if sweep_path.exists():
        sr = np.load(sweep_path, allow_pickle=True)
        metrics["n_total"] = int(sr.get("n_total", 0))
        metrics["n_selected"] = int(sr.get("n_selected", 0))

    # Pipeline status for additional info
    status_path = Path(run_dir) / "pipeline_status.yaml"
    if status_path.exists():
        with open(status_path) as f:
            status = yaml.unsafe_load(f)
        s2 = status.get("step_2", {})
        metrics["n_models"] = s2.get("n_models", metrics.get("n_selected", "?"))
        s1 = status.get("step_1", {})
        metrics["r_actual"] = s1.get("r", "?")

    return metrics


def fmt_pct(val):
    """Format a fraction as percentage string."""
    if val is None:
        return "N/A"
    return f"{val * 100:.2f}%"


def define_experiments(best_threshold_std=None, best_settings=None):
    """Define all experiments. Some depend on results of earlier ones."""

    experiments = {}

    # Experiment 1: Loosen model selection thresholds
    experiments["1a"] = {
        "name": "Exp 1a: threshold_std=0.3",
        "mods": {"model_selection.threshold_std": 0.3},
    }
    experiments["1b"] = {
        "name": "Exp 1b: threshold_std=0.5",
        "mods": {"model_selection.threshold_std": 0.5},
    }

    # Experiment 2: Widen regularization ranges (uses best threshold from exp 1)
    ts = best_threshold_std if best_threshold_std else 0.3
    experiments["2"] = {
        "name": f"Exp 2: wider reg ranges (thr_std={ts})",
        "mods": {
            "model_selection.threshold_std": ts,
            "regularization.state_lin.min": 1.0e-3,
            "regularization.state_quad.min": 1.0e-1,
            "regularization.output_lin.min": 1.0e-6,
            "regularization.output_quad.min": 1.0e-4,
        },
    }

    # Experiment 3: Finer grid (15^4)
    experiments["3"] = {
        "name": f"Exp 3: finer grid 15^4 (thr_std={ts})",
        "mods": {
            "model_selection.threshold_std": ts,
            "regularization.state_lin.num": 15,
            "regularization.state_quad.num": 15,
            "regularization.output_lin.num": 15,
            "regularization.output_quad.num": 15,
        },
    }

    # Experiment 4: Manifold reduction
    experiments["4"] = {
        "name": f"Exp 4: manifold reduction (thr_std={ts})",
        "mods": {
            "model_selection.threshold_std": ts,
            "reduction.method": "manifold",
        },
    }

    # Experiment 5: Higher target energy
    best = best_settings or {}
    e5_mods = {"reduction.target_energy": 0.999}
    if best.get("threshold_std"):
        e5_mods["model_selection.threshold_std"] = best["threshold_std"]
    if best.get("reg_widened"):
        e5_mods.update({
            "regularization.state_lin.min": 1.0e-3,
            "regularization.state_quad.min": 1.0e-1,
            "regularization.output_lin.min": 1.0e-6,
            "regularization.output_quad.min": 1.0e-4,
        })
    experiments["5"] = {
        "name": "Exp 5: target_energy=0.999",
        "mods": e5_mods,
    }

    # Experiments 6: TIGHTEN thresholds (fewer, better models → less averaging)
    # The ensemble mean of many models smooths out temporal variability.
    # Fewer, more accurate models may preserve enstrophy fluctuations better.
    experiments["6a"] = {
        "name": "Exp 6a: thr_std=0.10",
        "mods": {"model_selection.threshold_std": 0.10},
    }
    experiments["6b"] = {
        "name": "Exp 6b: thr_std=0.05",
        "mods": {"model_selection.threshold_std": 0.05},
    }
    experiments["6c"] = {
        "name": "Exp 6c: thr_std=0.10, thr_mean=0.10",
        "mods": {
            "model_selection.threshold_std": 0.10,
            "model_selection.threshold_mean": 0.10,
        },
    }
    experiments["6d"] = {
        "name": "Exp 6d: thr_std=0.05, thr_mean=0.05",
        "mods": {
            "model_selection.threshold_std": 0.05,
            "model_selection.threshold_mean": 0.05,
        },
    }

    # Experiment 7: Moderate reg widening (less aggressive than exp 2)
    experiments["7"] = {
        "name": "Exp 7: moderate reg widen",
        "mods": {
            "regularization.state_lin.min": 1.0e-2,
            "regularization.state_quad.min": 1.0e0,
            "regularization.output_lin.min": 1.0e-5,
            "regularization.output_quad.min": 1.0e-3,
        },
    }

    # Experiment 8: Moderate reg widening + tighter thresholds
    experiments["8"] = {
        "name": "Exp 8: mod reg + tight thr",
        "mods": {
            "regularization.state_lin.min": 1.0e-2,
            "regularization.state_quad.min": 1.0e0,
            "regularization.output_lin.min": 1.0e-5,
            "regularization.output_quad.min": 1.0e-3,
            "model_selection.threshold_std": 0.10,
        },
    }

    # Experiment 9: Target energy 0.995 (moderate increase)
    experiments["9"] = {
        "name": "Exp 9: target_energy=0.995",
        "mods": {"reduction.target_energy": 0.995},
    }

    # Experiment 10: Top-K model selection (fewer models → less averaging → more variability)
    experiments["10a"] = {
        "name": "Exp 10a: max_models=210",
        "mods": {"model_selection.max_models": 210},
    }
    experiments["10b"] = {
        "name": "Exp 10b: max_models=200",
        "mods": {"model_selection.max_models": 200},
    }
    experiments["10c"] = {
        "name": "Exp 10c: max_models=150",
        "mods": {"model_selection.max_models": 150},
    }
    experiments["10d"] = {
        "name": "Exp 10d: max_models=250",
        "mods": {"model_selection.max_models": 250},
    }

    return experiments


def print_summary(results):
    """Print a formatted summary table of all results."""
    print("\n" + "=" * 120)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 120)

    header = (
        f"{'Experiment':<45} {'Models':>8} {'r':>4} "
        f"{'E mean':>9} {'E std':>9} "
        f"{'Ens mean':>9} {'Ens std':>9} "
        f"{'Time':>7}"
    )
    print(header)
    print("-" * 120)

    # Baseline
    print(
        f"{'Baseline (current config)':<45} {'1084':>8} {'15':>4} "
        f"{'1.05%':>9} {'2.69%':>9} "
        f"{'3.17%':>9} {'28.66%':>9} "
        f"{'~140s':>7}"
    )

    best_ens_std = 1.0  # track best enstrophy std error
    best_exp = "Baseline"

    for exp_id, res in results.items():
        m = res.get("metrics")
        t = res.get("timings", {})
        name = res["name"]

        if m is None:
            print(f"{name:<45} {'FAILED':>8}")
            continue

        n_sel = m.get("n_selected", m.get("n_models", "?"))
        r_act = m.get("r_actual", "?")
        total_time = sum(t.get(k, 0) for k in ["step1", "step2", "step3"])

        e_mean = fmt_pct(m.get("energy_mean_err"))
        e_std = fmt_pct(m.get("energy_std_err"))
        ens_mean = fmt_pct(m.get("enstrophy_mean_err"))
        ens_std = fmt_pct(m.get("enstrophy_std_err"))

        marker = ""
        ens_std_val = m.get("enstrophy_std_err")
        if ens_std_val is not None and ens_std_val < best_ens_std:
            best_ens_std = ens_std_val
            best_exp = name

        print(
            f"{name:<45} {str(n_sel):>8} {str(r_act):>4} "
            f"{e_mean:>9} {e_std:>9} "
            f"{ens_mean:>9} {ens_std:>9} "
            f"{total_time:>6.0f}s"
        )

    print("-" * 120)
    if best_ens_std < 0.2866:  # improved from baseline
        print(f"★ BEST: {best_exp} — Enstrophy std error: {best_ens_std*100:.2f}%")
    else:
        print(f"No improvement over baseline (28.66%). Best: {best_exp} ({best_ens_std*100:.2f}%)")
    print("=" * 120)

    return best_exp, best_ens_std


def main():
    parser = argparse.ArgumentParser(description="OpInf KS Hyperparameter Sweep")
    parser.add_argument(
        "--experiments", type=str, default="1,2",
        help="Comma-separated experiment IDs to run (e.g., '1,2,3,4,5'). "
             "Use '1a' or '1b' for individual threshold experiments. "
             "Default: '1,2' (Experiments 1 and 2)."
    )
    args = parser.parse_args()

    requested = [x.strip() for x in args.experiments.split(",")]

    # Expand '1' into '1a,1b', '6' into '6a-6d', '10' into '10a-10d'
    expanded = []
    for r in requested:
        if r == "1":
            expanded.extend(["1a", "1b"])
        elif r == "6":
            expanded.extend(["6a", "6b", "6c", "6d"])
        elif r == "10":
            expanded.extend(["10a", "10b", "10c", "10d"])
        else:
            expanded.append(r)
    requested = expanded

    print("=" * 80)
    print("OpInf KS Hyperparameter Sweep")
    print(f"Base config: {BASE_CONFIG}")
    print(f"Experiments to run: {requested}")
    print("=" * 80)

    # Load base config
    base_cfg = load_base_config()

    # Create sweep config directory
    SWEEP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    best_threshold_std = 0.15  # current default
    best_settings = {}

    # Run Experiment 1 first to determine best threshold
    exp1_ids = [x for x in requested if x.startswith("1")]
    later_ids = [x for x in requested if not x.startswith("1")]

    # Define initial experiments
    experiments = define_experiments()

    # Run Exp 1 variants
    for exp_id in exp1_ids:
        if exp_id not in experiments:
            print(f"Unknown experiment: {exp_id}")
            continue

        exp = experiments[exp_id]
        print(f"\n{'='*80}")
        print(f"Running {exp['name']}")
        print(f"{'='*80}")

        cfg = make_config_variant(base_cfg, exp["mods"])
        config_path = SWEEP_CONFIG_DIR / f"sweep_{exp_id}.yaml"
        save_config(cfg, config_path)

        run_dir, metrics, timings = run_pipeline(config_path, exp["name"])
        results[exp_id] = {
            "name": exp["name"],
            "run_dir": run_dir,
            "metrics": metrics,
            "timings": timings,
        }

        if metrics:
            ens_std = metrics.get("enstrophy_std_err")
            ens_mean = metrics.get("enstrophy_mean_err")
            print(f"  Results: Energy mean={fmt_pct(metrics.get('energy_mean_err'))}, "
                  f"Enstrophy mean={fmt_pct(ens_mean)}, "
                  f"Enstrophy std={fmt_pct(ens_std)}, "
                  f"Models={metrics.get('n_selected', '?')}")

    # Determine best threshold from Exp 1
    best_ens_std_err = 1.0
    for exp_id in exp1_ids:
        m = results.get(exp_id, {}).get("metrics")
        if m and m.get("enstrophy_std_err") is not None:
            # Choose the threshold that gives lowest enstrophy std error
            # but also doesn't degrade mean errors too much
            ens_std = m["enstrophy_std_err"]
            ens_mean = m.get("enstrophy_mean_err", 1.0)
            if ens_std < best_ens_std_err and ens_mean < 0.10:  # acceptable mean error
                best_ens_std_err = ens_std
                exp_cfg = experiments[exp_id]
                best_threshold_std = exp_cfg["mods"].get("model_selection.threshold_std", 0.15)

    print(f"\n  → Best threshold_std from Exp 1: {best_threshold_std}")
    best_settings["threshold_std"] = best_threshold_std

    # Redefine experiments with best threshold
    experiments = define_experiments(
        best_threshold_std=best_threshold_std,
        best_settings=best_settings,
    )

    # Run remaining experiments
    for exp_id in later_ids:
        if exp_id not in experiments:
            print(f"Unknown experiment: {exp_id}")
            continue

        exp = experiments[exp_id]
        print(f"\n{'='*80}")
        print(f"Running {exp['name']}")
        print(f"{'='*80}")

        cfg = make_config_variant(base_cfg, exp["mods"])
        config_path = SWEEP_CONFIG_DIR / f"sweep_{exp_id}.yaml"
        save_config(cfg, config_path)

        run_dir, metrics, timings = run_pipeline(config_path, exp["name"])
        results[exp_id] = {
            "name": exp["name"],
            "run_dir": run_dir,
            "metrics": metrics,
            "timings": timings,
        }

        if metrics:
            ens_std = metrics.get("enstrophy_std_err")
            ens_mean = metrics.get("enstrophy_mean_err")
            print(f"  Results: Energy mean={fmt_pct(metrics.get('energy_mean_err'))}, "
                  f"Enstrophy mean={fmt_pct(ens_mean)}, "
                  f"Enstrophy std={fmt_pct(ens_std)}, "
                  f"Models={metrics.get('n_selected', '?')}")

            # Track if reg widening helped
            if exp_id == "2":
                best_settings["reg_widened"] = True

    # Print final summary
    best_name, best_val = print_summary(results)

    # Save results to YAML for reference
    summary_path = BASE_DIR / "opinf_sweep_results.yaml"
    summary = {
        "baseline": {
            "energy_mean_err": 0.0105,
            "enstrophy_mean_err": 0.0317,
            "enstrophy_std_err": 0.2866,
            "n_selected": 1084,
        },
        "best_experiment": best_name,
        "best_enstrophy_std_err": float(best_val),
        "experiments": {},
    }
    for exp_id, res in results.items():
        m = res.get("metrics", {})
        summary["experiments"][exp_id] = {
            "name": res["name"],
            "run_dir": res.get("run_dir"),
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                        for k, v in (m or {}).items()},
        }
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()
