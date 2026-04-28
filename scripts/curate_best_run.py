#!/usr/bin/env python3
"""Curate a best run into the results/ directory for version control.

Copies the essential artifacts (configs, metrics, figures, and small model/
prediction files) from a completed pipeline run directory into
``results/<method>/<pde>/``, stripping large intermediates that don't belong
in Git.

Author: Anthony Poole
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# ── Size thresholds ──────────────────────────────────────────────────────────
MAX_PREDICTION_SIZE_MB = 50  # skip prediction files larger than this
MAX_MODEL_SIZE_MB = 25       # skip model files larger than this (unless forced)

# ── Per-method file rules ────────────────────────────────────────────────────
# "configs"   — YAML config files
# "metrics"   — evaluation metric YAML files
# "status"    — pipeline_status.yaml
# "figures"   — entire figures/ tree (PNGs)
# "model"     — trained model artifacts (method-specific)
# "predictions" — prediction .npz files

METHOD_ARTIFACTS = {
    "dmd": {
        "model": [
            "dmd_model.npz",
            "learning_matrices.npz",
        ],
        "predictions": [
            "dmd_predictions.npz",
        ],
    },
    "opinf": {
        "model": [
            "ensemble_models.npz",
        ],
        "predictions": [],  # ensemble_predictions.npz is ~100 MB — too large
    },
    "fno": {
        "model": [
            "checkpoint_best.pt",
            "checkpoint_rollout_final.pt",
        ],
        "predictions": [
            "predictions.npz",
            "state_predictions.npz",
        ],
    },
}


# =============================================================================
# Detection helpers
# =============================================================================

def detect_method(run_dir: Path) -> str:
    """Detect the method from the run directory name or contents.

    Parameters
    ----------
    run_dir : Path
        Path to the run directory.

    Returns
    -------
    str
        One of "dmd", "opinf", "fno".

    Raises
    ------
    SystemExit
        If the method cannot be determined.
    """
    name = run_dir.name.lower()
    for method in ("dmd", "opinf", "fno"):
        if method in name:
            return method

    # Fallback: check for method-specific files
    if (run_dir / "dmd_model.npz").exists():
        return "dmd"
    if (run_dir / "operators").is_dir() or (run_dir / "ensemble_models.npz").exists():
        return "opinf"
    if (run_dir / "checkpoint_best.pt").exists():
        return "fno"

    print(f"ERROR: Cannot detect method from '{run_dir.name}'.")
    print("       Use --method to specify explicitly.")
    sys.exit(1)


def detect_pde(run_dir: Path) -> str:
    """Detect the PDE benchmark from the run directory name or config.

    Parameters
    ----------
    run_dir : Path
        Path to the run directory.

    Returns
    -------
    str
        One of "ks", "hw2d".

    Raises
    ------
    SystemExit
        If the PDE cannot be determined.
    """
    name = run_dir.name.lower()
    if "hw2d" in name or "hasegawa" in name:
        return "hw2d"
    if "ks" in name or "kuramoto" in name:
        return "ks"

    # Fallback: peek into config files
    for cfg in run_dir.glob("config*.yaml"):
        text = cfg.read_text()
        if "hw2d" in text or "hasegawa" in text:
            return "hw2d"
        if "ks" in text or "kuramoto" in text:
            return "ks"

    print(f"ERROR: Cannot detect PDE from '{run_dir.name}'.")
    print("       Use --pde to specify explicitly.")
    sys.exit(1)


# =============================================================================
# File collection
# =============================================================================

def _file_size_mb(path: Path) -> float:
    """Return file size in megabytes."""
    return path.stat().st_size / (1024 * 1024)


def collect_files(run_dir: Path, method: str, include_model: bool):
    """Build a list of (src_path, relative_dest_path) tuples to copy.

    Parameters
    ----------
    run_dir : Path
        Source run directory.
    method : str
        One of "dmd", "opinf", "fno".
    include_model : bool
        If True, include model files even if they exceed the size threshold.

    Returns
    -------
    list[tuple[Path, Path]]
        List of (source, relative_destination) pairs.
    list[tuple[Path, str]]
        List of (source, reason) pairs for skipped files.
    """
    files_to_copy = []
    skipped = []
    artifacts = METHOD_ARTIFACTS[method]

    # ── Configs ──────────────────────────────────────────────────────────
    for cfg in sorted(run_dir.glob("config*.yaml")):
        files_to_copy.append((cfg, Path(cfg.name)))

    # ── Pipeline status ──────────────────────────────────────────────────
    status = run_dir / "pipeline_status.yaml"
    if status.exists():
        files_to_copy.append((status, Path(status.name)))

    # FNO also has status.txt
    status_txt = run_dir / "status.txt"
    if status_txt.exists():
        files_to_copy.append((status_txt, Path(status_txt.name)))

    # ── Metrics ──────────────────────────────────────────────────────────
    for pattern in ["*metrics*.yaml", "*evaluation*.yaml"]:
        for f in sorted(run_dir.glob(pattern)):
            if f.name not in [p.name for _, p in files_to_copy]:
                files_to_copy.append((f, Path(f.name)))

    # ── Figures ──────────────────────────────────────────────────────────
    figures_dir = run_dir / "figures"
    if figures_dir.is_dir():
        for png in sorted(figures_dir.rglob("*.png")):
            rel = Path("figures") / png.relative_to(figures_dir)
            files_to_copy.append((png, rel))

    # Also grab any top-level PNGs (e.g., pod_energy.png)
    for png in sorted(run_dir.glob("*.png")):
        files_to_copy.append((png, Path("figures") / png.name))

    # ── Model files ──────────────────────────────────────────────────────
    for model_file in artifacts.get("model", []):
        src = run_dir / model_file
        if not src.exists():
            continue
        size = _file_size_mb(src)
        if size > MAX_MODEL_SIZE_MB and not include_model:
            skipped.append((src, f"model file {size:.1f} MB > {MAX_MODEL_SIZE_MB} MB "
                                 f"(use --include-model to force)"))
            continue
        files_to_copy.append((src, Path("model") / src.name))

    # ── Predictions ──────────────────────────────────────────────────────
    for pred_file in artifacts.get("predictions", []):
        src = run_dir / pred_file
        if not src.exists():
            continue
        size = _file_size_mb(src)
        if size > MAX_PREDICTION_SIZE_MB:
            skipped.append((src, f"prediction file {size:.1f} MB > "
                                 f"{MAX_PREDICTION_SIZE_MB} MB"))
            continue
        files_to_copy.append((src, Path("predictions") / src.name))

    return files_to_copy, skipped


# =============================================================================
# Copy logic
# =============================================================================

def curate(run_dir: Path, method: str, pde: str,
           include_model: bool, dry_run: bool):
    """Copy curated artifacts from a run directory into results/.

    Parameters
    ----------
    run_dir : Path
        Source run directory (e.g., local_output/20260316_...).
    method : str
        Method name ("dmd", "opinf", "fno").
    pde : str
        PDE name ("ks", "hw2d").
    include_model : bool
        Force inclusion of large model files.
    dry_run : bool
        If True, print what would be copied without actually copying.
    """
    dest_dir = RESULTS_DIR / method / pde
    files_to_copy, skipped = collect_files(run_dir, method, include_model)

    if not files_to_copy:
        print("ERROR: No files found to curate. Is the run directory valid?")
        sys.exit(1)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"  Method:      {method}")
    print(f"  PDE:         {pde}")
    print(f"  Source:      {run_dir}")
    print(f"  Destination: {dest_dir}")
    print(f"  Files:       {len(files_to_copy)}")
    total_size = sum(_file_size_mb(src) for src, _ in files_to_copy)
    print(f"  Total size:  {total_size:.1f} MB")
    print()

    if skipped:
        print("  Skipped (too large):")
        for src, reason in skipped:
            print(f"    {src.name}: {reason}")
        print()

    if dry_run:
        print("  Would copy:")
        for src, rel_dest in files_to_copy:
            print(f"    {src.name}  →  {dest_dir / rel_dest}")
        print("\n  (dry run — nothing was copied)")
        return

    # ── Check for existing results ───────────────────────────────────────
    if dest_dir.exists() and any(dest_dir.iterdir()):
        existing_files = list(dest_dir.rglob("*"))
        existing_count = sum(1 for f in existing_files if f.is_file())
        if existing_count > 0:
            print(f"  WARNING: {dest_dir} already has {existing_count} files.")
            response = input("  Overwrite? [y/N] ").strip().lower()
            if response != "y":
                print("  Aborted.")
                return
            # Clean out old results
            for item in dest_dir.iterdir():
                if item.name == ".gitkeep":
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

    # ── Copy files ───────────────────────────────────────────────────────
    for src, rel_dest in files_to_copy:
        dest = dest_dir / rel_dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    # ── Write provenance metadata ────────────────────────────────────────
    provenance = dest_dir / "provenance.yaml"
    provenance.write_text(
        f"source_run: {run_dir.name}\n"
        f"source_path: {run_dir}\n"
        f"method: {method}\n"
        f"pde: {pde}\n"
        f"files_copied: {len(files_to_copy)}\n"
        f"total_size_mb: {total_size:.1f}\n"
    )

    print(f"  ✓ Curated {len(files_to_copy)} files ({total_size:.1f} MB) "
          f"into {dest_dir}")
    print(f"  ✓ Provenance recorded in {provenance.name}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Curate a best pipeline run into results/ for version control.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s local_output/20260316_090304_dmd_ks_temporal_split
  %(prog)s local_output/fno_ks_... --include-model
  %(prog)s /path/to/run --method opinf --pde hw2d
  %(prog)s local_output/some_run --dry-run
        """,
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the completed pipeline run directory.",
    )
    parser.add_argument(
        "--method",
        choices=["dmd", "opinf", "fno"],
        help="Override auto-detected method.",
    )
    parser.add_argument(
        "--pde",
        choices=["ks", "hw2d"],
        help="Override auto-detected PDE.",
    )
    parser.add_argument(
        "--include-model",
        action="store_true",
        help="Include large model files (e.g., FNO checkpoint_best.pt).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without actually copying.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"ERROR: '{run_dir}' is not a directory.")
        sys.exit(1)

    method = args.method or detect_method(run_dir)
    pde = args.pde or detect_pde(run_dir)

    print(f"\n  Curating best run for {method.upper()} / {pde.upper()}")
    print(f"  {'─' * 50}")

    curate(run_dir, method, pde, args.include_model, args.dry_run)
    print()


if __name__ == "__main__":
    main()
