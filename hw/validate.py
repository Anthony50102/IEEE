"""Phase 1 sanity gate: do our DNS Gamma_n values match the hw2d reference?

Author: Anthony Poole

Reads the `data_card.yaml` produced for one or more DNS runs and compares
the steady-state mean Gamma_n to the hw2d README reference table
(`hw/reference_gammas.yaml`). Reports per-alpha:

  - reference mean +- std
  - our mean +- std
  - z-score abs(ours - ref) / ref_std
  - verdict: "ok" (z < 2), "warn" (2 <= z < 3), "fail" (z >= 3 or NaN)

This is the hard gate before Phase 2: if alpha=1.0 fails, the simulator
setup is wrong and no surrogate work proceeds.

Usage:
    python -m hw.validate <out_dir1>/data_card.yaml [<out_dir2>/data_card.yaml ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def load_reference(ref_path: str | Path) -> dict:
    with open(ref_path) as f:
        raw = yaml.safe_load(f)
    return raw["gamma_n_reference"]


def verdict(z: float) -> str:
    if z != z:  # NaN
        return "fail"
    if z >= 3.0:
        return "fail"
    if z >= 2.0:
        return "warn"
    return "ok"


def compare_one(card_path: Path, reference: dict) -> dict:
    with open(card_path) as f:
        card = yaml.safe_load(f)
    alpha = float(card["config"]["alpha"])
    ours_mean = float(card["gamma_n_mean"])
    ours_std = float(card["gamma_n_std"])

    if alpha not in reference and not any(abs(alpha - k) < 1e-9 for k in reference):
        return {
            "alpha": alpha, "card": str(card_path),
            "verdict": "no-reference",
            "ours_mean": ours_mean, "ours_std": ours_std,
        }

    ref_key = next(k for k in reference if abs(float(k) - alpha) < 1e-9)
    ref = reference[ref_key]
    ref_mean = float(ref["gamma_n_mean"])
    ref_std = float(ref["gamma_n_std"])
    z = abs(ours_mean - ref_mean) / max(ref_std, 1e-12)

    return {
        "alpha": alpha,
        "card": str(card_path),
        "ref_mean": ref_mean,
        "ref_std": ref_std,
        "ours_mean": ours_mean,
        "ours_std": ours_std,
        "z": z,
        "verdict": verdict(z),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cards", nargs="+", help="Paths to data_card.yaml files.")
    p.add_argument(
        "--reference",
        default=str(Path(__file__).parent / "reference_gammas.yaml"),
        help="Path to reference table.",
    )
    args = p.parse_args(argv)

    reference = load_reference(args.reference)

    rows = [compare_one(Path(c), reference) for c in args.cards]

    # Pretty print
    print(
        f"{'alpha':>6} {'ref_mean':>10} {'ref_std':>9} "
        f"{'ours_mean':>11} {'ours_std':>10} {'z':>6} {'verdict':>10}"
    )
    print("-" * 70)
    fail = 0
    for r in rows:
        if r["verdict"] == "no-reference":
            print(
                f"{r['alpha']:>6.3f}    (no reference) "
                f"{r['ours_mean']:>+11.4f} {r['ours_std']:>10.4f}    --    "
                f"{r['verdict']:>10}"
            )
            continue
        marker = "" if r["verdict"] == "ok" else "  <-- check"
        print(
            f"{r['alpha']:>6.3f} {r['ref_mean']:>+10.4f} {r['ref_std']:>9.4f} "
            f"{r['ours_mean']:>+11.4f} {r['ours_std']:>10.4f} "
            f"{r['z']:>6.2f} {r['verdict']:>10}{marker}"
        )
        if r["verdict"] == "fail":
            fail += 1

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
