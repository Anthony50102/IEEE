"""HW2D DATASET_SPECS entries for DISCO.

These mirror the upstream ``DATASET_SPECS`` dict shape (see
``disco/upstream/utils/data_utils.py``) so a DISCO-style trainer can be
configured by name.

**Architectural note (2026-05-17).** DISCO's per-dataset machinery
(``param_gen_channels[dset_name]``, ``opnns[dset_name]``,
``theta_norm_{dset_name}``) exists to handle **different channel
counts across PDE families** (e.g. heat=1 field, shallow water=3,
Navier–Stokes=4). It is *not* a regime-conditioning mechanism. Within
HW2D, all αs share the same 2 channels (density n, electrostatic
potential φ), so registering each α as a separate dataset would
leak α-identity through the ``dset_name`` label and bypass the
hypernetwork's snippet-inference job. We therefore register HW2D as
**one** dataset (``"hw2d"``) and discriminate α only through the
snippet content. The per-α metadata below (``HW2D_ALPHA_META``) is
the source of truth for which on-disk file each α maps to; the
single-key ``HW2D_DATASET_SPECS`` is built from it.

Author: Anthony Poole

Schema fields (for an entry in ``HW2D_DATASET_SPECS``):
    main_path           absolute path to the directory containing ``trajectory.h5``.
                        In unified mode this is overridden per-α from
                        ``HW2D_ALPHA_META``; the value here is a placeholder
                        used only when DISCO introspects the spec.
    include_string      substring filter (kept for parity; unused for HW2D)
    resolution          target (H, W); the loader will spatial-stride if the
                        HDF5 is larger
    in_channels         2  (density n, electrostatic potential phi)
    spatial_ndims       2
    boundary_conditions 'periodic'
    n_steps             max length of the (past + future) snippet — set to the
                        smallest n_steps across the included αs so DISCO's
                        bookkeeping never asks for more frames than any file has
    group               'HW2D'
    burn_in_frac        fraction of the head of the trajectory to discard

The defaults below point at the Frontera data tree. Override
``data_root`` from the YAML or a build helper for local smoke runs.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Tuple

# Default location of HW2D DNS on Frontera ($WORK2 alloc 10407).
HW2D_FRONTERA_ROOT = "/work2/10407/anthony50102/frontera/data/IEEE/hw2d"


# ---------------------------------------------------------------------------
# Per-α metadata (source of truth)
# ---------------------------------------------------------------------------

HW2D_ALPHA_META: Dict[float, Dict] = {
    0.1: dict(dirname="alpha0.1_n512", n_steps=10700, role="train"),
    1.0: dict(dirname="alpha1.0_n512", n_steps=12001, role="train"),
    5.0: dict(dirname="alpha5.0_n512", n_steps=12001, role="train"),
    1.5: dict(dirname="alpha1.5_n512", n_steps=2001,  role="g3_held_out"),
}


def alpha_dir(alpha: float, root: str = HW2D_FRONTERA_ROOT) -> str:
    """Resolve an α value to an absolute directory containing its HDF5."""
    if alpha not in HW2D_ALPHA_META:
        raise KeyError(
            f"alpha={alpha} not in HW2D_ALPHA_META; "
            f"known αs: {sorted(HW2D_ALPHA_META)}"
        )
    return f"{root}/{HW2D_ALPHA_META[alpha]['dirname']}"


def alpha_paths(alphas: Iterable[float], root: str = HW2D_FRONTERA_ROOT) -> List[Tuple[float, str]]:
    """List of (alpha, main_path) for the requested αs, preserving input order."""
    return [(a, alpha_dir(a, root)) for a in alphas]


# ---------------------------------------------------------------------------
# Unified single-dataset spec
# ---------------------------------------------------------------------------

_HW2D_BASE: Dict = {
    "include_string": "",
    "resolution": (256, 256),
    "in_channels": 2,
    "spatial_ndims": 2,
    "boundary_conditions": "periodic",
    "group": "HW2D",
    "burn_in_frac": 0.5,
}


def make_unified_spec(
    alphas: Iterable[float],
    root: str = HW2D_FRONTERA_ROOT,
    dataset_name: str = "hw2d",
) -> Dict[str, Dict]:
    """Build a single-entry ``{dataset_name: spec}`` dict.

    The spec's ``n_steps`` is set to the minimum across the included
    αs so DISCO never asks for more frames than the shortest file has.
    ``main_path`` is set to the first α's directory; it's only used by
    upstream's bookkeeping (we drive the actual file I/O from
    ``HW2DMixedDataset`` directly).
    """
    alphas = list(alphas)
    if not alphas:
        raise ValueError("make_unified_spec requires at least one α")
    n_steps = min(HW2D_ALPHA_META[a]["n_steps"] for a in alphas)
    spec = deepcopy(_HW2D_BASE)
    spec.update(
        main_path=alpha_dir(alphas[0], root),
        n_steps=n_steps,
    )
    return {dataset_name: spec}


# ---------------------------------------------------------------------------
# Legacy per-α spec table (retained for tooling that introspects by name)
# ---------------------------------------------------------------------------

def _legacy_spec(alpha: float, dirname: str, n_steps: int, role: str) -> Dict:
    s = deepcopy(_HW2D_BASE)
    s.update(
        main_path=f"{HW2D_FRONTERA_ROOT}/{dirname}",
        n_steps=n_steps,
        alpha=alpha,
        role=role,
    )
    return s


HW2D_DATASET_SPECS: Dict[str, Dict] = {
    f"hw2d_a{int(round(a * 10)):02d}": _legacy_spec(
        alpha=a, dirname=m["dirname"], n_steps=m["n_steps"], role=m["role"]
    )
    for a, m in HW2D_ALPHA_META.items()
}


def with_root(specs: Dict[str, Dict], root: str) -> Dict[str, Dict]:
    """Return a copy of ``specs`` with ``main_path`` rewritten under ``root``.

    Useful when running locally with a sample HDF5 outside Frontera.
    """
    out = {}
    for name, s in specs.items():
        s2 = deepcopy(s)
        dirname = s["main_path"].rsplit("/", 1)[-1]
        s2["main_path"] = f"{root}/{dirname}"
        out[name] = s2
    return out

