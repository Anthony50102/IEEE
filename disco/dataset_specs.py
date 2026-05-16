"""HW2D DATASET_SPECS entries for DISCO.

These mirror the upstream ``DATASET_SPECS`` dict shape (see
``disco/upstream/utils/data_utils.py``) so a DISCO-style trainer can be
configured by name. Each entry registers ONE alpha-regime of HW2D as a
separate dataset, because the DISCO hypernetwork infers the parameter
regime from the context snippet rather than from an explicit label.

Author: Anthony Poole

Schema fields:
    main_path           absolute path to the directory containing ``trajectory.h5``
    include_string      substring filter (kept for parity; unused for HW2D)
    resolution          target (H, W); the loader will spatial-stride if the
                        HDF5 is larger
    in_channels         2  (density n, electrostatic potential phi)
    spatial_ndims       2
    boundary_conditions 'periodic'
    n_steps             max length of the (past + future) snippet
    group               'HW2D'
    alpha               the HW adiabaticity parameter c1 (HW2D-specific extra field)
    burn_in_frac        fraction of the head of the trajectory to discard
    role                'train' | 'g3_held_out'   (HW2D-specific extra field)

The defaults below point at the Frontera data tree. Override
``main_path`` from the YAML or a build helper for local smoke runs.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict

# Default location of HW2D DNS on Frontera ($WORK2 alloc 10407).
HW2D_FRONTERA_ROOT = "/work2/10407/anthony50102/frontera/data/IEEE/hw2d"


_HW2D_BASE: Dict = {
    "include_string": "",
    "resolution": (256, 256),
    "in_channels": 2,
    "spatial_ndims": 2,
    "boundary_conditions": "periodic",
    "group": "HW2D",
    "burn_in_frac": 0.5,
}


def _spec(alpha: float, dirname: str, n_steps: int, role: str) -> Dict:
    s = deepcopy(_HW2D_BASE)
    s.update(
        main_path=f"{HW2D_FRONTERA_ROOT}/{dirname}",
        n_steps=n_steps,
        alpha=alpha,
        role=role,
    )
    return s


HW2D_A01_SPECS = _spec(alpha=0.1, dirname="alpha0.1_n512", n_steps=10700, role="train")
HW2D_A10_SPECS = _spec(alpha=1.0, dirname="alpha1.0_n512", n_steps=12001, role="train")
HW2D_A50_SPECS = _spec(alpha=5.0, dirname="alpha5.0_n512", n_steps=12001, role="train")
HW2D_A15_SPECS = _spec(alpha=1.5, dirname="alpha1.5_n512", n_steps=2001, role="g3_held_out")


HW2D_DATASET_SPECS: Dict[str, Dict] = {
    "hw2d_a01": HW2D_A01_SPECS,
    "hw2d_a10": HW2D_A10_SPECS,
    "hw2d_a50": HW2D_A50_SPECS,
    "hw2d_a15": HW2D_A15_SPECS,
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
