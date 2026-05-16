"""DISCO on HW2D — IEEE/CiSE Special Issue.

This package wraps the upstream DISCO architecture (Morel/Han/Oyallon 2025,
vendored in `upstream/`) with HW2D-specific dataloaders, configs, and
training/eval entry points.

Sub-modules:
    upstream/         — third-party DISCO code, frozen at ddd18f17
    hw2d_dataset.py   — HW2D snippet adapter for the DISCO trainer
    dataset_specs.py  — HW2D analogs of upstream's DATASET_SPECS dict
    train_hw2d.py     — multi-alpha training entry point

The headline claim of the paper this package supports:

    DISCO, when trained on Hasegawa-Wakatani snippets at alpha in {0.1, 1, 5},
    produces stable, statistically faithful rollouts at the held-out
    alpha = 1.5 without retraining.

Upstream-import shim
--------------------
The vendored upstream modules use ``from src.X import ...`` style imports
(see ``upstream/models/disco.py``, ``upstream/utils/data_utils.py``). To
let those imports resolve without touching upstream code, we register the
``upstream/`` sub-package as the alias ``src`` in ``sys.modules`` on first
import of ``disco``. This is invisible to the upstream code and keeps the
modification policy in ``upstream/SOURCE.md`` intact.
"""

import importlib as _importlib
import sys as _sys
import types as _types

if "src" not in _sys.modules:
    _upstream = _importlib.import_module(__name__ + ".upstream")
    _sys.modules["src"] = _upstream
    for _sub in ("models", "utils", "torchdiffeq"):
        try:
            _sys.modules[f"src.{_sub}"] = _importlib.import_module(
                __name__ + f".upstream.{_sub}"
            )
        except ImportError:
            pass

    # Stub `src.the_well.datasets.GenericWellDataset` — upstream's
    # ``data_utils.py`` imports it unconditionally for the_well datasets,
    # which we never use for HW2D. A placeholder class lets the import
    # succeed without our needing to vendor the the_well subpackage.
    if "src.the_well" not in _sys.modules:
        _the_well = _types.ModuleType("src.the_well")
        _the_well_dsets = _types.ModuleType("src.the_well.datasets")

        class _GenericWellDatasetStub:  # noqa: D401
            """Placeholder; HW2D pipeline never instantiates this."""

            def __init__(self, *_args, **_kwargs):
                raise RuntimeError(
                    "GenericWellDataset is not available in this build; "
                    "HW2D pipeline uses disco.hw2d_dataset instead."
                )

        _the_well_dsets.GenericWellDataset = _GenericWellDatasetStub
        _the_well.datasets = _the_well_dsets
        _sys.modules["src.the_well"] = _the_well
        _sys.modules["src.the_well.datasets"] = _the_well_dsets


def register_hw2d_specs():
    """Merge ``HW2D_DATASET_SPECS`` into the upstream ``DATASET_SPECS`` dict.

    Called by ``train_hw2d.py`` before constructing a ``DISCO`` model, so
    that the model's ``DATASET_SPECS[dname]`` lookups for HW2D names
    resolve. Safe to call more than once.
    """
    from disco.dataset_specs import HW2D_DATASET_SPECS  # noqa: PLC0415
    from src.utils.data_utils import DATASET_SPECS  # noqa: PLC0415

    DATASET_SPECS.update(HW2D_DATASET_SPECS)
    return DATASET_SPECS
