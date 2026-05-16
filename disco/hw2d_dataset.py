"""HW2D → DISCO snippet dataset adapter.

Author: Anthony Poole

DISCO's upstream trainer (``disco/upstream/train_reference.py``) consumes a
``MixedDataset`` whose per-sample dict is::

    {
        'input_fields'        : (n_past,   C, H, W)  float tensor
        'output_fields'       : (n_future, C, H, W)  float tensor
        'boundary_conditions' : tensor of ints (1 = periodic per axis)
        'name'                : str       (dataset name, e.g. 'hw2d_a01')
        'file'                : str       (source filename)
        'index'               : tensor    (global sample index)
        'field_labels'        : tensor    (per-channel global field ids)
        'file_index'          : int       (which sub-dataset)
    }

This module implements that contract for HW2D trajectories produced by
``hw.dns`` (one long HDF5 per alpha). Differences from upstream's
``BaseHDF5DirectoryDataset``:

  - one trajectory per HDF5, not many → no sample axis
  - a burn-in head must be discarded
  - we hold out a *tail window* per alpha as the G1 evaluation set, so
    'train' and 'g1' splits are disjoint in time

The classes here are intentionally PyTorch-only and free of upstream
imports, so they can be CPU-smoke-tested without the full DISCO stack.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


__all__ = [
    "HW2DDataset",
    "HW2DMixedDataset",
    "split_indices",
]


@dataclass
class _Split:
    """Index range [t_lo, t_hi) into the post-burn-in trajectory."""
    t_lo: int
    t_hi: int


def split_indices(
    n_frames_post_burn_in: int,
    n_past: int,
    n_future: int,
    g1_tail_frac: float = 0.1,
) -> Tuple[_Split, _Split]:
    """Carve a post-burn-in trajectory into (train, g1_tail) ranges.

    A *sample* indexed by start time ``t`` consumes frames
    ``[t, t + n_past + n_future)``. To prevent leakage from the G1 tail
    back into the train set, we shift the train upper bound left by
    ``n_past + n_future`` so the very last train sample ends strictly
    before the first G1 frame.
    """
    win = n_past + n_future
    n_g1 = int(round(n_frames_post_burn_in * g1_tail_frac))
    t_split = n_frames_post_burn_in - n_g1
    train = _Split(t_lo=0, t_hi=t_split - win)
    g1 = _Split(t_lo=t_split, t_hi=n_frames_post_burn_in - win)
    return train, g1


class HW2DDataset(Dataset):
    """One alpha-regime of HW2D as a snippet dataset.

    Parameters
    ----------
    main_path
        Directory containing exactly one ``trajectory.h5`` (the HW2D
        schema with ``/density`` and ``/phi``).
    resolution
        Target ``(H, W)`` after spatial subsampling. If the file's
        native resolution is larger, the loader takes every Nth pixel
        (no anti-alias). Set equal to native res to disable.
    n_past, n_future
        Snippet length and target length, in frames.
    burn_in_frac
        Fraction of the head of the file to discard as transient.
    split
        ``'train'`` or ``'g1'``. ``'g1'`` returns samples drawn from
        the held-out tail of the same trajectory.
    g1_tail_frac
        Fraction of the post-burn-in trajectory reserved as the G1 tail.
    name
        Logical dataset name (e.g. ``'hw2d_a01'``), surfaced in the
        per-sample dict so the hypernet can be conditioned on identity.
    file_index
        Position of this sub-dataset within an enclosing mixed dataset.

    Notes
    -----
    We open the HDF5 lazily (the first time it's needed in each worker)
    and keep the handle. This matches upstream's pattern and avoids the
    cost of re-opening every ``__getitem__`` call.
    """

    BCS_PERIODIC_2D = (1, 1)

    def __init__(
        self,
        main_path: str,
        resolution: Tuple[int, int],
        n_past: int,
        n_future: int,
        burn_in_frac: float = 0.5,
        split: str = "train",
        g1_tail_frac: float = 0.1,
        name: str = "hw2d",
        file_index: int = 0,
    ):
        super().__init__()
        assert split in ("train", "g1"), f"unknown split: {split}"
        self.main_path = main_path
        self.resolution = tuple(resolution)
        self.n_past = int(n_past)
        self.n_future = int(n_future)
        self.burn_in_frac = float(burn_in_frac)
        self.split = split
        self.g1_tail_frac = float(g1_tail_frac)
        self.name = name
        self.file_index = int(file_index)

        self._file_path = self._find_trajectory_h5(main_path)
        self._h5 = None
        self._t_burn: Optional[int] = None
        self._n_post: Optional[int] = None
        self._native_shape: Optional[Tuple[int, int]] = None
        self._stride: Optional[Tuple[int, int]] = None
        self._train_split: Optional[_Split] = None
        self._g1_split: Optional[_Split] = None
        self._scan_metadata()

    @staticmethod
    def _find_trajectory_h5(main_path: str) -> str:
        candidates = sorted(
            glob.glob(os.path.join(main_path, "trajectory.h5"))
            + glob.glob(os.path.join(main_path, "*.h5"))
            + glob.glob(os.path.join(main_path, "*.hdf5"))
        )
        if not candidates:
            raise FileNotFoundError(f"No HDF5 file found under {main_path}")
        return candidates[0]

    def _scan_metadata(self) -> None:
        import h5py  # noqa: PLC0415

        with h5py.File(self._file_path, "r") as f:
            n_t, n_y, n_x = f["density"].shape
        self._native_shape = (n_y, n_x)
        sy = max(1, n_y // self.resolution[0])
        sx = max(1, n_x // self.resolution[1])
        self._stride = (sy, sx)

        self._t_burn = int(self.burn_in_frac * n_t)
        self._n_post = n_t - self._t_burn
        self._train_split, self._g1_split = split_indices(
            self._n_post, self.n_past, self.n_future, self.g1_tail_frac
        )

    def _open(self):
        if self._h5 is None:
            import h5py  # noqa: PLC0415
            self._h5 = h5py.File(self._file_path, "r", swmr=True)
        return self._h5

    def _active_split(self) -> _Split:
        return self._train_split if self.split == "train" else self._g1_split

    def __len__(self) -> int:
        s = self._active_split()
        return max(0, s.t_hi - s.t_lo)

    def __getitem__(self, index: int) -> Dict:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        s = self._active_split()
        t = s.t_lo + index + self._t_burn

        h5 = self._open()
        sy, sx = self._stride
        n_past = self.n_past
        n_future = self.n_future

        past = np.stack(
            [
                h5["density"][t : t + n_past, ::sy, ::sx],
                h5["phi"][t : t + n_past, ::sy, ::sx],
            ],
            axis=1,
        )
        fut = np.stack(
            [
                h5["density"][t + n_past : t + n_past + n_future, ::sy, ::sx],
                h5["phi"][t + n_past : t + n_past + n_future, ::sy, ::sx],
            ],
            axis=1,
        )

        return {
            "input_fields": torch.from_numpy(past.astype(np.float32)),
            "output_fields": torch.from_numpy(fut.astype(np.float32)),
            "boundary_conditions": torch.as_tensor(self.BCS_PERIODIC_2D, dtype=torch.long),
            "name": self.name,
            "file": os.path.basename(self._file_path),
            "index": torch.as_tensor(index, dtype=torch.long),
            "file_index": self.file_index,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state


class HW2DMixedDataset(Dataset):
    """Concatenation of several ``HW2DDataset`` instances.

    The dict returned by ``__getitem__`` adds two keys to the per-α dict:

      - ``field_labels`` : global per-channel ids (here just ``[0, 1]``,
        since both density and phi are HW2D-specific and shared across
        all alphas). Mirrors upstream's ``MixedDataset`` contract.
      - ``file_index``   : index of the sub-dataset inside this mix.

    Notes
    -----
    The hypernetwork is supposed to infer the alpha regime from the
    context snippet; we deliberately do **not** expose alpha as a label.
    """

    HW2D_FIELD_LABELS = (0, 1)  # density=0, phi=1; shared across all HW2D alphas

    def __init__(
        self,
        dataset_names: Sequence[str],
        specs_table: Dict[str, Dict],
        n_past: int,
        n_future: int,
        split: str = "train",
        g1_tail_frac: float = 0.1,
        resolution_override: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.dataset_names = list(dataset_names)
        self.split = split

        sub_dsets: List[HW2DDataset] = []
        for i, name in enumerate(self.dataset_names):
            spec = specs_table[name]
            resolution = resolution_override or spec["resolution"]
            sub_dsets.append(
                HW2DDataset(
                    main_path=spec["main_path"],
                    resolution=resolution,
                    n_past=n_past,
                    n_future=n_future,
                    burn_in_frac=spec.get("burn_in_frac", 0.5),
                    split=split,
                    g1_tail_frac=g1_tail_frac,
                    name=name,
                    file_index=i,
                )
            )
        self.sub_dsets = sub_dsets
        self.offsets: List[int] = [0]
        for d in sub_dsets:
            self.offsets.append(self.offsets[-1] + len(d))

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, index: int) -> Dict:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        # locate which sub-dataset this global index falls in
        file_idx = max(0, np.searchsorted(self.offsets, index, side="right") - 1)
        local_idx = index - self.offsets[file_idx]
        d = self.sub_dsets[file_idx][local_idx]
        d["field_labels"] = torch.as_tensor(self.HW2D_FIELD_LABELS, dtype=torch.long)
        d["file_index"] = int(file_idx)
        return d


# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "usage: python -m disco.hw2d_dataset <path/to/alpha_dir/> [n_past] [n_future]"
        )
        sys.exit(1)

    main_path = sys.argv[1]
    n_past = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    n_future = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    d = HW2DDataset(
        main_path=main_path,
        resolution=(256, 256),
        n_past=n_past,
        n_future=n_future,
        burn_in_frac=0.5,
        split="train",
        name="hw2d_smoke",
    )
    print(
        f"file              : {d._file_path}\n"
        f"native (H, W)     : {d._native_shape}\n"
        f"target (H, W)     : {d.resolution}\n"
        f"stride (sy, sx)   : {d._stride}\n"
        f"burn_in frame     : {d._t_burn}\n"
        f"post-burn-in len  : {d._n_post}\n"
        f"len(train split)  : {len(d)}\n"
    )
    sample = d[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:22s} {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k:22s} {v!r}")
