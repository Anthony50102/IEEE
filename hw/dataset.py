"""HW dataset loaders.

Author: Anthony Poole

Loads HW2D HDF5 trajectories and produces:
  - whole-trajectory tensors for OpInf-style baselines
  - (snippet, future-rollout) pairs for context-conditioned ROMs

Parametric loaders span multiple alpha values for B2 (affine-mu OpInf) and
B4 (our framework). HDF5 access uses h5netcdf via xarray, matching
`shared/data_io.py` conventions.

The detailed implementations are being ported from `shared/data_io.py` as
the refactor progresses.
"""

raise NotImplementedError("hw.dataset: port from shared/data_io.py during Phase 0")
