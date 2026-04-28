"""P3: context encoder.

Author: Anthony Poole

Maps a short snippet of the trajectory (in reduced/Fourier space) to a
fixed-dimensional context vector c.

Two backends, same interface:
  - MLP:         small, flat, easy to debug. Phase 3 default.
  - Transformer: deeper, attention over time and (optionally) over modes.
                 Phase 4+ swap-in.

Both are nn.Module instances exposing forward(snippet) -> c.
"""

raise NotImplementedError("rom.encoder: implement Phase 3 (MLP first)")
