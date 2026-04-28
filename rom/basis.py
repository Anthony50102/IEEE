"""P1: reduced representation (Fourier basis for HW).

Author: Anthony Poole

Provides:
  project(u, k_max) -> z_hat       : real/physical space -> truncated Fourier coeffs
  lift(z_hat, n)    -> u           : truncated Fourier coeffs -> physical grid
  basis_indices(k_max, n)          : indices of retained modes (sorted by |k|)

Implementation note: 2D rfft for real fields; truncation by |k| <= k_max.

Sanity check (run as __main__): project then lift a random snapshot, verify
relative L2 error < 1e-10 for k_max equal to Nyquist, and that error grows
monotonically as k_max decreases.
"""

raise NotImplementedError("rom.basis: implement Phase 3")
