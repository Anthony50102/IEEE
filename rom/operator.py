"""P2: structured reduced operator.

Author: Anthony Poole

For HW, the operator is

    dz/dt = A(theta) z + F_tri(theta) (z otimes z)|_sparse

where:
  - A is a small dense matrix mixing density and potential modes,
    parameterized by alpha, kappa, and the dissipation coefficient(s).
  - F_tri is the Poisson-bracket triadic-sum tensor in k-space, sparse by
    construction (only triples (p, q, k) with p + q = k contribute).

Coefficients theta are either:
  (a) hand-set from the analytic HW equations (used for the Phase 3 sanity
      check that the framework is a valid spectral solver), or
  (b) emitted by the coefficient head (rom/head.py) at inference time.

Sanity check (run as __main__): with hand-set theta, compute RHS at a fixed
state and compare to hw.physics.hw_rhs at the same state; relative error
should be at numerical precision.
"""

raise NotImplementedError("rom.operator: implement Phase 3")
