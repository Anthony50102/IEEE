"""P4: coefficient head.

Author: Anthony Poole

Maps the context vector c -> structured-operator coefficients theta.

Physics-imposed structure is enforced by construction, not learned:
  - dissipation rates are positive (softplus output)
  - reality of the operator on real fields (Hermitian symmetry of Fourier
    coefficients) is preserved by emitting only the independent half
  - sparsity of the quadratic tensor is fixed (the head emits scalars
    multiplying a fixed sparsity pattern, not the dense tensor)
"""

raise NotImplementedError("rom.head: implement Phase 3")
