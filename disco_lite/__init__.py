"""DISCO-lite: unstructured-operator ablation of our framework.

Identical to `rom/` except for the operator (P2): instead of a structured
quadratic operator with Poisson-bracket sparsity, we use a generic U-Net
that maps z -> dz/dt in the reduced space (or directly in physical space if
the basis is the identity).

This is the controlled ablation for the paper's central claim: holding
encoder, head, integrator, data, and training budget fixed, swap structured
operator for unstructured operator, and measure the difference.
"""
