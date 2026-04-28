"""ROM framework: context-conditioned structured reduced-order modeling.

Five primitives, one file each:
  basis.py        -- Fourier projection / lift (P1)
  operator.py     -- linear + Poisson-bracket-sparse quadratic operator (P2)
  encoder.py      -- MLP / Transformer over short snippets (P3)
  head.py         -- coefficient head with sign / sparsity priors (P4)
  integrator.py   -- differentiable RK4 / ETDRK4 in reduced space (P5)

Composed in:
  model.py        -- end-to-end module
  losses.py       -- one-step + rollout + QoI losses
  train.py        -- generic train loop driven by a YAML config

This package is PDE-class-agnostic in intent: HW specifics live in `hw/` and
are bound at config-load time, not via imports here.
"""
