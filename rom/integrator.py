"""P5: differentiable time integrator in the reduced space.

Author: Anthony Poole

Two backends, same interface:
  - rk4:    explicit Runge-Kutta 4. Phase 3 default. Fine when stiffness
            is softened by the Fourier truncation / spectral filter.
  - etdrk4: exponential time-differencing RK4 for the linear+nonlinear split.
            Required if the operator includes the full nabla^6 hyperdiffusion.

Both expose:
  step(z, f_theta, dt) -> z_next
  rollout(z0, f_theta, n_steps, dt) -> [z0, z1, ..., zn]

where f_theta is a callable z -> dz/dt obtained from rom.operator with
coefficients theta bound. Rollout is reverse-mode differentiable.
"""

raise NotImplementedError("rom.integrator: implement Phase 3 (rk4 first)")
