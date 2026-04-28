"""Training losses.

Author: Anthony Poole

  - one_step:   MSE on z_{k+1} given z_k and the predicted operator
  - rollout:    MSE on the trajectory over a horizon T (curriculum-friendly)
  - qoi:        relative error on time-averaged Gamma_n / Gamma_c
  - reg:        coefficient regularization (L2 on theta, optional sparsity prior)

Total loss is a weighted sum; weights live in the experiment config.
"""

raise NotImplementedError("rom.losses: implement Phase 3")
