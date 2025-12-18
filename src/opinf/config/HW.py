import numpy as np
import traceback
import os
from opinf_for_hw.utils import *


# normalization
def normalize(X, td):
    X, scl = scale(X)  # automatically scales by max. abs. value

    return X


# function for rescaling
def rescale(X, Xm, td):
    ndw, K = X.shape
    n = int(ndw / dw)

    for i in range(dw):
        varmax = td[i, 1]
        varmin = td[i, 0]
        scl = np.maximum(np.abs(varmax), np.abs(varmin))
        X[i * n : (i + 1) * n, :] = (
            Xm[i * n : (i + 1) * n][:, np.newaxis] + X[i * n : (i + 1) * n, :] * scl
        )

    return X


dw = 2
dt = 2.5e-2

n_x = 512
n_y = 512
n_cells = n_x * n_y

t_init = 0
training_start = 0
training_end = 5000

training_size = training_end - training_start

svd_save = 100

r = 100
n_steps = 25000 - training_start

r_all = [64, 74, 82, 90, 100]


# ridge_alf_lin_all   = np.linspace(1e3, 1e8, 10)
# ridge_alf_quad_all  = np.linspace(1e6, 1e14, 10)

# gamma_reg_lin   = np.linspace(1e-3, 1e2, 20)
# gamma_reg_quad  = np.linspace(1e-3, 1e2, 20)

time_steps_rec = [
    999,
    1999,
    3999,
    5999,
    7999,
    9999,
    10000,
    11999,
    12999,
    13999,
    14999,
    15999,
    16999,
    17999,
    18999,
    20000,
]