# Cluster settings
from .HW import *

data_dir = "/work2/10407/anthony50102/frontera/data/hw2d_sim/t600_d256x256_raw/" # Raw data directory
output_path = "/scratch2/10407/anthony50102/sciml_roms_hasegawa_wakatani/" # Output directory
training_files = [
    data_dir + "hw2d_sim_step0.025_end1_pts512_c11_k015_N3_nu5e-8_20250315142044_11702_0.h5"
]
test_files = [
    data_dir + "hw2d_sim_step0.025_end1_pts512_c11_k015_N3_nu5e-8_20250316085602_1116_2.h5"
]

POD_file = output_path + "POD.npz"
Xhat_file = output_path + "X_hat.npy"

# ridge_alf_lin_all = np.linspace(1e4, 1e-4, 9)
ridge_alf_lin_all = np.linspace(1e4, 1e-4, 4)
# ridge_alf_quad_all = np.linspace(1e5, 1e-5, 11)
ridge_alf_quad_all = np.linspace(1e5, 1e-5, 5)

# gamma_reg_lin = np.linspace(1e-4, 1e1, 4)
gamma_reg_lin = np.linspace(1e-4, 1e1, 2)
# gamma_reg_quad = np.linspace(1e-3, 1e2, 5)
gamma_reg_quad = np.linspace(1e-3, 1e2, 3)


ridge_alf_lin_all = np.linspace(1e2, 1e5, 10)
ridge_alf_quad_all = np.linspace(1e11, 1e14, 10)

gamma_reg_lin = np.linspace(1e-4, 1e1, 10)
gamma_reg_quad = np.linspace(1e-3, 1e2, 10)

ridge_alf_lin_all = np.linspace(1e0, 1e3, 10)
ridge_alf_quad_all = np.linspace(1e9, 1e12, 10)

gamma_reg_lin = np.linspace(1e-6, 1e-2, 10)
gamma_reg_quad = np.linspace(1e-8, 1e-2, 10)

ridge_alf_lin_all = np.linspace(1e-2, 1e3, 12)
ridge_alf_quad_all = np.linspace(1e7, 1e12, 12)

gamma_reg_lin = np.linspace(1e-8, 1e-2, 12)
gamma_reg_quad = np.linspace(1e-10, 1e-2, 12)

n_steps = 16001
