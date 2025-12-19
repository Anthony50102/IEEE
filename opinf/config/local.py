# Local testing configurations
from .HW import *

data_dir = "/work2/10407/anthony50102/frontera/data/hw2d_sim/t600_d256x256_raw/" # Raw data directory
output_path = "/work2/10407/anthony50102/frontera/data/sciml_roms_hasegawa_wakatani/" # Output directory
training_files = [
    data_dir + "hw2d_ss.025_time1000_grid512_c11.5_k0.15_N3_nu5e-8_down8.h5",
    data_dir + "hw2d_ss.025_time1000_grid512_c11_k0.15_N3_nu5e-8_down8.h5",
]
test_files = [
    data_dir + "hw2d_ss.025_time1000_grid512_c12_k0.15_N3_nu5e-8_down8.h5"
]

POD_file = output_path + "POD.npz"
Xhat_file = output_path + "X_hat.npy"

ridge_alf_lin_all = np.linspace(1e2, 1e5, 10)
ridge_alf_quad_all = np.linspace(1e11, 1e14, 10)

gamma_reg_lin = np.linspace(1e-4, 1e1, 20)
gamma_reg_quad = np.linspace(1e-3, 1e2, 20)
