from simplified_opinf.config import *
from simplified_opinf.utils import *

if __name__ == '__main__':

	######## INITIALIZATION ########
	# MPI initialization
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()

	# the start and end indices, and the total number of snapshots for each MPI rank
	nx_i_start, nx_i_end, nx_i = distribute_nx(rank, nx, size)
	###### INITIALIZATION END ######

	start_time_global 		= MPI.Wtime()
	start_time_data_loading = MPI.Wtime()
	
	######## STEP I: DISTRIBUTED TRAINING DATA LOADING ########
	# allocate memory for the snapshot data corresponding to each MPI rank
	# note that the full snapshot data has been saved to disk in HDF5 format
	Q_rank = np.zeros((ns * nx_i, nt))
	with h5.File(H5_training_snapshots, 'r') as file:

		for j in range(ns):
			Q_rank[j*nx_i : (j + 1)*nx_i, :] = \
			    file[state_variables[j]][nx_i_start : nx_i_end, :]

	file.close()

	end_time_data_loading 	= MPI.Wtime()
	data_loading_time 		= end_time_data_loading - start_time_data_loading
	#################### STEP I END ###########################
	

	######## STEP II: DISTRIBUTED DATA TRANSFORMATIONS ########
	compute_time 					= 0	
	communication_time 				= 0
	start_time_data_transformations = MPI.Wtime()

	end_time_data_transformations 	= MPI.Wtime()
	compute_time					+= end_time_data_transformations - start_time_data_transformations
	#################### STEP II END ##########################

	
	######## STEP III: DISTRIBUTED DIMENSIONALITY REDUCTION ########
	start_time_matmul 	= MPI.Wtime()
	# compute the local Gram matrices on each rank
	D_rank  			= np.matmul(Q_rank.T, Q_rank)
	end_time_matmul 	= MPI.Wtime()
	compute_time		+= end_time_matmul - start_time_matmul
	
	start_time_reduction = MPI.Wtime()

	# aggregate local Gram matrices to form global Gram matrix and distribute the result to all ranks
	D_global = np.zeros_like(D_rank)
	comm.Allreduce(D_rank, D_global, op=MPI.SUM)

	end_time_reduction 		= MPI.Wtime()
	communication_time		+= end_time_reduction - start_time_reduction

	start_time = MPI.Wtime()

	# compute the eigendecomposition of the positive, semi-definite global Gram matrix
	eigs, eigv = np.linalg.eigh(D_global)

	# order eigenpairs by increasing eigenvalue magnitude
	sorted_indices 	= np.argsort(eigs)[::-1]
	eigs 			= eigs[sorted_indices]
	eigv 			= eigv[:, sorted_indices]

	if rank == 0:
		np.save('postprocessing/dOpInf_postprocessing/Sigma_sq_global.npy', eigs)

	# compute retained energy for r bteween 1 and nt
	ret_energy 	= np.cumsum(eigs)/np.sum(eigs)
	# select reduced dimension r for that the retained energy exceeds the prescribed threshold
	r 			= np.argmax(ret_energy > target_ret_energy) + 1

	# compute the auxiliary Tr matrix
	Tr_global 	= np.matmul(eigv[:, :r], np.diag(eigs[:r]**(-0.5)))
	# compute the low-dimensional representation of the high-dimensional transformed snapshot data
	Qhat_global = np.matmul(Tr_global.T, D_global)

	end_time 		= MPI.Wtime()
	compute_time	+= end_time - start_time
	##################### STEP III END #############################


	# Terminate MPI
	MPI.Finalize()