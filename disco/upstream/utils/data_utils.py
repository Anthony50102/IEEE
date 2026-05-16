from typing import Tuple, List, TypeVar, Iterator, Dict, Sized, Optional
import os
import glob
import h5py
import numpy as np
import torch
import torch.nn
from torch.utils.data import Sampler, Dataset, DataLoader, DistributedSampler

from src.utils.database import subsample
from src.the_well.datasets import GenericWellDataset


PDEBENCH_PATH = "..."  # TODO: replace this with your own path to the PDEBench dataset
THE_WELL_PATH = "..."  # TODO: replace this with your own path to the the_well dataset

BURGERS_SPECS = {
    'main_path': PDEBENCH_PATH + '/1D/Burgers/Train',
    'include_string': '',
    'resolution': (1024,),
    'in_channels': 1,
    'spatial_ndims': 1,
    'boundary_conditions': 'periodic',
    'n_steps': 200,
    'group': 'PDEBench',
}
SWE_SPECS = {
    'main_path': PDEBENCH_PATH + '/2D/shallow-water',
    'include_string': '',
    'resolution': (128,128),
    'in_channels': 1,
    'spatial_ndims': 2,
    'boundary_conditions': 'open',
    'n_steps': 100,
    'group': 'PDEBench',
}
DIFFRE2D_SPECS = {
    'main_path': PDEBENCH_PATH + '/2D/diffusion-reaction',
    'include_string': '',
    'resolution': (128,128),
    'in_channels': 2,
    'spatial_ndims': 2,
    'boundary_conditions': 'neumann',
    'n_steps': 100,
    'group': 'PDEBench',
}
INCOMPNS_SPECS = {
    'main_path': PDEBENCH_PATH + '/2D/NS_incom',
    'include_string': '',
    'resolution': (512,512),
    'in_channels': 3,
    'spatial_ndims': 2,
    'boundary_conditions': 'dirichlet',
    'n_steps': 1000,
    'group': 'PDEBench',
}
COMPNS128_SPECS = {
    'main_path': PDEBENCH_PATH + '/2D/CFD/2D_Train_Rand',
    'include_string': '128',
    'resolution': (128,128),
    'in_channels': 4,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 21,
    'group': 'PDEBench',
}
COMPNS512_SPECS = {
    'main_path': PDEBENCH_PATH + '/2D/CFD/2D_Train_Rand',
    'include_string': '512',
    'resolution': (512,512),
    'in_channels': 4,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 21,
    'group': 'PDEBench',
}
COMPNS_SPECS = {
    'main_path': PDEBENCH_PATH + '/2D/CFD/2D_Train_Turb',
    'include_string': '',
    'resolution': (512,512),
    'in_channels': 4,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 21,
    'group': 'PDEBench',
}
EULER_PBC_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/euler_multi_quadrants_periodicBC',
    'include_string': '',
    'resolution': (512,512),
    'in_channels': 5,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 100,
    'group': 'the_well',
}
EULER_OBC_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/euler_multi_quadrants_openBC',
    'include_string': '',
    'resolution': (512,512),
    'in_channels': 5,
    'spatial_ndims': 2,
    'boundary_conditions': 'open',
    'n_steps': 100,
    'group': 'the_well',
}
ACTIVE_MATTER_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/active_matter',
    'include_string': '',
    'resolution': (256,256),
    'in_channels': 11,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 81,
    'group': 'the_well',
}
CONVECTIVE_ENVELOPPE_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/convective_envelope_rsg',
    'include_string': '',
    # 'resolution': (256,128,256),  # stored resolution
    'resolution': (64,32,64),
    'in_channels': 6,
    'spatial_ndims': 3,
    'boundary_conditions': 'custom',  # seems Neumann, but should be confirmed
    'n_steps': 100,
    'group': 'the_well',
}
GRAY_SCOTT_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/gray_scott_reaction_diffusion',
    'include_string': '',
    'resolution': (128,128),
    'in_channels': 2,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 1001,
    'group': 'the_well',
}
HELMHOLTZ_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/helmholtz_staircase',
    'include_string': '',
    'resolution': (1024,256),
    'in_channels': 2,
    'spatial_ndims': 2,
    'boundary_conditions': 'neumann',
    'n_steps': 50,
    'group': 'the_well',
}
MHD_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/MHD_64',
    'include_string': '',
    'resolution': (64,64,64),
    'in_channels': 7,
    'spatial_ndims': 3,
    'boundary_conditions': 'periodic',
    'n_steps': 100,
    'group': 'the_well',
}
RAYLEIGH_BENARD_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/rayleigh_benard_uniform',
    'include_string': '',
    'resolution': (512,128),
    'in_channels': 4,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic x dirichlet',
    'n_steps': 200,
    'group': 'the_well',
} 
RAYLEIGH_TAYLOR_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/rayleigh_taylor_instability',
    'include_string': '',
    # 'resolution': (128,128,128),  # stored resolution 
    'resolution': (64,64,64),
    'in_channels': 4,
    'spatial_ndims': 3,
    'boundary_conditions': 'periodic x dirichlet',
    'n_steps': 120,  # the github says 119
    'group': 'the_well',
}
SHEARFLOW_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/shear_flow',
    'include_string': '',
    'resolution': (256,512),
    'in_channels': 4,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 200,
    'group': 'the_well',
}
SUPERNOVA_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/supernova_explosion_64',
    'include_string': '',
    'resolution': (64,64,64),
    'in_channels': 6,
    'spatial_ndims': 3,
    'boundary_conditions': 'open',
    'n_steps': 59,
    'group': 'the_well',
}
GRAVITY_COOLING_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/turbulence_gravity_cooling',
    'include_string': '',
    'resolution': (64,64,64),
    'in_channels': 6,
    'spatial_ndims': 3,
    'boundary_conditions': 'open',
    'n_steps': 50,
    'group': 'the_well',
}
RADIATIVE_LAYER_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/turbulent_radiative_layer_2D',
    'include_string': '',
    'resolution': (128,384),
    'in_channels': 4,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic x neumann',
    'n_steps': 101,
    'group': 'the_well',
}
COMPNS_M1_SPECS = {
    'main_path': PDEBENCH_PATH + '/2D/CFD/2D_Train_Rand',
    'include_string': 'M1.0',
    'resolution': (512,512),  # the maximum resolution achievable
    'in_channels': 4,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 21,
    'group': 'PDEBench',
}
COMPNS_M01_SPECS = {
    'main_path': PDEBENCH_PATH + '/2D/CFD/2D_Train_Rand',
    'include_string': 'M0.1',
    'resolution': (512,512),  # the maximum resolution achievable
    'in_channels': 4,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 21,
    'group': 'PDEBench',
}
EULER_OBC_FINETUNE_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/euler_multi_quadrants_openBC',
    'include_string': 'gamma_1.3',
    'resolution': (512,512),
    'in_channels': 5,
    'spatial_ndims': 2,
    'boundary_conditions': 'open',
    'n_steps': 100,
    'group': 'the_well',
}
DATASET_SPECS = {
    'burgers': BURGERS_SPECS,
    'swe': SWE_SPECS,
    'diffre2d': DIFFRE2D_SPECS,
    'incompNS': INCOMPNS_SPECS,
    'compNS128': COMPNS128_SPECS,
    'compNS512': COMPNS512_SPECS,
    'compNS': COMPNS_SPECS,
    'active_matter': ACTIVE_MATTER_SPECS,
    'convective_envelope': CONVECTIVE_ENVELOPPE_SPECS,
    'euler_obc': EULER_OBC_SPECS,
    'euler_pbc': EULER_PBC_SPECS,
    'gray_scott': GRAY_SCOTT_SPECS,
    'helmholtz': HELMHOLTZ_SPECS,
    'mhd': MHD_SPECS,
    'rayleigh_benard': RAYLEIGH_BENARD_SPECS,
    'rayleigh_taylor': RAYLEIGH_TAYLOR_SPECS,
    'shear_flow': SHEARFLOW_SPECS,
    'supernova': SUPERNOVA_SPECS,
    'gravity_cooling': GRAVITY_COOLING_SPECS,
    'radiative_layer': RADIATIVE_LAYER_SPECS,
    # finetuning datasets
    'compNSM1.0': COMPNS_M1_SPECS,
    'compNSM0.1': COMPNS_M01_SPECS,
    'euler_obc_finetune': EULER_OBC_FINETUNE_SPECS
}
 

broken_paths = [
    PDEBENCH_PATH + '/1D/Burgers/Train/1D_Burgers_Sols_Nu4.0.hdf5'
]


class BaseHDF5DirectoryDataset(Dataset):
    """
    Base class for data loaders. Returns data in T x B x C x H x W format.

    Note - doesn't currently normalize because the data is on wildly different
    scales but probably should.

    Split is provided so I can be lazy and not separate out HDF5 files.

    Takes in path to directory of HDF5 files to construct dset. 

    Args:
        path (str): Path to directory of HDF5 files
        include_string (str): Only include files with this string in name
        n_steps (int): Number of steps to include in each sample
        dt (int): Time step between samples
        split (str): train/val/test split
        train_val_test (tuple): Percent of data to use for train/val/test
        subname (str): Name to use for dataset
        split_level (str): 'sample' or 'file' - whether to split by samples within a file
                        (useful for data segmented by parameters) or file (mostly INS right now)
    """
    def __init__(self, path, resolution, include_string='', n_steps=1, dt=1, split='train', 
                 train_val_test=None, subname=None, extra_specific=False):
        super().__init__()
        self.path = path
        self.resolution = resolution
        self.split = split
        self.extra_specific = extra_specific # Whether to use parameters in name
        if subname is None:
            self.subname = path.split('/')[-1]
        else:
            self.subname = subname
        self.dt = 1
        self.n_steps = n_steps
        self.include_string = include_string
        # self.time_index, self.sample_index = self._set_specifics()
        self.train_val_test = train_val_test
        self.partition = {'train': 0, 'valid': 1, 'test': 2}[split]
        self.time_index, self.sample_index, self.field_names, self.type, self.split_level = self._specifics()
        self._get_directory_stats(path)
        if self.extra_specific:
            self.title = self.more_specific_title(self.type, path, include_string)
        else:
            self.title = self.type
        self.dataset_name = self.get_name() + self.include_string

    def get_name(self, full_name=False):
        if full_name:
            return self.subname + '_' + self.type
        else:
            return self.type
    
    def more_specific_title(self, type, path, include_string):
        """
        Override this to add more info to the dataset name
        """
        return type

    @staticmethod
    def _specifics():
        # Sets self.field_names, self.dataset_type
        raise NotImplementedError # Per dset
    
    def get_per_file_dsets(self):
        if self.split_level == 'file' or len(self.files_paths) == 1:
            return [self]
        else:
            sub_dsets = []
            for file in self.files_paths:
                subd = self.__class__(self.path, file, n_steps=self.n_steps, dt=self.dt, split=self.split,
                               train_val_test=self.train_val_test, subname=self.subname,
                                 extra_specific=True)
                sub_dsets.append(subd)
            return sub_dsets

    def _get_specific_stats(self, f):
        raise NotImplementedError # Per dset
    
    def _get_specific_bcs(self, f):
        raise NotImplementedError # Per dset

    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        raise NotImplementedError # Per dset - should be (x=(-history:local_idx+dt) so that get_item can split into x, y

    def _get_directory_stats(self, path):
        self.files_paths = glob.glob(path + "/*.h5") + glob.glob(path + "/*.hdf5")
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        self.file_steps = []
        self.file_nsteps = []
        self.file_samples = []
        self.split_offsets = []
        self.offsets = [0]
        file_paths = []
        for file in self.files_paths:
            # Total hack to avoid complications from folder with two sizes. 
            if len(self.include_string) > 0 and self.include_string not in file:
                continue
            elif file in broken_paths:
                continue
            else:
                file_paths.append(file)
                try:
                    with h5py.File(file, 'r') as _f:
                        samples, steps = self._get_specific_stats(_f)
                        if steps-self.n_steps-(self.dt-1) < 1:
                            print('WARNING: File {} has {} steps, but n_steps is {}. Setting file steps = max allowable.'.format(file, steps, self.n_steps))
                            file_nsteps = steps - self.dt
                        else:
                            file_nsteps = self.n_steps
                        self.file_nsteps.append(file_nsteps)
                        self.file_steps.append(steps-file_nsteps-(self.dt-1))
                        if self.split_level == 'sample':
                            # Compute which are in the given partition
                            partition = self.partition
                            sample_per_part = np.ceil(np.array(self.train_val_test)*samples).astype(int)
                            # Make sure rounding works
                            sample_per_part[2] = max(samples - sample_per_part[0] - sample_per_part[1], 0)
                            # I forget where the file steps formula came from, but offset by steps per sample
                            # * samples of previous partitions
                            self.split_offsets.append(self.file_steps[-1]*sum(sample_per_part[:partition]))
                            split_samples = sample_per_part[partition]
                        else:
                            split_samples = samples
                        self.file_samples.append(split_samples)
                        self.offsets.append(self.offsets[-1]+(steps-file_nsteps-(self.dt-1))*split_samples)
                except:
                    print('WARNING: Failed to open file {}. Continuing without it.'.format(file))
                    raise RuntimeError('Failed to open file {}'.format(file))
        # print(self.file_steps, self.file_samples)
        self.files_paths = file_paths
        self.offsets[0] = -1 # Just to make sure it doesn't put us in file -1
        self.files = [None for _ in self.files_paths]
        self.len = self.offsets[-1]
        if self.split_level == 'file':
        # Figure out our split offset - by sample
            if self.train_val_test is None:
                print('WARNING: No train/val/test split specified. Using all data for training.')
                self.split_offset = 0
                self.len = self.offsets[-1]
            else:
                # print('Using train/val/test split: {}'.format(self.train_val_test))
                total_samples = sum(self.file_samples)
                ideal_split_offsets = [int(self.train_val_test[i]*total_samples) for i in range(3)]
                # Doing this the naive way because I only need to do it once
                # Iterate through files until we get enough samples for set
                end_ind = 0
                for i in range(self.partition+1):
                    run_sum = 0
                    start_ind = end_ind
                    for samples, steps in zip(self.file_samples, self.file_steps):
                        run_sum += samples
                        if run_sum <= ideal_split_offsets[i]:
                            end_ind += samples * (steps)
                            if run_sum == ideal_split_offsets[i]:
                                break
                        else:
                            end_ind += np.abs((run_sum - samples) - ideal_split_offsets[i]) * (steps)
                            break
                self.split_offset = start_ind
                self.len = end_ind - start_ind
            # else:
                
                
    def _open_file(self, file_ind):
        _file = h5py.File(self.files_paths[file_ind], 'r')
        self.files[file_ind] = _file

    def __getitem__(self, index):
        if self.split_level == 'file':
            index = index + self.split_offset

        file_idx = int(np.searchsorted(self.offsets, index, side='right')-1) #which file we are on
        # print('sample from:', self.files_paths[file_idx])
        nsteps = self.file_nsteps[file_idx] # Number of steps per sample in given file
        local_idx = index - max(self.offsets[file_idx], 0) # First offset is -1
        if self.split_level == 'sample':
            sample_idx = (local_idx + self.split_offsets[file_idx]) // self.file_steps[file_idx]
        else:
            sample_idx = local_idx // self.file_steps[file_idx]
        time_idx = local_idx % self.file_steps[file_idx]

        #open image file
        if self.files[file_idx] is None:
            self._open_file(file_idx)

        #if we are on the last image in a file shift backward. Double counting until I bother fixing this.
        time_idx = time_idx - self.dt if time_idx >= self.file_steps[file_idx] else time_idx
        time_idx += nsteps
        try:
            # print(self.files[file_idx], sample_idx, time_idx, index)
            trajectory = self._reconstruct_sample(self.files[file_idx], sample_idx, time_idx, nsteps)
            bcs = self._get_specific_bcs(self.files[file_idx])
        except:
            raise RuntimeError(f'Failed to reconstruct sample for file {self.files_paths[file_idx]} sample {sample_idx} time {time_idx}')
        return {
            'input_fields': subsample(trajectory[:-1,...], self.resolution),
            'output_fields': subsample(trajectory[[-1],...], self.resolution),  # single step in the future
            # 'field_labels': torch.tensor(self.subset_dict[self.sub_dsets[file_idx].get_name()]),
            'boundary_conditions': torch.as_tensor(bcs),
            'name': self.dataset_name,
            'file': self.files_paths[file_idx].split('/')[-1],
            'index': torch.as_tensor(index),
            # 'file_index': file_idx,
            # 'name': dataset.get_name() + dataset.include_string  # not really supposed to be unique
        }
        # return trajectory[:-1], torch.as_tensor(bcs), trajectory[-1]

    def __len__(self):
        return self.len


class SWEDataset(BaseHDF5DirectoryDataset):
    @staticmethod
    def _specifics():
        time_index = 0
        sample_index = None
        field_names = ['h']
        type = 'swe'
        split_level = 'sample'
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f):
        samples = list(f.keys())
        steps = f[samples[0]]['data'].shape[0]
        return len(samples), steps
    
    def _get_specific_bcs(self, f):
        return [0, 0] # Non-periodic

    def _reconstruct_file(self, file, sample_idx):
        samples = list(file.keys())
        return file[samples[sample_idx]]['data'][:].transpose(0, 3, 1, 2)

    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        samples = list(file.keys())
        return file[samples[sample_idx]]['data'][time_idx-n_steps*self.dt:time_idx+self.dt].transpose(0, 3, 1, 2)


class DiffRe2DDataset(BaseHDF5DirectoryDataset):
    @staticmethod
    def _specifics():
        time_index = 0
        sample_index = None
        field_names = ['activator', 'inhibitor']
        type = 'diffre2d'
        split_level = 'sample'
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f):
        samples = list(f.keys())
        steps = f[samples[0]]['data'].shape[0]
        return len(samples), steps
    
    def _get_specific_bcs(self, f):
        return [0, 0] # Non-periodic

    def _reconstruct_file(self, file, sample_idx):
        samples = list(file.keys())
        return file[samples[sample_idx]]['data'][:].transpose(0, 3, 1, 2)
    
    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        samples = list(file.keys())
        return file[samples[sample_idx]]['data'][time_idx-n_steps*self.dt:time_idx+self.dt].transpose(0, 3, 1, 2)


class IncompNSDataset(BaseHDF5DirectoryDataset):
    """
    Order Vx, Vy, "particles"
    """
    @staticmethod
    def _specifics():
        time_index = 1
        sample_index = 0
        field_names = ['Vx', 'Vy', 'particles']
        type = 'incompNS'
        split_level = 'file'
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f):
        samples = f['velocity'].shape[0] 
        steps = f['velocity'].shape[1]# Per dset
        return samples, steps

    def _reconstruct_file(self, file, sample_idx):
        velocity = file['velocity'][sample_idx, :]
        particles = file['particles'][sample_idx, :]
        comb =  np.concatenate([velocity, particles], -1)
        return comb.transpose((0, 3, 1, 2))
    
    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        velocity = file['velocity'][sample_idx, time_idx-n_steps*self.dt:time_idx+self.dt]
        particles = file['particles'][sample_idx, time_idx-n_steps*self.dt:time_idx+self.dt]
        comb =  np.concatenate([velocity, particles], -1)
        return comb.transpose((0, 3, 1, 2))
    
    def _get_specific_bcs(self, f):
        return [0, 0] # Non-periodic


class CompNSDataset(BaseHDF5DirectoryDataset):
    """
    Order Vx, Vy, density, pressure
    """
    @staticmethod
    def _specifics():
        time_index = 1
        sample_index = 0
        field_names = ['Vx', 'Vy', 'density', 'pressure']
        type = 'compNS'
        split_level = 'sample'
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f):
        samples = f['Vx'].shape[0] 
        steps = f['Vx'].shape[1]# Per dset
        return samples, steps

    def more_specific_title(self, type, path, include_string):
        """
        Override this to add more info to the dataset name
        """
        cns_path = self.include_string.split('/')[-1].split('_')
        ic = cns_path[2]
        m = cns_path[3]
        res = cns_path[-2]

        return f'{type}_{ic}_{m}_res{res}'

    def _reconstruct_file(self, file, sample_idx):
        vx = file['Vx'][sample_idx, :]
        vy = file['Vy'][sample_idx, :]
        density = file['density'][sample_idx, :]
        p = file['pressure'][sample_idx, :]

        comb =  np.stack([vx, vy, density, p], 1)
        return comb#.transpose((0, 3, 1, 2))
    
    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        vx = file['Vx'][sample_idx, time_idx-n_steps*self.dt:time_idx+self.dt]
        vy = file['Vy'][sample_idx, time_idx-n_steps*self.dt:time_idx+self.dt]
        density = file['density'][sample_idx, time_idx-n_steps*self.dt:time_idx+self.dt]
        p = file['pressure'][sample_idx, time_idx-n_steps*self.dt:time_idx+self.dt]

        comb =  np.stack([vx, vy, density, p], 1)
        return comb#.transpose((0, 3, 1, 2))
    
    def _get_specific_bcs(self, f):
        return [1, 1] # Periodic


class BurgersDataset(BaseHDF5DirectoryDataset):
    """
    Order Vx, Vy, density, pressure
    """
    @staticmethod
    def _specifics():
        time_index = 1
        sample_index = 0
        field_names = ['Vx']
        type = 'burgers'
        split_level = 'sample'
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f):
        samples = f['tensor'].shape[0] 
        steps = f['tensor'].shape[1]# Per dset
        return samples, steps

    def _reconstruct_file(self, file, sample_idx):
        vx = file['tensor'][sample_idx, :]
        # print(vx.shape)
        vx = vx[:, None, :]
        return vx#.transpose((0, 3, 1, 2))
    
    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        vx = file['tensor'][sample_idx, time_idx-n_steps*self.dt:time_idx+self.dt]
        # print(vx.shape)
        vx = vx[:, None, :]
        return vx#.transpose((0, 3, 1, 2))
    
    def _get_specific_bcs(self, f):
        return [1] # Periodic


PDEBENCH_NAME_TO_OBJECT = {
    'swe': SWEDataset,
    'incompNS': IncompNSDataset,
    'diffre2d': DiffRe2DDataset,
    'compNS': CompNSDataset,
    'burgers': BurgersDataset,
}


class MixedDataset(Dataset):
    def __init__(
        self, dataset_names=[], n_past=1, n_future=1, dt=1, train_val_test=(.8, .1, .1),
        split='train', extended_names=False, train_offset=0
    ):
        super().__init__()
        # Global dicts used by Mixed DSET. 
        self.train_offset = train_offset
        self.path_list = [DATASET_SPECS[name]['main_path'] for name in dataset_names]
        # self.type_list = [('compNS' if name[:6] == 'compNS' in name else name) for name in dataset_names]
        self.include_string = [DATASET_SPECS[name]['include_string'] for name in dataset_names]
        # self.reso = [[int(d) for d in tuple(str(res).split('x'))] for res in self.reso]
        self.reso = [DATASET_SPECS[name]['resolution'] for name in dataset_names]

        self.extended_names = extended_names
        self.split = split
        self.sub_dsets = []
        self.offsets = [0]
        self.train_val_test = train_val_test

        for dset_name, path, include_string, reso in zip(dataset_names, self.path_list, self.include_string, self.reso):
            object_type = 'compNS' if dset_name[:6] == 'compNS' else dset_name
            if object_type in PDEBENCH_NAME_TO_OBJECT:  # PDEBench dataset
                n_steps_adjusted = min(n_past+n_future-1, DATASET_SPECS[dset_name]['n_steps']-1)  # nb of steps in a context (past)
                subdset = PDEBENCH_NAME_TO_OBJECT[object_type](
                    path, reso, include_string, n_steps=n_steps_adjusted, dt=dt, train_val_test=train_val_test, split=split
                )
            else:  # the-well dataset
                subdset = GenericWellDataset(
                    path=path,
                    # well_dataset_name=dset_name,
                    well_split_name=split,
                    resolution=reso,
                    include_filters=include_string,
                    exclude_filters=[],
                    n_steps_input=n_past,
                    n_steps_output=n_future,
                    dt_stride=dt,
                    name_override=dset_name,
                )
            # Check to make sure our dataset actually exists with these settings
            try:
                len(subdset)
            except ValueError:
                raise ValueError(f'Dataset {path} is empty. Check that n_steps < trajectory_length in file.')
            self.sub_dsets.append(subdset)
            self.offsets.append(self.offsets[-1]+len(self.sub_dsets[-1]))
        self.offsets[0] = -1

        self.subset_dict = self._build_subset_dict()

    def get_state_names(self):
        name_list = []
        visited = set()
        for dset in self.sub_dsets:
            name = dset.get_name() # Could use extended names here
            if not name in visited:
                visited.add(name)
                name_list.append(dset.field_names)
        return [f for fl in name_list for f in fl] # Flatten the names

    def _build_subset_dict(self):
        """ Maps fields to subsets of variables """
        subdset = self.sub_dsets[0]
        if subdset.get_name() in PDEBENCH_NAME_TO_OBJECT: # PDEBench dataset
            cur_max = 0
            subset_dict = {}
            for name, dset in PDEBENCH_NAME_TO_OBJECT.items():
                field_names = dset._specifics()[2]
                subset_dict[name] = list(range(cur_max, cur_max + len(field_names)))
                cur_max += len(field_names)
            return subset_dict
        else: # the-well dataset
            return {
                'euler_pbc': [0,1,2,3,4],                # density, energy, pressure, mom_x, mom_y
                'euler_obc': [0,1,2,3,4],                # density, energy, pressure, mom_x, mom_y
                'shear_flow': [5,2,6,7],                 # tracer, pressure, vel_x, vel_y
                'active_matter': [8,6,7,9,10,11,12,25,26,27,28], # concentration, vel_x, vel_y, tsr_xx, tsr_xy, tsr_yx, tsr_yy, tsr_xx, tsr_xy, tsr_yx, tsr_yy,
                'gray_scott': [13,14],                   # concentration_1, concentration_2
                'helmholtz': [15,16],                    # pressure_real, pressure_imag
                'rayleigh_benard': [17,2,6,7],           # buoyancy, pressure, vel_x, vel_y
                'rayleigh_taylor': [0,18,19,20],         # density, vel_x, vel_y, vel_z
                'supernova': [2,0,21,18,19,20],          # pressure, density, temperature, vel_x, vel_y, vel_z
                'mhd': [0,18,19,20,22,23,24],            # density, vel_x, vel_y, vel_z, mag_x, mag_y, mag_z
                'gravity_cooling': [2,0,21,18,19,20],    # pressure, density, temperature, vel_x, vel_y, vel_z
                'radiative_layer': [0,2,6,7],            # density, pressure, vel_x, vel_y
                'convective_envelope': [1,0,2,18,19,20], # energy, density, pressure, vel_x, vel_y, vel_z
                # finetuning datasets
                'euler_obc_finetune': [0,1,2,3,4],       # density, energy, pressure, mom_x, mom_y
            }
            
    def __getitem__(self, index):
        file_idx = np.searchsorted(self.offsets, index, side='right')-1 #which dataset are we on
        local_idx = index - max(self.offsets[file_idx], 0)
        try:
            dict = self.sub_dsets[file_idx][local_idx]
        except:
            print('FAILED AT ', file_idx, local_idx, index,int(os.environ.get("RANK", 0)))
            thisvariabledoesntexist
        # return x, file_idx, torch.tensor(self.subset_dict[self.sub_dsets[file_idx].get_name()]), bcs, y
        dataset = self.sub_dsets[file_idx]
        dict['field_labels'] = torch.tensor(self.subset_dict[dataset.get_name()])
        dict['file_index'] = file_idx
        return dict
    
    def __len__(self):
        return sum([len(dset) for dset in self.sub_dsets])


class RandomSamplerSeed(Sampler[int]):
    """ Overwrite the RandomSampler to allow for a seed for each epoch.
    Effectively going over the same data at same epochs. """

    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None, 
                 epoch: Optional[int] = None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.epoch = epoch

        if not isinstance(self.replacement, bool):
            raise TypeError(f"replacement should be a boolean value, but got replacement={self.replacement}")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            seed = int(torch.empty((), dtype=torch.int64).random_(generator=g).item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


T_co = TypeVar('T_co', covariant=True)


class MultisetSampler(Sampler[T_co]):
    """ Sampler that restricts data loading to a subset of the dataset. """

    def __init__(self, 
        dataset: MixedDataset, batch_size: int, 
        shuffle: bool = True, seed: int = 0,
        max_samples: int = 10, rank: int = 0, 
        distributed: bool = True,
    ):
        self.batch_size = batch_size
        self.sub_dsets = dataset.sub_dsets
        if distributed:
            sampler = DistributedSampler
        else:
            sampler = RandomSamplerSeed
        self.sub_samplers = [sampler(dataset) for dataset in self.sub_dsets]
        self.dataset = dataset
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples
        self.rank = rank

    def __iter__(self) -> Iterator[T_co]:
        samplers = [iter(sampler) for sampler in self.sub_samplers]
        sampler_choices = list(range(len(samplers)))
        generator = torch.Generator()
        generator.manual_seed(100*self.epoch+10*self.seed+self.rank)
        count = 0
        while len(sampler_choices) > 0:
            count += 1
            index_sampled = torch.randint(0, len(sampler_choices), size=(1,), generator=generator).item()
            dset_sampled = sampler_choices[index_sampled]
            offset = max(0, self.dataset.offsets[dset_sampled])
            # Do drop last batch type logic - if you can get a full batch, yield it, otherwise move to next dataset
            try:
                queue = []
                for _ in range(self.batch_size):
                    queue.append(next(samplers[dset_sampled]) + offset)
                if len(queue) == self.batch_size:
                    for d in queue:
                        yield d
            except Exception as err:
                print('ERRRR', err)
                sampler_choices.pop(index_sampled)
                print(f'Note: dset {dset_sampled} fully used. Dsets remaining: {len(sampler_choices)}')
                continue
            if count >= self.max_samples:
                break
    
    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        """ Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        for sampler in self.sub_samplers:
            sampler.set_epoch(epoch)
        self.epoch = epoch


def get_data_objects(
    dataset_names: List[str],
    batch_size: int,
    epoch_size: int,
    train_val_test: Tuple[float],
    n_past: int, 
    n_future: int,
    distributed: bool, 
    num_data_workers: int,
    rank: int, 
    split: str,
) -> Tuple:
    
    dataset = MixedDataset(
        dataset_names, n_past=n_past, n_future=n_future, 
        train_val_test=train_val_test, split=split,
        train_offset=0
    )

    sampler = MultisetSampler(  # default: shuffle = True 
        dataset, batch_size, distributed=distributed, 
        max_samples=epoch_size, rank=rank
    )

    dataloader = DataLoader(
        dataset, batch_size=int(batch_size), 
        num_workers=num_data_workers,
        shuffle=False, #(sampler is None), # shuffle determined by the sampler
        sampler=sampler, # Since validation is on a subset, use a fixed random subset,
        drop_last=True,
        pin_memory=torch.cuda.is_available()
    )

    return dataset, sampler, dataloader
