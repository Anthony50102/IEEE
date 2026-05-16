""" Main training script, inspired from: 
https://github.com/PolymathicAI/multiple_physics_pretraining/blob/main/train_basic.py """
import argparse
import os
import sys
sys.path.append("..")
import time
from pathlib import Path
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from collections import OrderedDict
import wandb
import gc
from torchinfo import summary

from src.models import build_model
from src.utils.data_utils import get_data_objects, DATASET_SPECS
from src.utils import is_debug, YParams, logging_utils, TimeTracker


def nrmse(y_true, y_pred, dims=(-1,)):
    """ Normalized root mean squared error. """
    residual = y_true - y_pred
    mse = residual.pow(2).mean(dims, keepdim=True)
    norm = 1e-7 + y_true.pow(2).mean(dims, keepdim=True)
    return (mse / norm).sqrt()


def add_weight_decay(params, weight_decay=1e-5, skip_list=()):
    """ From Ross Wightman at:
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3 
    
    Goes through the parameter list and if the squeeze dim is 1 or 0 (usually means bias or scale) 
    then don't apply weight decay. 
    """
    decay = []
    no_decay = []
    for name, param in params:
        if not param.requires_grad:
            continue
        if (len(param.squeeze().shape) <= 1 or name in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.,},
        {'params': decay, 'weight_decay': weight_decay}
    ]


def param_norm(parameters):
    with torch.no_grad():
        total_norm = 0
        for p in parameters:
            total_norm += p.pow(2).sum().item()
        return total_norm**.5


def grad_norm(parameters):
    with torch.no_grad():
        total_norm = 0
        for p in parameters:
            if p.grad is not None:
                total_norm += p.grad.pow(2).sum().item()
        return total_norm**.5


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)  # Convert to GB


def model_rollout(
    model: nn.Module,
    x: torch.Tensor,
    predict_normed: bool = False,
    n_future_steps: int = 1,
    state_labels: torch.Tensor | None = None,
    dset_name: str | None = None,
):
    """ x is B T C H W """
    # first iteration 
    out, metadata = model(x, predict_normed=predict_normed, state_labels=state_labels, dset_name=dset_name)
    if n_future_steps > 1:
        # more iterations: rollouts
        context = x.clone()
        outputs = [out]
        for _ in range(n_future_steps-1):
            context = torch.cat([context[:,1:,...], out], dim=1)
            out, _ = model(context, predict_normed=predict_normed, state_labels=state_labels, dset_name=dset_name)
            outputs.append(out)
        out = torch.cat(outputs, dim=1)
    return out, metadata


class Trainer:

    def __init__(self, params, global_rank, local_rank, device):
        self.device = device
        self.params = params
        self.base_dtype = torch.float
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.log_to_screen = params.log_to_screen

        # Basic setup
        self.start_epoch = 0
        self.epoch = 0
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.mp_type = torch.bfloat16
        else:
            self.mp_type = torch.half
        self.true_time = params.true_time

        self.iters = 0
        self.initialize_data(self.params)
        self.initialize_model(self.params)
        self.initialize_optimizer(self.params)
        if params.resuming:
            print("Loading checkpoint %s"%params.checkpoint_path)
            print('LOADING CHECKPOINTTTTTT')
            self.restore_checkpoint(params.checkpoint_path)  # no finetuning in that case
        if not params.resuming and params.pretrained:  # finetuning
            print("Starting from pretrained model at %s"%params.pretrained_ckpt_path)
            self.restore_checkpoint(params.pretrained_ckpt_path)
            self.iters = 0
            self.start_epoch = 0
        self.initialize_scheduler(self.params)

    def initialize_data(self, params):

        if len(set(DATASET_SPECS[dname]['group'] for dname in params.train_datasets + params.valid_datasets)) > 1:
            # this would require adapting the way the field labels are determined
            raise ValueError("Cannot mix datasets from PDEBench and the_well")
                             
        if self.log_to_screen:
            print(f"Initializing data on rank {self.global_rank}")
        self.train_dataset, self.train_sampler, self.train_data_loader = get_data_objects(
            params.train_datasets, params.batch_size, params.epoch_size, params.train_val_test,
            params.n_past, params.n_future, dist.is_initialized(), params.num_data_workers,
            rank=self.global_rank, split='train', 
        )
        self.valid_dataset, _, _ = get_data_objects(
            params.valid_datasets, params.batch_size, params.epoch_size, params.train_val_test,
            params.n_past, params.val_rollout, dist.is_initialized(), params.num_data_workers,
            rank=self.global_rank, split='valid', 
        )
        self.train_sampler.set_epoch(0)

    def initialize_model(self, params):

        print(f"Initializing model on rank {self.global_rank}")
        
        self.model = build_model(params).to(device=self.device)

        if dist.is_initialized():
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.local_rank], 
                output_device=[self.local_rank], find_unused_parameters=True,
            )

        n_params = sum([p.numel() for p in self.model.parameters()])
        
        self.single_print(f'Model parameter count: {n_params:,}')
        if self.params.model == "metaop" and ('finetune' not in self.params or not self.params.finetune):
            if dist.is_initialized():
                params_count = {k: opnn.weight_count() for k,opnn in self.model.module.opnn.items()}
            else:
                params_count = {k: opnn.weight_count() for k,opnn in self.model.opnn.items()}
            self.single_print(f"Operator class params: {params_count}")

    def initialize_optimizer(self, params): 

        parameters_standard = self.model.named_parameters()
        parameters = add_weight_decay(parameters_standard, self.params.weight_decay) # Dont use weight decay on bias/scaling terms
        if params.optimizer == 'adam':
            if self.params.learning_rate < 0:
                self.optimizer =  DAdaptAdam(parameters, lr=1., growth_rate=1.05, log_every=100, decouple=True)
            else:
                self.optimizer = optim.AdamW(parameters, lr=params.learning_rate)
        elif params.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=params.learning_rate, momentum=0.9)
        else: 
            raise ValueError(f"Optimizer {params.optimizer} not supported")
        self.gscaler = torch.amp.GradScaler(enabled=(self.mp_type == torch.half and params.enable_amp))

    def initialize_scheduler(self, params):

        self.scheduler = None

        if params.scheduler == 'cosine':
            
            last_step = (self.start_epoch * params.epoch_size) // params.accum_grad - 1
            warmup_steps = (params.warmup_epochs * params.epoch_size) // params.accum_grad
            total_steps = (params.max_epochs * params.epoch_size) // params.accum_grad

            if warmup_steps > 0 and params.learning_rate > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=.01, end_factor=1.0, total_iters=warmup_steps)
                decay = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=params.learning_rate/100, T_max=total_steps-warmup_steps)
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, [warmup, decay], [warmup_steps])
                for _ in range(last_step+1):  # sequentialLR: need to manually step through the last_step
                    self.scheduler.step()
            else:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=max(0,params.learning_rate/100), T_max=total_steps, last_epoch=last_step)

    def single_print(self, *text):
        if self.global_rank == 0 and self.log_to_screen:
            print(' '.join([str(t) for t in text]))

    def restore_checkpoint(self, checkpoint_path):
        """ Load model/opt from path """
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.local_rank))
        new_state_dict = OrderedDict()
        new_state_dict_place_holder = self.model.state_dict()
        pretrained = self.params.pretrained
        model_type = self.params.model
        for key, val in checkpoint['model_state'].items():
            current_name = new_name = key
            if not self.params.use_ddp:
                current_name = new_name = key[7:]  # remove the "module." prefix
            if pretrained:  
                # for pretarined models, some useless parameters must be removed and some must be kept fresh
                # remove: basically the weights related to 1d contexts (finetuning is done on 2d only)
                # keep fresh: the weights related to the fields-specific weights in the finetuning dataset
                if model_type == "disco":
                    filter_remove = ['burgers', 'diffre', 'swe', 'compNS', 'hpnn.encoder.1', 'proj_dim_variant_param.1']
                    filter_keep_fresh = ['space_bag.bias', 'space_bag.weight']
                if any([f in key for f in filter_remove]):
                    continue
                elif any([f in key for f in filter_keep_fresh]):
                    new_state_dict[new_name] = new_state_dict_place_holder[new_name]
                    continue
            if new_state_dict_place_holder[current_name].shape != val.shape:
                assert new_state_dict_place_holder[current_name].shape[:-1] == val.shape
                new_state_dict[new_name] = val.unsqueeze(-1)  # correct an incompatibility issue when the code was extended to 3d
            else:
                new_state_dict[new_name] = val
        if pretrained and model_type in ["disco"]:
            for key, val in new_state_dict_place_holder.items():
                if 'euler' in key:  # for fietuning, keep the fresh fields-specific weights
                    new_state_dict[key] = val
        self.model.load_state_dict(new_state_dict)
        self.iters = checkpoint['iters']
        if self.params.resuming:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.epoch = self.start_epoch
        checkpoint = None

    def save_checkpoint(self, checkpoint_path, model=None):
        """ Save model and optimizer to checkpoint """
        if not model:
            model = self.model
        d = {
            'iters': self.epoch*self.params.epoch_size, 
            'epoch': self.epoch, 
            'model_state': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(d, checkpoint_path)

    @staticmethod
    def mse_loss(y_ref, y):
        """ y and y_ref are B T C H (W) (D) """
        spatial_dims = tuple(range(3,y.ndim))
        var = 1e-7 + y_ref.var(spatial_dims, keepdim=True)
        loss = F.gaussian_nll_loss(y, y_ref, torch.ones_like(y)*var, eps=1e-8, reduction='mean')
        with torch.no_grad():
            residual = y - y_ref
            norm_ref = 1e-7 + y_ref.pow(2).mean(spatial_dims, keepdim=True)
            raw_loss = residual.pow(2.0).mean(spatial_dims, keepdims=True) / norm_ref
        return loss, raw_loss

    def train_one_epoch(self):
        tt = TimeTracker()
        tt.track("data", "training")
        self.epoch += 1
        self.model.train()
        logs = {
            'train_nrmse': torch.zeros(1).to(self.device, dtype=self.base_dtype),
            'train_l1': torch.zeros(1).to(self.device, dtype=self.base_dtype),
        }
        steps = 0
        grad_logs = {k: torch.zeros(1, device=self.device) for k in self.params.train_datasets}
        grad_counts = {k: torch.zeros(1, device=self.device) for k in self.params.train_datasets}
        theta_logs = {k: torch.zeros(1, device=self.device) for k in self.params.train_datasets}
        steps_logs = {k: torch.zeros(1, device=self.device) for k in self.params.train_datasets}
        loss_logs = {k: torch.zeros(1, device=self.device) for k in self.params.train_datasets}
        loss_counts = {k: torch.zeros(1, device=self.device) for k in self.params.train_datasets}
        dset_counts = {}
        self.single_print("--------")
        self.single_print('train_loader_size', len(self.train_data_loader), len(self.train_dataset))

        for batch_idx, batch in enumerate(self.train_data_loader):
            if batch_idx >= self.params.epoch_size:  # certain dataloaders are not restricted in length
                break
            steps += 1
            tt.track("data", "training", "data_batch")
            if self.true_time:
                torch.cuda.synchronize()

            inp, tar = batch['input_fields'].to(device=self.device), batch['output_fields'].to(device=self.device)
            dset_name = batch['name'][0]
            state_labels = batch['field_labels'].to(device=self.device)

            dset_counts[dset_name] = dset_counts.get(dset_name, 0) + 1
            loss_counts[dset_name] += 1

            # whether the model weights should be updated this batch
            self.model.require_backward_grad_sync = ((1+batch_idx) % self.params.accum_grad == 0)
            with torch.amp.autocast("cuda", enabled=self.params.enable_amp, dtype=self.mp_type):
                
                # forward
                tt.track("forward", "training", "forw_batch")
                output, metadata = self.model(
                    inp, predict_normed=False, 
                    # n_future_steps=self.params.n_future, 
                    state_labels=state_labels[0],
                    dset_name=dset_name,
                )
                tar = (tar - metadata['mean']) / metadata['std']  # normalize tar

                # loss
                tt.track("loss", "training", "loss_batch")
                loss, loss_raw = self.mse_loss(tar, output)
                loss = loss / self.params.accum_grad

                # logs
                tt.track("logs", "training")
                with torch.no_grad():
                    log_nrmse = loss_raw.sqrt().mean()
                    logs['train_nrmse'] += log_nrmse
                    loss_logs[dset_name] += loss.item()
                    loss_print = log_nrmse.item()
                    if 'theta' in metadata:
                        theta_logs[dset_name] += metadata['theta'].abs().mean()
                    if 'n_steps' in metadata:
                        steps_logs[dset_name] += metadata['n_steps']
                
                # backward
                tt.track("backward", "training", "back_batch")
                loss.backward()

                if self.true_time:
                    torch.cuda.synchronize()

                # gradient step
                tt.track("gradient_step", "training", "optim_batch")
                if self.model.require_backward_grad_sync: # Only take step once per accumulation cycle
                    grad_logs[dset_name] += grad_norm(self.model.parameters())
                    grad_counts[dset_name] += 1
                    # clip the gradients 
                    if self.params.gnorm is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.params.gnorm)
                    self.gscaler.step(self.optimizer)
                    self.gscaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    if self.true_time:
                        torch.cuda.synchronize()

                # logs
                if self.true_time:
                    torch.cuda.synchronize()
                if self.log_to_screen and batch_idx % self.params.log_interval == 0 and self.global_rank == 0:
                    current_mem_GB = torch.cuda.memory_allocated() / 1024**3
                    max_mem_GB = torch.cuda.max_memory_allocated() / 1024**3
                    reserved_mem_GB = torch.cuda.memory_reserved() / 1024**3
                    current_CPU_mem_GB = get_memory_usage()
                    print(f"Epoch {self.epoch} Batch {batch_idx} Train Loss {loss_print:.3f}")
                    print('Total Times. Batch: {}, Rank: {}, Data Shape: {}, Data time: {:.2f}, Forward: {:.2f}, Backward: {:.2f}, Optimizer: {:.2f}'.format(
                        batch_idx, self.global_rank, list(inp.shape), tt.get("data_batch"), tt.get("forw_batch"), tt.get("back_batch"), tt.get("optim_batch"))
                    )
                    print(f"Memory: CPU {current_CPU_mem_GB:5.2f} GB, Current {current_mem_GB:5.2f} GB, Max {max_mem_GB:5.2f} GB, Reserved {reserved_mem_GB:5.2f} GB")
                tt.reset("data_batch", "forw_batch", "back_batch", "loss_batch", "optim_batch")

        # logs
        logs = {k: v/steps for k, v in logs.items()}
        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key].detach()) # Mike: there was a bug with means when I originally implemented this - dont know if fixed
                logs[key] = float(logs[key]/dist.get_world_size())
            for key in sorted(loss_logs.keys()):
                dist.all_reduce(loss_logs[key].detach())
            for key in sorted(grad_logs.keys()):
                dist.all_reduce(grad_logs[key].detach())
            for key in sorted(theta_logs.keys()):
                dist.all_reduce(theta_logs[key].detach())
            for key in sorted(steps_logs.keys()):
                dist.all_reduce(steps_logs[key].detach())
            for key in sorted(loss_counts.keys()):
                dist.all_reduce(loss_counts[key].detach())
            for key in sorted(grad_counts.keys()):
                dist.all_reduce(grad_counts[key].detach())
        logs['learning_rate'] = self.optimizer.param_groups[0]['lr']

        times = tt.stop()
        for k in times:
            logs[f'time/{k}'] = times[k] / steps

        for key in loss_logs.keys():
            logs[f'{key}/train_nrmse'] = loss_logs[key] / loss_counts[key]
        for key in grad_logs.keys():
            logs[f'{key}/train_grad_norm'] = grad_logs[key] / grad_counts[key]
        for key in theta_logs.keys():
            logs[f'{key}/train_theta_norm'] = theta_logs[key] / loss_counts[key]
        for key in steps_logs.keys():
            logs[f'{key}/train_steps'] = steps_logs[key] / loss_counts[key]
        self.iters += steps
        if self.global_rank == 0:
            logs['iters'] = self.iters
            logs['parameter norm'] = param_norm(self.model.parameters())
        self.single_print('all reduces executed!')

        return times["training"], times["data"], logs

    def validate_one_epoch(self, cutoff):
        """
        Validates - for each batch just use a small subset to make it easier.

        Note: need to split datasets for meaningful metrics, but TBD. 
        """
        # Don't bother with full validation set between epochs
        self.model.eval()
        self.single_print('STARTING VALIDATION')
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=False, dtype=self.mp_type):
                sub_dsets = self.valid_dataset.sub_dsets
                logs = {}
                distinct_dsets = [dset.dataset_name for dset in sub_dsets]
                counts = {dset: 0 for dset in distinct_dsets}
                # iterate over the validation datasets
                for subset in sub_dsets:
                    dset_name = subset.dataset_name
                    # validation dataloser: technically shuffled but the shuffling order will always be the same
                    if self.params.use_ddp:
                        val_loader = torch.utils.data.DataLoader(
                            subset, batch_size=self.params.batch_size,
                            num_workers=self.params.num_data_workers,
                            sampler=torch.utils.data.distributed.DistributedSampler(subset, drop_last=True)
                        )
                    else:
                        val_loader = torch.utils.data.DataLoader(
                            subset, batch_size=self.params.batch_size,
                            num_workers=self.params.num_data_workers, 
                            shuffle=True, generator=torch.Generator().manual_seed(0), drop_last=True
                        )
                    count = 0
                    for batch_idx, batch in enumerate(val_loader):
                        # Only do a few batches of each dataset if not doing full validation
                        if count >= cutoff:  # validating on burgers equations is extremely long
                            del(val_loader)
                            break
                        count += 1
                        counts[dset_name] += 1

                        inp, tar = batch['input_fields'].to(self.device), batch['output_fields'].to(self.device)
                        n_past, n_future_val = self.params.n_past, self.params.val_rollout
                        if inp.shape[1] > n_past:
                            # indicates we need to split the input 
                            tar = torch.cat([inp[:,n_past:,...], tar], dim=1)
                            inp = inp[:,:n_past,...]

                        state_labels = torch.tensor(
                            self.train_dataset.subset_dict.get(subset.get_name(), [-1]*len(self.valid_dataset.subset_dict[subset.get_name()])),
                            device=self.device
                        ).unsqueeze(0)

                        # forward
                        output, _ = model_rollout(
                            self.model, inp, predict_normed=True, 
                            n_future_steps=min(self.params.val_rollout, tar.shape[1]),
                            state_labels=state_labels[0],
                            dset_name=dset_name,
                        )

                        # loss
                        residuals = output - tar
                        spatial_dims = tuple(range(residuals.ndim))[3:] # Assume 0, 1 are B, C
                        val_nrmse = nrmse(tar, output, dims=spatial_dims)
                        for step in [1,2,4,8]:
                            if step <= tar.shape[1]:
                                logs[f'{dset_name}/valid_nrmse_t{step}'] = logs.get(f'{dset_name}/valid_nrmse_t{step}',0) + val_nrmse[:,step-1,...].mean()

            self.single_print('DONE VALIDATING - NOW SYNCING')
            
            # divide by number of batches
            for k, v in logs.items():
                dset_name = k.split('/')[0]
                logs[k] = v / counts[dset_name]

            # # replace keys <>
            # average nrmse across datasets
            logs['valid_nrmse'] = 0
            for dset_name in distinct_dsets:
                logs['valid_nrmse'] += logs[f'{dset_name}/valid_nrmse_t1']/len(distinct_dsets)

            if dist.is_initialized():
                for key in sorted(logs.keys()):
                    dist.all_reduce(logs[key].detach()) # Mike: There was a bug with means when I implemented this - dont know if fixed
                    logs[key] = float(logs[key].item()/dist.get_world_size())
                    if 'rmse' in key:
                        logs[key] = logs[key]
            self.single_print('DONE SYNCING - NOW LOGGING')
        return logs

    def train(self):

        if self.params.log_to_wandb:
            wandb.init(
                dir=self.params.experiment_dir, config=self.params, 
                name=self.params.name, group=self.params.group,
                project=self.params.project, entity=self.params.entity, 
                resume=True
            )
        if self.global_rank == 0:
            summary(self.model)
        if self.params.log_to_wandb:
            wandb.watch(self.model, log="parameters")
        self.single_print("Starting Training Loop...")
    
        # Actually train now, saving checkpoints, logging time, and logging to wandb
        best_valid_loss = 1.e6
        for epoch in range(self.start_epoch, self.params.max_epochs):
            # if dist.is_initialized():
            if 'overfit' in self.params and self.params.overfit:
                self.train_sampler.set_epoch(0)
            else:
                self.train_sampler.set_epoch(epoch)
            start = time.time()

            tr_time, data_time, train_logs = self.train_one_epoch()
            
            valid_start = time.time()
            
            # decide whether to do a small/medium/complete validation
            val_cutoff = self.params.val_cutoff
            if epoch == self.params.max_epochs - 1:
                val_cutoff = 999
            if self.params.debug:
                val_cutoff = 1
            valid_logs = self.validate_one_epoch(val_cutoff)
            
            post_start = time.time()
            train_logs.update(valid_logs)
            train_logs['time/train_time'] = valid_start - start
            train_logs['time/train_data_time'] = data_time
            train_logs['time/train_compute_time'] = tr_time
            train_logs['time/valid_time'] = post_start - valid_start
            if self.params.log_to_wandb:
                wandb.log(train_logs) 

            if self.global_rank == 0:
                if self.params.save_checkpoint:
                    self.save_checkpoint(self.params.checkpoint_path)
                if epoch % self.params.checkpoint_save_interval == 0:
                    self.save_checkpoint(self.params.checkpoint_path + f'_epoch{epoch}')
                if valid_logs['valid_nrmse'] <= best_valid_loss:
                    self.save_checkpoint(self.params.best_checkpoint_path)
                    best_valid_loss = valid_logs['valid_nrmse']
                
                cur_time = time.time()
                self.single_print(f'Time for train {valid_start-start:.2f}. For valid: {post_start-valid_start:.2f}. For postprocessing:{cur_time-post_start:.2f}')
                self.single_print('Time taken for epoch {} is {:.2f} sec'.format(1+epoch, time.time()-start))
                self.single_print('Train loss: {}. Valid loss: {}'.format(train_logs['train_nrmse'], valid_logs['valid_nrmse']))

            # Clear references to large tensors after we're done using them
            train_logs = None
            valid_logs = None
            
            # More aggressive memory cleanup
            gc.collect()
            
            # Force CUDA to synchronize and clear any pending operations
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()


if __name__ == '__main__':

    print(f"DEBUG : {is_debug()}")

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ddp", action='store_true', help='Use distributed data parallel')
    parser.add_argument("--yaml_config", default='_debug.yaml', type=str)
    args = parser.parse_args()

    # Config 
    CONFIG_PATH = Path(__file__).resolve().parent / "config"
    params = YParams(CONFIG_PATH/"_base.yaml")
    refined_params = YParams(CONFIG_PATH/args.yaml_config)
    params.update_params(refined_params.params)
    if is_debug():
        debug_params = YParams(CONFIG_PATH/"_debug.yaml")
        params.update_params(debug_params.params)
    params['debug'] = is_debug()
    params['use_ddp'] = args.use_ddp

    # Set up distributed training
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.use_ddp:
        dist.init_process_group("nccl")  # backend for nvidia gpus, multi-node, multi-gpu
    torch.cuda.set_device(local_rank)
        
    device = torch.device(local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # Modify params
    params['batch_size'] = int(params.batch_size//world_size)
    params['start_epoch'] = 0
    exp_dir = Path(params.exp_dir) / params.run_name
    params['experiment_dir'] = str(exp_dir)
    params['checkpoint_path'] = str(exp_dir / 'training_checkpoints' / 'ckpt.tar')
    params['best_checkpoint_path'] = str(exp_dir / 'training_checkpoints' / 'best_ckpt.tar')

    # Have rank 0 check for and/or make directory
    if global_rank == 0:
        if not exp_dir.exists():
            exp_dir.mkdir(parents=True)
            (exp_dir / 'training_checkpoints').mkdir(parents=True)
    params['resuming'] = Path(params.checkpoint_path).is_file()

    # Wandb setup
    params['name'] = params.run_name
    if global_rank==0:
        logging_utils.log_to_file(logger_name=None, log_filename=exp_dir/'out.log')
        logging_utils.log_versions()
        params.log()

    params['log_to_wandb'] = (global_rank==0) and params['log_to_wandb']
    params['log_to_screen'] = (global_rank==0) and params['log_to_screen']

    if global_rank == 0:  # save config for this run
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = value
        with open(exp_dir/'hyperparams.yaml', 'w') as hpfile:
            yaml.dump(hparams, hpfile)

    # Start training
    trainer = Trainer(params, global_rank, local_rank, device)
    trainer.train()
    if params.log_to_screen:
        print('DONE ---- rank %d'%global_rank)
