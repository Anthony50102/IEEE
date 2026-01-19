"""
Step 1: FNO Training for Single-Step Forward Predictions.

This script trains a Fourier Neural Operator (FNO) model to predict
the next state from the current state (single-step forward prediction).

Supports temporal split mode:
- Train on first n snapshots of a trajectory
- Test on remaining snapshots (single-step only)

Usage:
    python step_1_train.py --config config/fno_temporal_split.yaml

Author: Anthony Poole
"""

import argparse
import os
import sys
import time
import logging
import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

# Add parent directory for shared imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FNOConfig:
    """Configuration for FNO training."""
    run_name: str = "fno_experiment"
    
    # Paths
    output_base: str = "./output/"
    data_dir: str = ""
    training_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    
    # Physics
    dt: float = 0.025
    n_fields: int = 2
    n_x: int = 256
    n_y: int = 256
    
    # Training mode
    training_mode: str = "temporal_split"
    train_start: int = 0
    train_end: int = 400
    test_start: int = 400
    test_end: int = 500
    
    # Model
    n_modes: Tuple[int, int] = (32, 32)
    in_channels: int = 2
    out_channels: int = 2
    hidden_channels: int = 64
    n_layers: int = 4
    
    # Training
    batch_size: int = 16
    n_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    scheduler_type: str = "cosine"
    warmup_epochs: int = 5
    save_every: int = 20
    eval_every: int = 10
    
    # Logging
    log_level: str = "INFO"
    run_dir: str = ""


def load_config(config_path: str) -> FNOConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    cfg = FNOConfig()
    cfg.run_name = cfg_dict.get('run_name', cfg.run_name)
    
    # Paths
    paths = cfg_dict.get('paths', {})
    cfg.output_base = paths.get('output_base', cfg.output_base)
    cfg.data_dir = paths.get('data_dir', cfg.data_dir)
    cfg.training_files = paths.get('training_files', cfg.training_files)
    cfg.test_files = paths.get('test_files', cfg.test_files)
    
    # Physics
    physics = cfg_dict.get('physics', {})
    cfg.dt = physics.get('dt', cfg.dt)
    cfg.n_fields = physics.get('n_fields', cfg.n_fields)
    cfg.n_x = physics.get('n_x', cfg.n_x)
    cfg.n_y = physics.get('n_y', cfg.n_y)
    
    # Training mode
    cfg.training_mode = cfg_dict.get('training_mode', cfg.training_mode)
    cfg.train_start = cfg_dict.get('train_start', cfg.train_start)
    cfg.train_end = cfg_dict.get('train_end', cfg.train_end)
    cfg.test_start = cfg_dict.get('test_start', cfg.test_start)
    cfg.test_end = cfg_dict.get('test_end', cfg.test_end)
    
    # Model
    model_cfg = cfg_dict.get('model', {})
    n_modes = model_cfg.get('n_modes', list(cfg.n_modes))
    cfg.n_modes = tuple(n_modes) if isinstance(n_modes, list) else n_modes
    cfg.in_channels = model_cfg.get('in_channels', cfg.in_channels)
    cfg.out_channels = model_cfg.get('out_channels', cfg.out_channels)
    cfg.hidden_channels = model_cfg.get('hidden_channels', cfg.hidden_channels)
    cfg.n_layers = model_cfg.get('n_layers', cfg.n_layers)
    
    # Training
    train_cfg = cfg_dict.get('training', {})
    cfg.batch_size = train_cfg.get('batch_size', cfg.batch_size)
    cfg.n_epochs = train_cfg.get('n_epochs', cfg.n_epochs)
    cfg.learning_rate = train_cfg.get('learning_rate', cfg.learning_rate)
    cfg.weight_decay = train_cfg.get('weight_decay', cfg.weight_decay)
    cfg.grad_clip = train_cfg.get('grad_clip', cfg.grad_clip)
    cfg.save_every = train_cfg.get('save_every', cfg.save_every)
    cfg.eval_every = train_cfg.get('eval_every', cfg.eval_every)
    
    scheduler = train_cfg.get('scheduler', {})
    cfg.scheduler_type = scheduler.get('type', cfg.scheduler_type)
    cfg.warmup_epochs = scheduler.get('warmup_epochs', cfg.warmup_epochs)
    
    cfg.log_level = cfg_dict.get('log_level', cfg.log_level)
    
    return cfg


# =============================================================================
# Dataset
# =============================================================================

class SingleStepDataset(Dataset):
    """Dataset for single-step (input, target) pairs."""
    
    def __init__(self, states: np.ndarray):
        """
        Args:
            states: (T, C, H, W) array of state snapshots
        """
        self.inputs = torch.from_numpy(states[:-1]).float()   # (T-1, C, H, W)
        self.targets = torch.from_numpy(states[1:]).float()   # (T-1, C, H, W)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# =============================================================================
# FNO Model
# =============================================================================

class SpectralConv2d(nn.Module):
    """2D Fourier layer for FNO."""
    
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
    
    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x):
        batchsize = x.shape[0]
        
        # Compute FFT
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        
        # Return to physical space
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
    """2D Fourier Neural Operator."""
    
    def __init__(self, modes1, modes2, width, in_channels=2, out_channels=2, n_layers=4):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        
        self.fc0 = nn.Linear(in_channels, width)
        
        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(SpectralConv2d(width, width, modes1, modes2))
            self.ws.append(nn.Conv2d(width, width, 1))
        
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
    
    def forward(self, x):
        # x: (batch, channels, height, width)
        x = x.permute(0, 2, 3, 1)  # (batch, height, width, channels)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (batch, width, height, width)
        
        for conv, w in zip(self.convs, self.ws):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            x = torch.nn.functional.gelu(x)
        
        x = x.permute(0, 2, 3, 1)  # (batch, height, width, width)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)  # (batch, out_channels, height, width)
        
        return x


def create_model(cfg: FNOConfig, device: str) -> nn.Module:
    """Create FNO model based on configuration."""
    model = FNO2d(
        modes1=cfg.n_modes[0],
        modes2=cfg.n_modes[1],
        width=cfg.hidden_channels,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        n_layers=cfg.n_layers,
    )
    return model.to(device)


# =============================================================================
# Data Loading
# =============================================================================

def load_trajectory(file_path: str, cfg: FNOConfig) -> np.ndarray:
    """
    Load trajectory data from HDF5 file.
    
    Returns:
        states: (T, C, H, W) array where C=2 (density, potential)
    """
    import h5py
    
    with h5py.File(file_path, 'r') as f:
        density = f['density'][:]      # (T, H, W)
        potential = f['phi'][:]        # (T, H, W)
    
    # Stack channels: (T, 2, H, W)
    states = np.stack([density, potential], axis=1)
    
    return states


def prepare_data(cfg: FNOConfig, logger: logging.Logger):
    """
    Prepare training and test data based on configuration.
    
    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for single-step test evaluation
        test_states: (T_test, C, H, W) numpy array for reference
    """
    logger.info("Loading data...")
    
    # Load full trajectory
    file_path = os.path.join(cfg.data_dir, cfg.training_files[0])
    states = load_trajectory(file_path, cfg)
    
    n_time, n_channels, n_y, n_x = states.shape
    logger.info(f"  Loaded trajectory: {states.shape}")
    logger.info(f"  Data range - density: [{states[:, 0].min():.4f}, {states[:, 0].max():.4f}]")
    logger.info(f"  Data range - potential: [{states[:, 1].min():.4f}, {states[:, 1].max():.4f}]")
    
    # Temporal split
    if cfg.training_mode == "temporal_split":
        train_states = states[cfg.train_start:cfg.train_end]
        test_states = states[cfg.test_start:cfg.test_end]
        
        logger.info(f"  Training: snapshots [{cfg.train_start}, {cfg.train_end}) = {len(train_states)}")
        logger.info(f"  Test: snapshots [{cfg.test_start}, {cfg.test_end}) = {len(test_states)}")
    else:
        raise ValueError(f"Unknown training mode: {cfg.training_mode}")
    
    # Create datasets and loaders
    train_dataset = SingleStepDataset(train_states)
    test_dataset = SingleStepDataset(test_states)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    logger.info(f"  Training pairs: {len(train_dataset)}")
    logger.info(f"  Test pairs: {len(test_dataset)}")
    
    return train_loader, test_loader, test_states


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, device, grad_clip=1.0):
    """Train for one epoch on single-step predictions."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = nn.functional.mse_loss(predictions, targets)
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate_single_step(model, test_loader, device):
    """
    Evaluate model on single-step predictions only.
    
    Args:
        model: Trained FNO model
        test_loader: DataLoader with (input, target) pairs
        device: torch device
    
    Returns:
        avg_mse: Average MSE over all test pairs
    """
    model.eval()
    total_mse = 0.0
    n_samples = 0
    
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        predictions = model(inputs)
        mse = nn.functional.mse_loss(predictions, targets, reduction='sum')
        
        total_mse += mse.item()
        n_samples += inputs.size(0)
    
    return total_mse / n_samples


# =============================================================================
# Utilities
# =============================================================================

def get_run_directory(cfg: FNOConfig, run_dir: Optional[str] = None) -> str:
    """Create or get run directory."""
    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.output_base, f"{cfg.run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def setup_logging(name: str, run_dir: str, level: str = "INFO") -> logging.Logger:
    """Set up logging to file and console."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # File handler
    fh = logging.FileHandler(os.path.join(run_dir, f"{name}.log"))
    fh.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper()))
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def save_checkpoint(model, optimizer, epoch, run_dir, cfg, metrics=None):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'n_modes': cfg.n_modes,
            'hidden_channels': cfg.hidden_channels,
            'n_layers': cfg.n_layers,
            'in_channels': cfg.in_channels,
            'out_channels': cfg.out_channels,
        },
    }
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    path = os.path.join(run_dir, f"checkpoint_epoch{epoch:04d}.pt")
    torch.save(checkpoint, path)
    
    # Also save as latest
    latest_path = os.path.join(run_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)
    
    return path


def print_header(title: str):
    """Print formatted header."""
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_config_summary(cfg: FNOConfig):
    """Print configuration summary."""
    print(f"\n  Configuration:")
    print(f"    Data: {cfg.data_dir}")
    print(f"    Training file: {cfg.training_files[0]}")
    print(f"    Grid: {cfg.n_y} x {cfg.n_x}")
    print(f"    Train range: [{cfg.train_start}, {cfg.train_end})")
    print(f"    Test range: [{cfg.test_start}, {cfg.test_end})")
    print(f"\n  Model:")
    print(f"    Fourier modes: {cfg.n_modes}")
    print(f"    Hidden channels: {cfg.hidden_channels}")
    print(f"    Layers: {cfg.n_layers}")
    print(f"\n  Training:")
    print(f"    Epochs: {cfg.n_epochs}")
    print(f"    Batch size: {cfg.batch_size}")
    print(f"    Learning rate: {cfg.learning_rate}")
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for FNO single-step training."""
    parser = argparse.ArgumentParser(description="Step 1: FNO Single-Step Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--run-dir", type=str, default=None, help="Existing run directory")
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Get/create run directory
    run_dir = get_run_directory(cfg, args.run_dir)
    cfg.run_dir = run_dir
    
    # Set up logging
    logger = setup_logging("step_1", run_dir, cfg.log_level)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print_header("STEP 1: FNO TRAINING (Single-Step Forward Prediction)")
    print(f"  Run directory: {run_dir}")
    print(f"  Device: {device}")
    print_config_summary(cfg)
    
    # Save config
    with open(os.path.join(run_dir, "config.yaml"), 'w') as f:
        yaml.dump({
            'run_name': cfg.run_name,
            'training_mode': cfg.training_mode,
            'train_range': [cfg.train_start, cfg.train_end],
            'test_range': [cfg.test_start, cfg.test_end],
            'model': {
                'n_modes': list(cfg.n_modes),
                'hidden_channels': cfg.hidden_channels,
                'n_layers': cfg.n_layers,
            },
            'training': {
                'n_epochs': cfg.n_epochs,
                'batch_size': cfg.batch_size,
                'learning_rate': cfg.learning_rate,
            },
        }, f, default_flow_style=False)
    
    t_start = time.time()
    
    try:
        # Load data
        train_loader, test_loader, test_states = prepare_data(cfg, logger)
        
        # Create model
        logger.info("Creating model...")
        model = create_model(cfg, device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model parameters: {n_params:,}")
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        
        # Scheduler
        if cfg.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.n_epochs, eta_min=cfg.learning_rate * 0.01
            )
        else:
            scheduler = None
        
        # Training loop
        logger.info("Starting training...")
        print("\n  Training Progress:")
        
        best_test_mse = float('inf')
        train_losses = []
        test_mses = []
        
        for epoch in range(1, cfg.n_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, device, cfg.grad_clip)
            train_losses.append(train_loss)
            
            if scheduler is not None:
                scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Evaluate periodically
            if epoch % cfg.eval_every == 0 or epoch == cfg.n_epochs:
                test_mse = evaluate_single_step(model, test_loader, device)
                test_mses.append((epoch, test_mse))
                
                if test_mse < best_test_mse:
                    best_test_mse = test_mse
                    save_checkpoint(model, optimizer, epoch, run_dir, cfg, 
                                    {'train_loss': train_loss, 'test_mse': test_mse})
                    logger.info(f"  New best model saved at epoch {epoch}")
                
                print(f"    Epoch {epoch:4d}/{cfg.n_epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Test MSE: {test_mse:.6f} | "
                      f"Time: {epoch_time:.1f}s")
            else:
                print(f"    Epoch {epoch:4d}/{cfg.n_epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Time: {epoch_time:.1f}s")
            
            # Periodic checkpoints
            if epoch % cfg.save_every == 0:
                save_checkpoint(model, optimizer, epoch, run_dir, cfg,
                                {'train_loss': train_loss})
        
        # Final evaluation
        logger.info("Final evaluation...")
        final_mse = evaluate_single_step(model, test_loader, device)
        
        # Save results
        results_path = os.path.join(run_dir, "step1_results.npz")
        np.savez(
            results_path,
            train_losses=np.array(train_losses),
            test_mses=np.array(test_mses),
            final_test_mse=final_mse,
        )
        
        t_elapsed = time.time() - t_start
        
        print("\n" + "=" * 60)
        print("  STEP 1 TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Total time: {t_elapsed:.1f}s ({t_elapsed/60:.1f} min)")
        print(f"  Final train loss: {train_losses[-1]:.6f}")
        print(f"  Final test MSE (single-step): {final_mse:.6f}")
        print(f"  Best test MSE (single-step): {best_test_mse:.6f}")
        print(f"  Results saved to: {results_path}")
        print(f"\n  Next: Run step_2_train.py for autoregressive rollout training")
        print()
        
        logger.info(f"Step 1 completed in {t_elapsed:.1f}s")
        logger.info(f"Final test MSE (single-step): {final_mse:.6f}")
        
        # Save status
        with open(os.path.join(run_dir, "status.txt"), 'w') as f:
            f.write("step1_completed\n")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        with open(os.path.join(run_dir, "status.txt"), 'w') as f:
            f.write(f"failed: {e}\n")
        raise


if __name__ == "__main__":
    main()
