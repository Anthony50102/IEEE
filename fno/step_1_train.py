"""
Step 1: FNO Single-Step Training.

Trains a Fourier Neural Operator (FNO) to predict the next state from the
current state. Uses temporal split: train on early snapshots, test on later.

Usage:
    python step_1_train.py --config config/fno_temporal_split.yaml
    python step_1_train.py --config config/fno_temporal_split.yaml --test

Author: Anthony Poole
"""

import argparse
import os
import sys
import time
import datetime
import logging
from typing import Tuple

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

# Neural operator library
from neuralop.models import FNO

# Parent directory for shared imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Configuration (simple dict-based for clarity)
# =============================================================================

def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_config_value(cfg: dict, *keys, default=None):
    """Safely get nested config value."""
    val = cfg
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val


# =============================================================================
# Memory-Efficient Dataset
# =============================================================================

class SingleStepDataset(Dataset):
    """
    Memory-efficient dataset for single-step predictions.
    
    Instead of storing inputs and targets separately (2x memory),
    we store states once and index into consecutive pairs.
    """
    
    def __init__(self, states: np.ndarray):
        """
        Args:
            states: (T, C, H, W) array - stored as reference, not copied
        """
        self.states = states  # Keep as numpy, convert on-the-fly
        self.n_pairs = len(states) - 1
    
    def __len__(self):
        return self.n_pairs
    
    def __getitem__(self, idx):
        # Convert to tensor on-the-fly to save GPU memory
        x = torch.from_numpy(self.states[idx].copy()).float()
        y = torch.from_numpy(self.states[idx + 1].copy()).float()
        return x, y


# =============================================================================
# Data Loading (Memory-Efficient)
# =============================================================================

def load_trajectory_slice(file_path: str, start: int, end: int) -> np.ndarray:
    """
    Load only a slice of the trajectory to save memory.
    
    Returns:
        states: (T, 2, H, W) array with density and potential
    """
    with h5py.File(file_path, 'r') as f:
        density = f['density'][start:end]      # (T, H, W)
        potential = f['phi'][start:end]        # (T, H, W)
    
    # Stack: (T, 2, H, W)
    return np.stack([density, potential], axis=1)


def create_data_loaders(cfg: dict, logger) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    """
    Create train and test data loaders.
    
    Returns:
        train_loader, test_loader, test_states (for visualization)
    """
    data_dir = get_config_value(cfg, 'paths', 'data_dir')
    train_file = get_config_value(cfg, 'paths', 'training_files')[0]
    file_path = os.path.join(data_dir, train_file)
    
    train_start = cfg.get('train_start', 0)
    train_end = cfg.get('train_end', 400)
    test_start = cfg.get('test_start', 400)
    test_end = cfg.get('test_end', 500)
    batch_size = get_config_value(cfg, 'training', 'batch_size', default=16)
    
    logger.info(f"Loading data from {file_path}")
    
    # Load train and test slices separately (memory efficient)
    train_states = load_trajectory_slice(file_path, train_start, train_end)
    test_states = load_trajectory_slice(file_path, test_start, test_end)
    
    logger.info(f"  Train: [{train_start}, {train_end}) -> {train_states.shape}")
    logger.info(f"  Test:  [{test_start}, {test_end}) -> {test_states.shape}")
    
    # Memory estimate
    mem_mb = (train_states.nbytes + test_states.nbytes) / 1e6
    logger.info(f"  Data memory: {mem_mb:.1f} MB")
    
    # Create data loaders
    train_loader = DataLoader(
        SingleStepDataset(train_states),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    test_loader = DataLoader(
        SingleStepDataset(test_states),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    return train_loader, test_loader, test_states


# =============================================================================
# Model Creation
# =============================================================================

def create_model(cfg: dict, device: str) -> nn.Module:
    """Create FNO model from config."""
    model_cfg = cfg.get('model', {})
    
    model = FNO(
        n_modes=tuple(model_cfg.get('n_modes', [32, 32])),
        in_channels=model_cfg.get('in_channels', 2),
        out_channels=model_cfg.get('out_channels', 2),
        hidden_channels=model_cfg.get('hidden_channels', 64),
        n_layers=model_cfg.get('n_layers', 4),
    )
    
    return model.to(device)


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        
        # Free memory
        del x, y, pred, loss
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate single-step MSE on test set."""
    model.eval()
    total_mse = 0.0
    n_samples = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        mse = nn.functional.mse_loss(pred, y, reduction='sum')
        total_mse += mse.item()
        n_samples += x.size(0)
        
        del x, y, pred
    
    return total_mse / n_samples


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(model, optimizer, epoch, path, cfg, metrics=None):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': cfg.get('model', {}),
        'metrics': metrics or {},
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, device='cpu'):
    """Load checkpoint. Returns epoch and metrics."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('metrics', {})


# =============================================================================
# Visualization (for --test flag)
# =============================================================================

def generate_test_figures(model, test_states, device, run_dir, n_figs=5):
    """
    Generate comparison figures: prediction vs ground truth + error.
    
    Saves n_figs equally-spaced snapshots from the test set.
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    T = len(test_states) - 1  # Number of (input, target) pairs
    
    # Select equally spaced indices
    indices = np.linspace(0, T - 1, n_figs, dtype=int)
    
    fig_dir = os.path.join(run_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    for i, idx in enumerate(indices):
        x = torch.from_numpy(test_states[idx:idx+1]).float().to(device)
        y_true = test_states[idx + 1]  # (C, H, W)
        
        with torch.no_grad():
            y_pred = model(x)[0].cpu().numpy()  # (C, H, W)
        
        # Create figure: 3 columns (pred, truth, error) x 2 rows (density, potential)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        field_names = ['Density', 'Potential']
        
        for row, (name, pred, truth) in enumerate(zip(field_names, y_pred, y_true)):
            error = pred - truth
            
            # Prediction
            vmin, vmax = truth.min(), truth.max()
            im0 = axes[row, 0].imshow(pred, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[row, 0].set_title(f'{name} - Prediction')
            axes[row, 0].axis('off')
            plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)
            
            # Ground truth
            im1 = axes[row, 1].imshow(truth, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[row, 1].set_title(f'{name} - Ground Truth')
            axes[row, 1].axis('off')
            plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)
            
            # Error
            err_max = max(abs(error.min()), abs(error.max()))
            im2 = axes[row, 2].imshow(error, cmap='RdBu_r', vmin=-err_max, vmax=err_max)
            axes[row, 2].set_title(f'{name} - Error (MSE={np.mean(error**2):.2e})')
            axes[row, 2].axis('off')
            plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)
        
        plt.suptitle(f'Single-Step Prediction: Test Index {idx}', fontsize=14)
        plt.tight_layout()
        
        fig_path = os.path.join(fig_dir, f'test_comparison_{i:02d}_idx{idx:04d}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {fig_path}")
    
    print(f"\n  Generated {n_figs} test figures in {fig_dir}")


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(name: str, run_dir: str, level: str = "INFO") -> logging.Logger:
    """Set up logging to file and console."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()  # Avoid duplicate handlers
    
    fh = logging.FileHandler(os.path.join(run_dir, f"{name}.log"))
    ch = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# =============================================================================
# Main
# =============================================================================

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Step 1: FNO Single-Step Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--run-dir", type=str, default=None, help="Existing run directory")
    parser.add_argument("--test", action="store_true", help="Generate test figures after training")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Create run directory
    if args.run_dir:
        run_dir = args.run_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = cfg.get('run_name', 'fno_experiment')
        output_base = get_config_value(cfg, 'paths', 'output_base', default='./output/')
        run_dir = os.path.join(output_base, f"{run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Setup
    logger = setup_logging("step_1", run_dir, cfg.get('log_level', 'INFO'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Print header
    print("=" * 60)
    print("  STEP 1: FNO SINGLE-STEP TRAINING")
    print("=" * 60)
    print(f"  Run directory: {run_dir}")
    print(f"  Device: {device}")
    print(f"  Test mode: {args.test}")
    
    # Save config copy
    with open(os.path.join(run_dir, "config.yaml"), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    t_start = time.time()
    
    try:
        # Load data
        train_loader, test_loader, test_states = create_data_loaders(cfg, logger)
        
        # Create model
        model = create_model(cfg, device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {n_params:,}")
        
        # Training setup
        train_cfg = cfg.get('training', {})
        n_epochs = train_cfg.get('n_epochs', 100)
        lr = train_cfg.get('learning_rate', 1e-3)
        weight_decay = train_cfg.get('weight_decay', 1e-4)
        grad_clip = train_cfg.get('grad_clip', 1.0)
        eval_every = train_cfg.get('eval_every', 10)
        save_every = train_cfg.get('save_every', 20)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01
        )
        
        # Training loop
        logger.info("Starting training...")
        print(f"\n  Training for {n_epochs} epochs:\n")
        
        best_mse = float('inf')
        train_losses = []
        test_mses = []
        
        for epoch in range(1, n_epochs + 1):
            t_epoch = time.time()
            
            # Train
            train_loss = train_one_epoch(model, train_loader, optimizer, device, grad_clip)
            train_losses.append(train_loss)
            scheduler.step()
            
            # Evaluate periodically
            if epoch % eval_every == 0 or epoch == n_epochs:
                test_mse = evaluate(model, test_loader, device)
                test_mses.append((epoch, test_mse))
                
                # Save best model
                if test_mse < best_mse:
                    best_mse = test_mse
                    save_checkpoint(model, optimizer, epoch, 
                                    os.path.join(run_dir, "checkpoint_best.pt"),
                                    cfg, {'train_loss': train_loss, 'test_mse': test_mse})
                
                print(f"  Epoch {epoch:4d}/{n_epochs} | "
                      f"Train: {train_loss:.6f} | Test: {test_mse:.6f} | "
                      f"Time: {time.time()-t_epoch:.1f}s")
            else:
                print(f"  Epoch {epoch:4d}/{n_epochs} | "
                      f"Train: {train_loss:.6f} | Time: {time.time()-t_epoch:.1f}s")
            
            # Periodic checkpoint
            if epoch % save_every == 0:
                save_checkpoint(model, optimizer, epoch,
                                os.path.join(run_dir, f"checkpoint_epoch{epoch:04d}.pt"),
                                cfg, {'train_loss': train_loss})
        
        # Save final checkpoint
        save_checkpoint(model, optimizer, n_epochs,
                        os.path.join(run_dir, "checkpoint_latest.pt"),
                        cfg, {'train_loss': train_losses[-1], 'test_mse': best_mse})
        
        # Final evaluation
        final_mse = evaluate(model, test_loader, device)
        
        # Save results
        np.savez(os.path.join(run_dir, "step1_results.npz"),
                 train_losses=np.array(train_losses),
                 test_mses=np.array(test_mses),
                 final_mse=final_mse, best_mse=best_mse)
        
        t_elapsed = time.time() - t_start
        
        print("\n" + "=" * 60)
        print("  STEP 1 COMPLETE")
        print("=" * 60)
        print(f"  Time: {t_elapsed:.1f}s ({t_elapsed/60:.1f} min)")
        print(f"  Final train loss: {train_losses[-1]:.6f}")
        print(f"  Final test MSE: {final_mse:.6f}")
        print(f"  Best test MSE: {best_mse:.6f}")
        
        # Generate test figures if requested
        if args.test:
            print("\n  Generating test figures...")
            generate_test_figures(model, test_states, device, run_dir, n_figs=5)
        
        print(f"\n  Results saved to: {run_dir}")
        print(f"\n  Next: python step_2_train.py --config {args.config} --run-dir {run_dir}")
        
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
