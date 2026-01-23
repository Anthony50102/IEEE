"""
Step 2: FNO Autoregressive Rollout Training.

Trains the single-step FNO model (from Step 1) for multi-step autoregressive 
predictions using curriculum learning. Gradually increases rollout length
while using scheduled sampling to improve long-term stability.

Usage:
    python step_2_train.py --config config/fno_temporal_split.yaml --run-dir <run_dir>
    python step_2_train.py --config config/fno_temporal_split.yaml --run-dir <run_dir> --test

Author: Anthony Poole
"""

import argparse
import os
import sys
import time
import logging
from typing import List, Tuple

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import yaml

# Reuse functions from step_1
from step_1_train import (
    load_config, get_config_value, create_model, setup_logging,
    save_checkpoint, load_checkpoint, load_trajectory_slice,
)

# Parent directory for shared imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Curriculum Schedule
# =============================================================================

# Each stage: (rollout_length, teacher_forcing_ratio, n_epochs)
# - rollout_length: number of autoregressive steps
# - teacher_forcing_ratio: probability of using ground truth (vs model prediction)
# - n_epochs: training epochs at this stage
#
# NOTE: Teacher forcing starts HIGH (training wheels) and decreases
# so the model learns to recover from its own errors over time.

DEFAULT_CURRICULUM = [
    (5,  0.8, 20),   # Stage 1: Short rollouts, 80% teacher forcing (learn basics)
    (10, 0.6, 20),   # Stage 2: Medium rollouts, 60% teacher forcing
    (20, 0.4, 20),   # Stage 3: Longer rollouts, 40% teacher forcing
    (40, 0.2, 20),   # Stage 4: Long rollouts, 20% teacher forcing
    (80, 0.0, 30),   # Stage 5: Very long rollouts, no teacher forcing (fully autoregressive)
]


def parse_curriculum(cfg: dict) -> List[Tuple[int, float, int]]:
    """Parse curriculum from config or use default."""
    if 'curriculum' in cfg:
        return [tuple(stage) for stage in cfg['curriculum']]
    return DEFAULT_CURRICULUM


# =============================================================================
# Memory-Efficient Rollout Training
# =============================================================================

def train_rollout_epoch(
    model: nn.Module,
    states: np.ndarray,
    optimizer: torch.optim.Optimizer,
    device: str,
    rollout_length: int,
    teacher_forcing_ratio: float,
    batch_size: int = 8,
    grad_clip: float = 1.0,
    use_checkpointing: bool = True,
    noise_std: float = 0.01,
) -> float:
    """
    Train one epoch of rollout training.
    
    Memory optimization: Instead of storing full trajectories on GPU,
    we sample starting indices and process one rollout at a time.
    Uses gradient checkpointing to reduce memory at the cost of compute.
    
    Args:
        model: FNO model
        states: (T, C, H, W) numpy array of training states
        optimizer: PyTorch optimizer
        device: 'cuda' or 'cpu'
        rollout_length: Number of autoregressive steps
        teacher_forcing_ratio: Probability of using GT instead of prediction
        batch_size: Number of parallel rollouts
        grad_clip: Gradient clipping value
        use_checkpointing: If True, use gradient checkpointing to save memory
        noise_std: Std of Gaussian noise added to inputs (robustness)
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    T = len(states)
    max_start = T - rollout_length - 1
    
    if max_start <= 0:
        raise ValueError(f"Trajectory too short ({T}) for rollout length {rollout_length}")
    
    # Sample starting indices for this epoch
    n_samples = max(16, max_start // 2)  # Reasonable number of rollouts
    start_indices = np.random.choice(max_start, size=n_samples, replace=False)
    
    total_loss = 0.0
    n_batches = 0
    
    # Process in mini-batches
    for batch_start in range(0, len(start_indices), batch_size):
        batch_indices = start_indices[batch_start:batch_start + batch_size]
        batch_loss = 0.0
        
        optimizer.zero_grad()
        
        for start_idx in batch_indices:
            # Load initial state (memory efficient: load one at a time)
            x = torch.from_numpy(states[start_idx:start_idx+1]).float().to(device)
            
            # Add noise for robustness (helps model handle its own prediction errors)
            if noise_std > 0:
                x = x + noise_std * torch.randn_like(x)
            
            rollout_loss = 0.0
            
            for step in range(rollout_length):
                # Ground truth for this step
                target_idx = start_idx + step + 1
                y_true = torch.from_numpy(states[target_idx:target_idx+1]).float().to(device)
                
                # Forward pass with optional gradient checkpointing
                if use_checkpointing and x.requires_grad:
                    # Checkpoint requires input to have requires_grad=True
                    y_pred = checkpoint(model, x, use_reentrant=False)
                else:
                    # First step or when checkpointing disabled
                    x.requires_grad_(True)
                    if use_checkpointing:
                        y_pred = checkpoint(model, x, use_reentrant=False)
                    else:
                        y_pred = model(x)
                
                # Accumulate loss
                step_loss = nn.functional.mse_loss(y_pred, y_true)
                rollout_loss = rollout_loss + step_loss
                
                # Scheduled sampling: use GT or prediction for next step
                if np.random.random() < teacher_forcing_ratio:
                    x = y_true
                    # Add noise even to GT to improve robustness
                    if noise_std > 0:
                        x = x + noise_std * torch.randn_like(x)
                else:
                    x = y_pred.detach()
                
                # Clean up
                del y_true, y_pred
            
            # Average loss over rollout steps
            batch_loss = batch_loss + rollout_loss / rollout_length
            del x
        
        # Average over batch
        batch_loss = batch_loss / len(batch_indices)
        batch_loss.backward()
        
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += batch_loss.item()
        n_batches += 1
        
        del batch_loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate_rollout(
    model: nn.Module,
    states: np.ndarray,
    device: str,
    rollout_length: int,
    n_eval: int = 10,
) -> Tuple[float, np.ndarray]:
    """
    Evaluate autoregressive rollout MSE on test set.
    
    Args:
        model: FNO model
        states: (T, C, H, W) numpy array of test states
        device: 'cuda' or 'cpu'
        rollout_length: Number of autoregressive steps
        n_eval: Number of rollouts to evaluate
    
    Returns:
        average_mse: Mean MSE over all rollouts and steps
        step_mses: MSE at each step (for analysis)
    """
    model.eval()
    T = len(states)
    max_start = T - rollout_length - 1
    
    if max_start <= 0:
        return float('inf'), np.array([])
    
    # Sample evaluation starting points (evenly spaced for consistency)
    start_indices = np.linspace(0, max_start, min(n_eval, max_start), dtype=int)
    
    step_mses = np.zeros(rollout_length)
    n_samples = 0
    
    for start_idx in start_indices:
        x = torch.from_numpy(states[start_idx:start_idx+1]).float().to(device)
        
        for step in range(rollout_length):
            y_pred = model(x)
            target_idx = start_idx + step + 1
            y_true = torch.from_numpy(states[target_idx:target_idx+1]).float().to(device)
            
            mse = nn.functional.mse_loss(y_pred, y_true).item()
            step_mses[step] += mse
            
            x = y_pred
            del y_true, y_pred
        
        n_samples += 1
        del x
    
    step_mses /= n_samples
    avg_mse = step_mses.mean()
    
    return avg_mse, step_mses


# =============================================================================
# Visualization (for --test flag)
# =============================================================================

def generate_rollout_figures(
    model: nn.Module,
    test_states: np.ndarray,
    device: str,
    run_dir: str,
    rollout_length: int = 20,
    n_snapshots: int = 5,
):
    """
    Generate rollout comparison figures.
    
    Shows prediction vs ground truth at several points along a rollout.
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    T = len(test_states)
    
    # Start near beginning of test set
    start_idx = 0
    rollout_length = min(rollout_length, T - 1)
    
    fig_dir = os.path.join(run_dir, 'figures_rollout')
    os.makedirs(fig_dir, exist_ok=True)
    
    # Run full rollout
    predictions = []
    x = torch.from_numpy(test_states[start_idx:start_idx+1]).float().to(device)
    
    with torch.no_grad():
        for step in range(rollout_length):
            y_pred = model(x)
            predictions.append(y_pred[0].cpu().numpy())
            x = y_pred
    
    predictions = np.array(predictions)  # (steps, C, H, W)
    
    # Select snapshots to visualize (equally spaced)
    snapshot_indices = np.linspace(0, rollout_length - 1, n_snapshots, dtype=int)
    
    for i, step in enumerate(snapshot_indices):
        pred = predictions[step]                    # (C, H, W)
        truth = test_states[start_idx + step + 1]  # (C, H, W)
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        field_names = ['Density', 'Potential']
        
        for row, (name, p, t) in enumerate(zip(field_names, pred, truth)):
            error = p - t
            
            vmin, vmax = t.min(), t.max()
            im0 = axes[row, 0].imshow(p, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[row, 0].set_title(f'{name} - Prediction')
            axes[row, 0].axis('off')
            plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)
            
            im1 = axes[row, 1].imshow(t, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[row, 1].set_title(f'{name} - Ground Truth')
            axes[row, 1].axis('off')
            plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)
            
            err_max = max(abs(error.min()), abs(error.max()))
            im2 = axes[row, 2].imshow(error, cmap='RdBu_r', vmin=-err_max, vmax=err_max)
            axes[row, 2].set_title(f'{name} - Error (MSE={np.mean(error**2):.2e})')
            axes[row, 2].axis('off')
            plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)
        
        plt.suptitle(f'Rollout Step {step + 1}/{rollout_length}', fontsize=14)
        plt.tight_layout()
        
        fig_path = os.path.join(fig_dir, f'rollout_step{step+1:03d}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {fig_path}")
    
    # Also save error-over-time plot
    step_mses = np.mean((predictions - test_states[start_idx+1:start_idx+rollout_length+1])**2, axis=(1,2,3))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(range(1, rollout_length + 1), step_mses, 'o-', markersize=4)
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('MSE (log scale)')
    ax.set_title('Rollout Error Accumulation')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig_path = os.path.join(fig_dir, 'rollout_error_curve.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")
    
    print(f"\n  Generated rollout figures in {fig_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Step 2: FNO Rollout Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory from step 1")
    parser.add_argument("--test", action="store_true", help="Generate rollout figures after training")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    run_dir = args.run_dir
    
    # Check that step 1 completed
    status_path = os.path.join(run_dir, "status.txt")
    if os.path.exists(status_path):
        with open(status_path) as f:
            if "step1_completed" not in f.read():
                print("Warning: Step 1 may not have completed successfully")
    
    # Setup
    logger = setup_logging("step_2", run_dir, cfg.get('log_level', 'INFO'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Print header
    print("=" * 60)
    print("  STEP 2: FNO AUTOREGRESSIVE ROLLOUT TRAINING")
    print("=" * 60)
    print(f"  Run directory: {run_dir}")
    print(f"  Device: {device}")
    print(f"  Test mode: {args.test}")
    
    t_start = time.time()
    
    try:
        # Load data
        data_dir = get_config_value(cfg, 'paths', 'data_dir')
        train_file = get_config_value(cfg, 'paths', 'training_files')[0]
        file_path = os.path.join(data_dir, train_file)
        
        train_start = cfg.get('train_start', 0)
        train_end = cfg.get('train_end', 400)
        test_start = cfg.get('test_start', 400)
        test_end = cfg.get('test_end', 500)
        
        logger.info(f"Loading data from {file_path}")
        train_states = load_trajectory_slice(file_path, train_start, train_end)
        test_states = load_trajectory_slice(file_path, test_start, test_end)
        
        logger.info(f"  Train: {train_states.shape}")
        logger.info(f"  Test: {test_states.shape}")
        
        # Create model and load step 1 checkpoint
        model = create_model(cfg, device)
        checkpoint_path = os.path.join(run_dir, "checkpoint_best.pt")
        epoch_loaded, metrics = load_checkpoint(checkpoint_path, model, device=device)
        logger.info(f"Loaded checkpoint from epoch {epoch_loaded}")
        logger.info(f"  Step 1 metrics: {metrics}")
        
        # Training setup
        train_cfg = cfg.get('training', {})
        lr = train_cfg.get('learning_rate', 1e-3) * 0.1  # Lower LR for fine-tuning
        weight_decay = train_cfg.get('weight_decay', 1e-4)
        grad_clip = train_cfg.get('grad_clip', 1.0)
        batch_size = train_cfg.get('batch_size', 8)
        use_checkpointing = train_cfg.get('gradient_checkpointing', True)  # Default ON for memory
        noise_std = train_cfg.get('noise_std', 0.01)  # Input noise for robustness
        
        if use_checkpointing:
            logger.info("Gradient checkpointing ENABLED (saves memory, slower training)")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Parse curriculum
        curriculum = parse_curriculum(cfg)
        total_epochs = sum(stage[2] for stage in curriculum)
        
        print(f"\n  Curriculum ({len(curriculum)} stages, {total_epochs} total epochs):")
        for i, (length, tf_ratio, epochs) in enumerate(curriculum):
            print(f"    Stage {i+1}: rollout={length:3d}, teacher_forcing={tf_ratio:.1f}, epochs={epochs}")
        print()
        
        # Training loop through curriculum stages
        all_losses = []
        all_eval_results = []
        epoch_counter = 0
        
        for stage_idx, (rollout_length, tf_ratio, n_epochs) in enumerate(curriculum):
            print(f"\n  --- Stage {stage_idx + 1}/{len(curriculum)} ---")
            print(f"      Rollout: {rollout_length}, Teacher Forcing: {tf_ratio:.1f}")
            
            # Scheduler for this stage
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=lr * 0.01
            )
            
            for epoch in range(1, n_epochs + 1):
                t_epoch = time.time()
                epoch_counter += 1
                
                # Train
                train_loss = train_rollout_epoch(
                    model, train_states, optimizer, device,
                    rollout_length, tf_ratio, batch_size, grad_clip,
                    use_checkpointing=use_checkpointing,
                    noise_std=noise_std,
                )
                all_losses.append(train_loss)
                scheduler.step()
                
                # Evaluate periodically
                if epoch % 5 == 0 or epoch == n_epochs:
                    eval_mse, step_mses = evaluate_rollout(
                        model, test_states, device, rollout_length
                    )
                    all_eval_results.append({
                        'epoch': epoch_counter,
                        'stage': stage_idx + 1,
                        'rollout_length': rollout_length,
                        'mse': eval_mse,
                    })
                    
                    print(f"      Epoch {epoch:3d}/{n_epochs} | "
                          f"Train: {train_loss:.6f} | Eval: {eval_mse:.6f} | "
                          f"Time: {time.time()-t_epoch:.1f}s")
                else:
                    print(f"      Epoch {epoch:3d}/{n_epochs} | "
                          f"Train: {train_loss:.6f} | Time: {time.time()-t_epoch:.1f}s")
            
            # Save checkpoint after each stage
            save_checkpoint(
                model, optimizer, epoch_counter,
                os.path.join(run_dir, f"checkpoint_stage{stage_idx+1}.pt"),
                cfg, {'train_loss': train_loss, 'rollout_length': rollout_length}
            )
        
        # Save final checkpoint
        save_checkpoint(
            model, optimizer, epoch_counter,
            os.path.join(run_dir, "checkpoint_rollout_final.pt"),
            cfg, {'train_loss': all_losses[-1]}
        )
        
        # Final long-horizon evaluation
        final_rollout = curriculum[-1][0]  # Use longest rollout from curriculum
        final_mse, final_step_mses = evaluate_rollout(
            model, test_states, device, final_rollout, n_eval=20
        )
        
        # Save results
        np.savez(
            os.path.join(run_dir, "step2_results.npz"),
            train_losses=np.array(all_losses),
            eval_results=all_eval_results,
            final_mse=final_mse,
            final_step_mses=final_step_mses,
            curriculum=np.array(curriculum),
        )
        
        t_elapsed = time.time() - t_start
        
        print("\n" + "=" * 60)
        print("  STEP 2 COMPLETE")
        print("=" * 60)
        print(f"  Time: {t_elapsed:.1f}s ({t_elapsed/60:.1f} min)")
        print(f"  Final rollout MSE (length={final_rollout}): {final_mse:.6f}")
        
        # Generate test figures if requested
        if args.test:
            print("\n  Generating rollout figures...")
            generate_rollout_figures(
                model, test_states, device, run_dir,
                rollout_length=min(final_rollout, len(test_states) - 1),
                n_snapshots=5
            )
        
        print(f"\n  Results saved to: {run_dir}")
        
        # Update status
        with open(os.path.join(run_dir, "status.txt"), 'w') as f:
            f.write("step2_completed\n")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        with open(os.path.join(run_dir, "status.txt"), 'w') as f:
            f.write(f"step2_failed: {e}\n")
        raise


if __name__ == "__main__":
    main()
