"""
Step 2: FNO Autoregressive Rollout Training.

This script fine-tunes a pre-trained FNO model (from Step 1) to perform
autoregressive rollouts. Uses curriculum learning with increasing rollout
lengths and scheduled sampling.

Usage:
    python step_2_train.py --config config/fno_temporal_split.yaml --run-dir <step1_run_dir>

Author: Anthony Poole
"""

import argparse
import os
import sys
import time
import logging
import datetime
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

# Add parent directory for shared imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from step_1
from step_1_train import (
    FNOConfig, load_config, create_model, load_trajectory,
    setup_logging, print_header, save_checkpoint,
)


# =============================================================================
# Dataset for Rollout Training
# =============================================================================

class TrajectoryDataset(Dataset):
    """Dataset that returns full trajectories for rollout training."""
    
    def __init__(self, states: np.ndarray):
        """
        Args:
            states: (T, C, H, W) array of state snapshots
        """
        self.states = torch.from_numpy(states).float()
    
    def __len__(self):
        return 1  # Single trajectory
    
    def __getitem__(self, idx):
        return self.states


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch_rollout(model, states, optimizer, device, rollout_len, 
                        ss_prob=0.0, grad_clip=1.0, batch_size=4):
    """
    Train one epoch with autoregressive rollout.
    
    Args:
        model: FNO model
        states: (T, C, H, W) tensor on device
        optimizer: Optimizer
        device: torch device
        rollout_len: Number of autoregressive steps
        ss_prob: Scheduled sampling probability (use prediction vs ground truth)
        grad_clip: Gradient clipping norm
        batch_size: Number of rollouts per batch
    
    Returns:
        avg_loss: Average loss over all rollouts
    """
    model.train()
    T, C, H, W = states.shape
    
    max_start = T - rollout_len - 1
    if max_start < 1:
        return 0.0
    
    # Sample random starting points
    n_rollouts = max(1, max_start // 2)  # Don't do too many rollouts per epoch
    start_indices = random.sample(range(max_start), min(n_rollouts, max_start))
    
    total_loss = 0.0
    n_batches = 0
    
    for i in range(0, len(start_indices), batch_size):
        batch_starts = start_indices[i:i+batch_size]
        batch_size_actual = len(batch_starts)
        
        optimizer.zero_grad()
        batch_loss = 0.0
        
        for start in batch_starts:
            # Initialize with ground truth
            state = states[start:start+1].clone()  # (1, C, H, W)
            
            rollout_loss = 0.0
            for step in range(rollout_len):
                # Predict next state
                pred = model(state)
                target = states[start + step + 1:start + step + 2]
                
                rollout_loss += nn.functional.mse_loss(pred, target)
                
                # Scheduled sampling: use prediction or ground truth for next step
                if random.random() < ss_prob:
                    state = pred  # Use prediction (with gradient)
                else:
                    state = target.clone()  # Use ground truth
            
            batch_loss += rollout_loss / rollout_len
        
        batch_loss = batch_loss / batch_size_actual
        batch_loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += batch_loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_rollout(model, states, device, rollout_len=None):
    """
    Evaluate model via full autoregressive rollout.
    
    Args:
        model: Trained FNO model
        states: (T, C, H, W) numpy array - ground truth
        device: torch device
        rollout_len: Number of steps to predict (None = full trajectory)
    
    Returns:
        mse_per_step: MSE at each timestep
        avg_mse: Average MSE over all steps
        predictions: (T, C, H, W) numpy array
    """
    model.eval()
    T, C, H, W = states.shape
    
    if rollout_len is None:
        rollout_len = T - 1
    else:
        rollout_len = min(rollout_len, T - 1)
    
    # Start from first state
    state = torch.from_numpy(states[0:1]).float().to(device)
    
    predictions = [states[0]]  # Start with IC
    
    for t in range(rollout_len):
        state = model(state)
        predictions.append(state[0].cpu().numpy())
    
    predictions = np.array(predictions)  # (rollout_len+1, C, H, W)
    ground_truth = states[:rollout_len + 1]
    
    # Compute MSE per timestep
    mse_per_step = np.mean((predictions - ground_truth) ** 2, axis=(1, 2, 3))
    avg_mse = np.mean(mse_per_step)
    
    return mse_per_step, avg_mse, predictions


# =============================================================================
# Utilities
# =============================================================================

def load_step1_checkpoint(run_dir: str, model: nn.Module, device: str):
    """Load the best checkpoint from Step 1."""
    checkpoint_path = os.path.join(run_dir, "checkpoint_latest.pt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def print_rollout_config(schedule):
    """Print rollout training schedule."""
    print("\n  Rollout Training Schedule:")
    print("    " + "-" * 50)
    print(f"    {'Rollout Len':>12} | {'SS Prob':>8} | {'Epochs':>8}")
    print("    " + "-" * 50)
    for rollout_len, ss_prob, n_epochs in schedule:
        print(f"    {rollout_len:>12} | {ss_prob:>8.2f} | {n_epochs:>8}")
    print("    " + "-" * 50)
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for FNO rollout training."""
    parser = argparse.ArgumentParser(description="Step 2: FNO Rollout Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--run-dir", type=str, required=True, help="Step 1 run directory")
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    run_dir = args.run_dir
    cfg.run_dir = run_dir
    
    # Set up logging
    logger = setup_logging("step_2", run_dir, cfg.log_level)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print_header("STEP 2: FNO TRAINING (Autoregressive Rollout)")
    print(f"  Run directory: {run_dir}")
    print(f"  Device: {device}")
    
    # Define rollout curriculum schedule
    # (rollout_length, scheduled_sampling_prob, n_epochs)
    rollout_schedule = [
        (5, 0.0, 20),      # Short rollouts, teacher forcing
        (10, 0.2, 20),     # Medium rollouts, some scheduled sampling
        (20, 0.4, 20),     # Longer rollouts
        (40, 0.6, 20),     # Even longer
        (80, 0.8, 30),     # Long rollouts, mostly use predictions
    ]
    
    print_rollout_config(rollout_schedule)
    
    t_start = time.time()
    
    try:
        # Load data
        logger.info("Loading data...")
        file_path = os.path.join(cfg.data_dir, cfg.training_files[0])
        states = load_trajectory(file_path, cfg)
        
        # Use training portion for rollout training
        train_states = states[cfg.train_start:cfg.train_end]
        test_states = states[cfg.test_start:cfg.test_end]
        
        train_states_tensor = torch.from_numpy(train_states).float().to(device)
        
        logger.info(f"  Train states: {train_states.shape}")
        logger.info(f"  Test states: {test_states.shape}")
        
        # Create and load model
        logger.info("Loading Step 1 model...")
        model = create_model(cfg, device)
        step1_epoch, step1_metrics = load_step1_checkpoint(run_dir, model, device)
        logger.info(f"  Loaded checkpoint from epoch {step1_epoch}")
        if step1_metrics:
            logger.info(f"  Step 1 metrics: {step1_metrics}")
        
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model parameters: {n_params:,}")
        
        # Optimizer with lower learning rate for fine-tuning
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate * 0.1,  # Lower LR for fine-tuning
            weight_decay=cfg.weight_decay,
        )
        
        # Evaluate before training
        logger.info("Evaluating before rollout training...")
        _, initial_mse, _ = evaluate_rollout(model, test_states, device)
        print(f"\n  Initial rollout MSE (before Step 2): {initial_mse:.6f}")
        
        # Training loop with curriculum
        print("\n  Rollout Training Progress:")
        
        train_losses = []
        eval_mses = []
        total_epochs = 0
        
        for rollout_len, ss_prob, n_epochs in rollout_schedule:
            print(f"\n  --- Rollout: {rollout_len}, Scheduled Sampling: {ss_prob:.2f} ---")
            
            for epoch in range(1, n_epochs + 1):
                total_epochs += 1
                epoch_start = time.time()
                
                # Train
                train_loss = train_epoch_rollout(
                    model, train_states_tensor, optimizer, device,
                    rollout_len=rollout_len, ss_prob=ss_prob, 
                    grad_clip=cfg.grad_clip, batch_size=4
                )
                train_losses.append(train_loss)
                
                epoch_time = time.time() - epoch_start
                
                # Evaluate every 10 epochs
                if epoch % 10 == 0 or epoch == n_epochs:
                    mse_per_step, test_mse, _ = evaluate_rollout(model, test_states, device)
                    eval_mses.append((total_epochs, rollout_len, test_mse))
                    
                    print(f"    Epoch {epoch:3d}/{n_epochs} | "
                          f"Train Loss: {train_loss:.6f} | "
                          f"Test Rollout MSE: {test_mse:.6f} | "
                          f"Time: {epoch_time:.1f}s")
                else:
                    print(f"    Epoch {epoch:3d}/{n_epochs} | "
                          f"Train Loss: {train_loss:.6f} | "
                          f"Time: {epoch_time:.1f}s")
            
            # Save checkpoint at end of each curriculum stage
            save_checkpoint(model, optimizer, total_epochs, run_dir, cfg,
                            {'rollout_len': rollout_len, 'ss_prob': ss_prob})
        
        # Final evaluation
        logger.info("Final evaluation...")
        mse_per_step, final_mse, predictions = evaluate_rollout(model, test_states, device)
        
        # Save results
        results_path = os.path.join(run_dir, "step2_results.npz")
        np.savez(
            results_path,
            train_losses=np.array(train_losses),
            eval_mses=np.array(eval_mses, dtype=object),
            mse_per_step=mse_per_step,
            predictions=predictions,
            ground_truth=test_states,
            initial_mse=initial_mse,
            final_mse=final_mse,
        )
        
        t_elapsed = time.time() - t_start
        
        print("\n" + "=" * 60)
        print("  STEP 2 TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Total time: {t_elapsed:.1f}s ({t_elapsed/60:.1f} min)")
        print(f"  Total epochs: {total_epochs}")
        print(f"  Initial rollout MSE: {initial_mse:.6f}")
        print(f"  Final rollout MSE: {final_mse:.6f}")
        print(f"  Improvement: {(initial_mse - final_mse) / initial_mse * 100:.1f}%")
        print(f"  Results saved to: {results_path}")
        print()
        
        logger.info(f"Step 2 completed in {t_elapsed:.1f}s")
        logger.info(f"Final rollout MSE: {final_mse:.6f}")
        
        # Save status
        with open(os.path.join(run_dir, "status.txt"), 'w') as f:
            f.write("step2_completed\n")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        with open(os.path.join(run_dir, "status.txt"), 'w') as f:
            f.write(f"step2_failed: {e}\n")
        raise


if __name__ == "__main__":
    main()
