"""
GPU Memory Test for FNO Configuration.

Tests whether a given FNO config will fit in GPU memory by simulating
Step 1 (single-step) and Step 2 (rollout) training under realistic conditions.

Runs MULTIPLE forward+backward iterations to capture memory fragmentation
that only appears during sustained training.

Usage:
    python test_gpu_memory.py --config config/fno_temporal_split.yaml

Author: Anthony Poole
"""

import argparse
import sys
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_1_train import create_model


def get_gpu_mem_mb():
    return torch.cuda.memory_allocated() / 1024**2

def get_gpu_peak_mb():
    return torch.cuda.max_memory_allocated() / 1024**2

def get_gpu_reserved_mb():
    return torch.cuda.memory_reserved() / 1024**2

def get_gpu_total_mb():
    return torch.cuda.get_device_properties(0).total_memory / 1024**2


def test_step1(model, cfg, device):
    """Simulate Step 1: multiple batches of forward + backward (realistic)."""
    batch_size = cfg.get('training', {}).get('batch_size', 1)
    weight_decay = cfg.get('training', {}).get('weight_decay', 1e-4)
    lr = cfg.get('training', {}).get('learning_rate', 5e-4)
    grad_clip = cfg.get('training', {}).get('grad_clip', 1.0)
    n_fields = cfg.get('physics', {}).get('n_fields', 2)
    n_x = cfg.get('physics', {}).get('n_x', 512)
    n_y = cfg.get('physics', {}).get('n_y', 512)
    n_iters = 10  # Multiple iterations to test fragmentation
    
    print(f"\n{'='*60}")
    print(f"  STEP 1 TEST: batch={batch_size}, {n_iters} iterations")
    print(f"  (AdamW, weight_decay={weight_decay}, grad_clip={grad_clip})")
    print(f"{'='*60}")
    
    torch.cuda.reset_peak_memory_stats()
    
    # Use AdamW to match real training exactly
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=lr*0.01)
    
    for i in range(n_iters):
        x = torch.randn(batch_size, n_fields, n_x, n_y, device=device)
        y = torch.randn(batch_size, n_fields, n_x, n_y, device=device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        del x, y, pred, loss
        
        print(f"  Iter {i+1}/{n_iters}: alloc={get_gpu_mem_mb():.0f} MB, "
              f"reserved={get_gpu_reserved_mb():.0f} MB, peak={get_gpu_peak_mb():.0f} MB")
    
    peak = get_gpu_peak_mb()
    reserved = get_gpu_reserved_mb()
    
    del optimizer, scheduler
    torch.cuda.empty_cache()
    
    return peak, reserved


def test_step2_rollout(model, cfg, device):
    """Simulate Step 2: rollout with gradient checkpointing at max rollout length."""
    curriculum = cfg.get('curriculum', [[80, 0.0, 30]])
    max_rollout = max(stage[0] for stage in curriculum)
    weight_decay = cfg.get('training', {}).get('weight_decay', 1e-4)
    lr = cfg.get('training', {}).get('learning_rate', 5e-4)
    n_fields = cfg.get('physics', {}).get('n_fields', 2)
    n_x = cfg.get('physics', {}).get('n_x', 512)
    n_y = cfg.get('physics', {}).get('n_y', 512)
    use_checkpointing = cfg.get('training', {}).get('gradient_checkpointing', True)
    
    print(f"\n{'='*60}")
    print(f"  STEP 2 TEST: rollout={max_rollout}, checkpointing={use_checkpointing}")
    print(f"{'='*60}")
    
    torch.cuda.reset_peak_memory_stats()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer.zero_grad()
    
    x = torch.randn(1, n_fields, n_x, n_y, device=device, requires_grad=True)
    rollout_loss = 0.0
    
    for step in range(max_rollout):
        y_true = torch.randn(1, n_fields, n_x, n_y, device=device)
        
        if use_checkpointing:
            y_pred = checkpoint(model, x, use_reentrant=False)
        else:
            y_pred = model(x)
        
        step_loss = nn.functional.mse_loss(y_pred, y_true)
        rollout_loss = rollout_loss + step_loss
        
        x = y_pred.detach().requires_grad_(True)
        del y_true, y_pred
        
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}/{max_rollout}: alloc={get_gpu_mem_mb():.0f} MB, "
                  f"reserved={get_gpu_reserved_mb():.0f} MB, peak={get_gpu_peak_mb():.0f} MB")
    
    rollout_loss = rollout_loss / max_rollout
    rollout_loss.backward()
    optimizer.step()
    
    peak = get_gpu_peak_mb()
    reserved = get_gpu_reserved_mb()
    print(f"  After backward: alloc={get_gpu_mem_mb():.0f} MB, "
          f"reserved={reserved:.0f} MB, peak={peak:.0f} MB")
    
    del x, rollout_loss, optimizer
    torch.cuda.empty_cache()
    
    return peak, reserved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = 'cuda'
    total_mem = get_gpu_total_mb()
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {total_mem:.0f} MB")
    print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(not set)')}")
    
    model = create_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"After model load: {get_gpu_mem_mb():.0f} MB")
    
    # Test Step 1 (multi-iteration)
    step1_peak, step1_reserved = test_step1(model, cfg, device)
    
    # Test Step 2
    step2_peak, step2_reserved = test_step2_rollout(model, cfg, device)
    
    max_peak = max(step1_peak, step2_peak)
    max_reserved = max(step1_reserved, step2_reserved)
    usage_pct = max_peak / total_mem * 100
    reserved_pct = max_reserved / total_mem * 100
    fragmentation = max_reserved - max_peak
    
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Step 1 peak: {step1_peak:.0f} MB ({step1_peak/total_mem*100:.1f}%)")
    print(f"  Step 2 peak: {step2_peak:.0f} MB ({step2_peak/total_mem*100:.1f}%)")
    print(f"  Max peak:      {max_peak:.0f} MB ({usage_pct:.1f}%)")
    print(f"  Max reserved:  {max_reserved:.0f} MB ({reserved_pct:.1f}%)")
    print(f"  Fragmentation: {fragmentation:.0f} MB")
    print(f"  True headroom: {total_mem - max_reserved:.0f} MB")
    
    if reserved_pct > 90:
        print(f"\n  *** FAIL: Reserved memory > 90% — will likely OOM! ***")
        sys.exit(1)
    elif usage_pct > 85:
        print(f"\n  *** WARNING: Peak usage > 85% — tight, may OOM under load ***")
        sys.exit(1)
    else:
        print(f"\n  *** PASS: Memory usage OK ***")
        sys.exit(0)


if __name__ == '__main__':
    main()
