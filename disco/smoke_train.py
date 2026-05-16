"""Single-α DISCO smoke trainer for HW2D.

Author: Anthony Poole

Purpose: end-to-end CPU smoke. Build a small DISCO, train a few iterations
on snippets from one HW2D HDF5, confirm the loss decreases, save a
checkpoint. **Not for production runs.** Real multi-α training goes
through ``train_hw2d.py`` (TBD) on Vista.

Example::

    python -m disco.smoke_train \\
        --data /tmp/hw2d_synth \\
        --dataset hw2d_a10 \\
        --resolution 32 32 \\
        --hidden-dim 96 \\
        --n-past 4 --n-future 1 \\
        --batch-size 2 --iters 5

Notes
-----
- ``hidden_dim`` must be divisible by 12 (upstream's hard-coded
  ``RMSGroupNorm(12, dim)``) and by ``num_heads``.
- ``patch_size`` must divide each spatial dim.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

import disco  # registers `src` alias and dataset-specs shim
from disco.dataset_specs import HW2D_DATASET_SPECS, with_root
from disco.hw2d_dataset import HW2DMixedDataset


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HW2D × DISCO CPU smoke trainer")
    p.add_argument("--data", required=True, help="Root directory containing per-α subdirs")
    p.add_argument("--dataset", default="hw2d_a10", help="Which HW2D α to train on")
    p.add_argument("--resolution", nargs=2, type=int, default=(32, 32), metavar=("H", "W"))
    p.add_argument("--n-past", type=int, default=4)
    p.add_argument("--n-future", type=int, default=1)
    p.add_argument("--hidden-dim", type=int, default=96, help="must be divisible by 12 and num_heads")
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--processor-blocks", type=int, default=1)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=8, help="ODE solver max steps")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=None, help="Optional path to write a smoke checkpoint")
    p.add_argument("--device", default="cpu")
    return p


def main():
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)

    disco.register_hw2d_specs()
    from src.models.disco import DISCO  # noqa: PLC0415

    specs = with_root({args.dataset: HW2D_DATASET_SPECS[args.dataset]}, args.data)
    dset = HW2DMixedDataset(
        dataset_names=[args.dataset],
        specs_table=specs,
        n_past=args.n_past,
        n_future=args.n_future,
        split="train",
        resolution_override=tuple(args.resolution),
    )
    assert len(dset) >= args.batch_size, (
        f"dataset has {len(dset)} samples but batch_size={args.batch_size}"
    )
    loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=True)
    print(f"dataset {args.dataset}: {len(dset)} samples; resolution={tuple(args.resolution)}")

    model = DISCO(
        n_states=2,
        hidden_dim=args.hidden_dim,
        patch_size=args.patch_size,
        ndims=[2],
        groups=1,
        processor_blocks=args.processor_blocks,
        drop_path=0.0,
        num_heads=args.num_heads,
        bias_type="space-time",
        hpnn_head_hidden_dim=args.hidden_dim,
        dataset_names=[args.dataset],
        max_steps=args.max_steps,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DISCO instantiated: {n_params:,} params on {args.device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    state_labels = torch.tensor([0, 1], dtype=torch.long, device=args.device)
    loss_fn = nn.MSELoss()

    losses = []
    it = iter(loader)
    t0 = time.time()
    for step in range(args.iters):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        x = batch["input_fields"].to(args.device)
        y_true = batch["output_fields"].to(args.device)
        opt.zero_grad()
        y_pred, _meta = model(x, state_labels=state_labels, dset_name=args.dataset)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        losses.append(loss.item())
        print(f"step {step:3d}  loss={loss.item():.4e}  grad_norm={gn.item():.4e}")

    dt = time.time() - t0
    print(f"\nfinished {args.iters} steps in {dt:.1f}s "
          f"({dt/args.iters:.2f}s/step), "
          f"loss: {losses[0]:.4e} → {losses[-1]:.4e}")

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "losses": losses,
                "args": vars(args),
            },
            save_path,
        )
        print(f"checkpoint written to {save_path}")


if __name__ == "__main__":
    main()
