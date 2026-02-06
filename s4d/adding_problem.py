"""
Adding Problem: long-range dependency benchmark (LRA-style setup).

Classic formulation: sequence of (value, mask) pairs; two positions are
marked (mask=1); target = sum of the two values at those positions.
Requires the model to find the two positions and add them (long-range).

Setup follows the spirit of Long Range Arena: fixed seeds, long sequences
(1K default), train/val/test splits, MSE + accuracy-within-tolerance.
Ref: Hochreiter & Schmidhuber; LRA (Tay et al., 2021).
"""

import argparse
import json
import math
from datetime import datetime, timezone
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_mini_s4d import MiniS4D


def adding_problem_sample(seq_len, rng=None):
    """
    One sample of the adding problem.
    Returns: x (seq_len, 2), target (scalar). x[:,0]=values in [0,1], x[:,1]=mask (two 1s, rest 0).
    """
    if rng is None:
        rng = torch.Generator()
    values = torch.rand(seq_len, generator=rng)
    indices = torch.randperm(seq_len, generator=rng)[:2]
    mask = torch.zeros(seq_len)
    mask[indices[0]] = 1.0
    mask[indices[1]] = 1.0
    target = (values[indices[0]] + values[indices[1]]).item()
    x = torch.stack([values, mask], dim=-1)  # (L, 2)
    return x, target


class AddingDataset(Dataset):
    """Adding problem dataset with fixed seed for reproducible splits (LRA-style)."""

    def __init__(self, num_samples, seq_len, seed=42):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.seed = seed
        self.rng = torch.Generator().manual_seed(seed)
        self._data = None  # generated on first access

    def _generate(self):
        if self._data is not None:
            return
        xs, ys = [], []
        for _ in range(self.num_samples):
            x, y = adding_problem_sample(self.seq_len, self.rng)
            xs.append(x)
            ys.append(y)
        self._data = (torch.stack(xs), torch.tensor(ys, dtype=torch.float32))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        self._generate()
        return self._data[0][idx], self._data[1][idx]


class AddingModel(nn.Module):
    """Encoder (2 -> d_model) + MiniS4D for adding problem. Input (B, L, 2) -> output (B, 1)."""

    def __init__(self, d_model=64, d_state=64, seq_len=1000, dropout=0.0):
        super().__init__()
        self.encoder = nn.Linear(2, d_model)
        self.s4d = MiniS4D(d_model=d_model, d_state=d_state, L=seq_len, dropout=dropout)

    def forward(self, x):
        # x: (B, L, 2)
        x = self.encoder(x)                    # (B, L, d_model)
        x = x.transpose(-1, -2)               # (B, d_model, L)
        return self.s4d(x)                    # (B, 1)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device).unsqueeze(-1)
        optimizer.zero_grad()
        out = model(x)
        loss = nn.functional.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def evaluate(model, loader, device, tolerance=0.04):
    model.eval()
    total_mse = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).unsqueeze(-1)
            out = model(x)
            mse = nn.functional.mse_loss(out, y, reduction="sum").item()
            total_mse += mse
            within = (out - y).abs().squeeze(-1) < tolerance
            correct += within.sum().item()
            n += x.size(0)
    mse = total_mse / max(n, 1)
    acc = correct / max(n, 1)
    return mse, acc


def save_results(path, info):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description="Adding problem (LRA-style) with MiniS4D")
    parser.add_argument("--seq_len", type=int, default=1000, help="Sequence length (LRA: long, e.g. 1K)")
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--d_state", type=int, default=64, help="S4D state size")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train_samples", type=int, default=10000, help="Train set size")
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--test_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ckpt", type=str, default=None, help="Load MiniS4D or full model checkpoint")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate (requires --ckpt)")
    parser.add_argument("--tolerance", type=float, default=0.04, help="Accuracy = frac within tolerance of target")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_results", type=str, default="s4d_adding_results.json",
                        help="Path to save test results JSON (set empty to disable)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Data (LRA-style fixed seeds per split)
    train_ds = AddingDataset(args.train_samples, args.seq_len, seed=args.seed)
    val_ds   = AddingDataset(args.val_samples,   args.seq_len, seed=args.seed + 1)
    test_ds  = AddingDataset(args.test_samples,  args.seq_len, seed=args.seed + 2)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # If a checkpoint is provided, try to infer d_model/d_state from it to avoid shape mismatch.
    ckpt_state = None
    if args.ckpt:
        ckpt_state = torch.load(args.ckpt, map_location=device)
        if isinstance(ckpt_state, dict) and "model" in ckpt_state:
            ckpt_state = ckpt_state["model"]
        # Full AddingModel checkpoint: encoder.weight is (d_model, 2)
        if "encoder.weight" in ckpt_state:
            inferred_d_model = ckpt_state["encoder.weight"].shape[0]
            args.d_model = inferred_d_model
            # Also infer d_state if present
            if "s4d.kernel_gen.log_A_real" in ckpt_state:
                _, inferred_d_state = ckpt_state["s4d.kernel_gen.log_A_real"].shape
                args.d_state = inferred_d_state
        # MiniS4D checkpoint: kernel_gen.log_A_real is (d_model, d_state)
        elif "kernel_gen.log_A_real" in ckpt_state:
            inferred_d_model, inferred_d_state = ckpt_state["kernel_gen.log_A_real"].shape
            args.d_model = inferred_d_model
            args.d_state = inferred_d_state

    model = AddingModel(
        d_model=args.d_model,
        d_state=args.d_state,
        seq_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)

    # Load checkpoint if provided
    if ckpt_state is not None:
        state = ckpt_state
        # If keys look like full AddingModel (has 'encoder'), load fully
        if any(k.startswith("encoder") for k in state.keys()):
            model.load_state_dict(state, strict=True)
            print("Loaded full AddingModel from checkpoint.")
        else:
            # Assume MiniS4D state_dict: load into s4d submodule
            model.s4d.load_state_dict(state, strict=False)
            print("Loaded MiniS4D weights into model.s4d (encoder left as init).")

    if args.eval_only:
        if not args.ckpt:
            raise SystemExit("--eval_only requires --ckpt")
        test_mse, test_acc = evaluate(model, test_loader, device, args.tolerance)
        print(f"Test MSE: {test_mse:.6f}  Acc (|pred-target|<{args.tolerance}): {test_acc*100:.2f}%")
        if args.save_results:
            info = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": "eval_only",
                "ckpt": args.ckpt,
                "seq_len": args.seq_len,
                "d_model": args.d_model,
                "d_state": args.d_state,
                "dropout": args.dropout,
                "train_samples": args.train_samples,
                "val_samples": args.val_samples,
                "test_samples": args.test_samples,
                "batch_size": args.batch_size,
                "tolerance": args.tolerance,
                "seed": args.seed,
                "split_seeds": {
                    "train": args.seed,
                    "val": args.seed + 1,
                    "test": args.seed + 2,
                },
                "uses_training_data_for_test": False,
                "test_mse": test_mse,
                "test_acc": test_acc,
            }
            save_results(args.save_results, info)
            print(f"Saved results to {args.save_results}")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_mse = float("inf")

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_mse, val_acc = evaluate(model, val_loader, device, args.tolerance)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), "s4d_adding_best.pt")
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.6f}  val_mse={val_mse:.6f}  val_acc={val_acc*100:.2f}%")

    model.load_state_dict(torch.load("s4d_adding_best.pt", map_location=device))
    test_mse, test_acc = evaluate(model, test_loader, device, args.tolerance)
    print(f"Test MSE: {test_mse:.6f}  Acc (|pred-target|<{args.tolerance}): {test_acc*100:.2f}%")
    if args.save_results:
        info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "train_and_eval",
            "ckpt": "s4d_adding_best.pt",
            "seq_len": args.seq_len,
            "d_model": args.d_model,
            "d_state": args.d_state,
            "dropout": args.dropout,
            "train_samples": args.train_samples,
            "val_samples": args.val_samples,
            "test_samples": args.test_samples,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "tolerance": args.tolerance,
            "seed": args.seed,
            "split_seeds": {
                "train": args.seed,
                "val": args.seed + 1,
                "test": args.seed + 2,
            },
            "uses_training_data_for_test": False,
            "test_mse": test_mse,
            "test_acc": test_acc,
        }
        save_results(args.save_results, info)
        print(f"Saved results to {args.save_results}")


if __name__ == "__main__":
    main()
