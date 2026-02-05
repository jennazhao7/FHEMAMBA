"""
Quick check: compute the teacher's validation accuracy from saved logits.

Usage:
    python check_teacher_accuracy.py --path teacher_sst2_logits.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check teacher accuracy on SST-2 validation")
    parser.add_argument(
        "--path",
        type=str,
        default=str(Path(__file__).resolve().parent / "teacher_sst2_logits.pt"),
        help="Path to teacher_sst2_logits.pt",
    )
    args = parser.parse_args()

    logits_path = Path(args.path).expanduser().resolve()
    print(f"Loading teacher logits from: {logits_path}")
    pack = torch.load(str(logits_path), map_location="cpu")

    val_logits = torch.as_tensor(pack["validation"]["logits"]).float()
    val_labels = torch.as_tensor(pack["validation"]["labels"]).long()

    if val_logits.ndim != 2 or val_logits.size(-1) != 2:
        raise ValueError(f"Expected logits of shape [N, 2], got {tuple(val_logits.shape)}")
    if val_labels.ndim != 1 or val_labels.size(0) != val_logits.size(0):
        raise ValueError(
            f"Mismatched sizes: logits N={val_logits.size(0)} vs labels N={val_labels.size(0)}"
        )

    acc = compute_accuracy(val_logits, val_labels)
    print(f"Teacher validation accuracy: {acc:.4f} ({acc * 100:.2f}%)")


if __name__ == "__main__":
    main()


