"""
One-batch overfit sanity check on SST-2.

Train TinyMamba on 64 training samples with only hard labels (alpha=0),
lr=1e-3, no scheduler/EMA, batch_size=32, for ~300 steps. Expect train
accuracy to approach ~100% if the model/optimization is wired correctly.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Iterable, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset  # type: ignore
from transformers import AutoTokenizer  # type: ignore

from mamba_student import TinyMambaClassifier


def make_tokenized_subset(
    split: str,
    tokenizer: AutoTokenizer,
    *,
    max_len: int = 128,
    subset_size: int = 64,
    seed: int = 7,
    batch_size: int = 32,
) -> tuple[Any, DataLoader]:
    ds = load_dataset("glue", "sst2")[split]

    if "idx" not in ds.column_names:
        ds = ds.add_column("idx", list(range(len(ds))))

    def _tok(batch: Dict[str, Iterable]) -> Dict[str, Any]:
        enc = tokenizer(
            batch["sentence"], truncation=True, padding="max_length", max_length=max_len
        )
        enc["idx"] = batch["idx"]
        return enc

    ds = ds.map(_tok, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "idx"])  # type: ignore

    # Deterministic subset of specified size
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=gen).tolist()
    indices = perm[: subset_size]
    sub = Subset(ds, indices)

    loader = DataLoader(sub, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    return sub, loader


@torch.no_grad()
def accuracy_on_loader(model: nn.Module, loader: Iterable, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids, attention_mask)
        preds = logits.argmax(-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(1, total)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters per spec
    alpha = 0.0  # hard labels only
    lr = 1e-3
    steps = 300
    batch_size = 32
    subset_size = 64
    max_len = 128
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    teacher_model_id = "textattack/bert-base-uncased-SST-2"
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id, use_fast=True)

    train_subset, train_loader = make_tokenized_subset(
        "train", tokenizer, max_len=max_len, subset_size=subset_size, seed=seed, batch_size=batch_size
    )
    print(f"Subset size: {len(train_subset)} (batch={batch_size}), total steps: {steps}")

    model = TinyMambaClassifier(
        vocab_size=tokenizer.vocab_size,
        d_model=192,
        n_layers=8,
        ssm_dim=64,
        expand=2,
        conv_kernel=3,
        num_classes=2,
        tie_embeddings=False,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.98))
    criterion = nn.functional.cross_entropy

    model.train()
    cyc = itertools.cycle(train_loader)
    for step in range(1, steps + 1):
        batch = next(cyc)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == steps:
            acc = accuracy_on_loader(model, train_loader, device)
            print(f"Step {step}/{steps} - loss={loss.item():.4f} train_acc={acc:.4f}")

    final_acc = accuracy_on_loader(model, train_loader, device)
    print(f"Final train accuracy on 64 examples: {final_acc:.4f} ({final_acc * 100:.2f}%)")


if __name__ == "__main__":
    main()






