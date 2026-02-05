"""
One-batch overfit test using EXACT same setup as gpt_train.py
to verify training works correctly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm

from mamba_student import TinyMambaClassifier


def make_tokenized(split, tokenizer, max_len=128, batch_size=128, shuffle=False):
    """Exact copy from gpt_train.py"""
    ds = load_dataset("glue", "sst2")[split]
    if "idx" not in ds.column_names:
        ds = ds.add_column("idx", list(range(len(ds))))

    def _tok(batch):
        enc = tokenizer(
            batch["sentence"], truncation=True, padding="max_length",
            max_length=max_len
        )
        enc["idx"] = batch["idx"]
        return enc

    ds = ds.map(_tok, batched=True)
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label", "idx"],
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )
    return ds, loader


def kd_loss(student_logits, teacher_logits, labels, *, T=2.0, alpha=0.7, label_smoothing=0.0):
    """Exact copy from gpt_train.py"""
    ce = nn.functional.cross_entropy(student_logits, labels, label_smoothing=label_smoothing)
    log_p_s = nn.functional.log_softmax(student_logits / T, dim=-1)
    p_t = nn.functional.softmax(teacher_logits / T, dim=-1)
    kl = nn.functional.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
    loss = (1.0 - alpha) * ce + alpha * kl
    return loss, {"ce": ce.item(), "kl": kl.item()}


@torch.no_grad()
def accuracy(model, loader, device):
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Overfit params (as specified)
    max_len = 128
    lr = 1e-3
    alpha = 0.0  # hard labels only
    T = 2.0
    label_smoothing = 0.0
    batch_size = 32
    subset_size = 64
    steps = 300
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Same setup as gpt_train.py
    teacher_model_id = "textattack/bert-base-uncased-SST-2"
    logits_path = Path(__file__).resolve().parent / "teacher_sst2_logits.pt"
    teacher_pack = torch.load(str(logits_path), map_location="cpu")
    train_logits = teacher_pack["train"]["logits"].float()

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id, use_fast=True)

    # Create full dataset, then subset
    train_ds, _ = make_tokenized("train", tokenizer, max_len=max_len, batch_size=batch_size, shuffle=False)
    
    # Create balanced subset (32 pos, 32 neg)
    torch.manual_seed(7)  # for reproducibility
    pos_indices = [i for i in range(len(train_ds)) if train_ds[i]["label"].item() == 1][:32]
    neg_indices = [i for i in range(len(train_ds)) if train_ds[i]["label"].item() == 0][:32]
    subset_indices = pos_indices + neg_indices
    
    subset_ds = Subset(train_ds, subset_indices)
    subset_loader = DataLoader(
        subset_ds,
        batch_size=batch_size,
        shuffle=True,  # shuffle for training
        drop_last=False,
        num_workers=0,
    )

    print(f"Subset size: {len(subset_indices)}")
    print(f"Batches per epoch: {len(subset_loader)}")
    print(f"Will train for {steps} steps")

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    # Initial accuracy
    init_acc = accuracy(model, subset_loader, device)
    print(f"\nInitial train accuracy: {init_acc:.4f}")

    # Training loop
    model.train()
    import itertools
    loader_cycle = itertools.cycle(subset_loader)
    
    for step in range(1, steps + 1):
        batch = next(loader_cycle)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch = batch["label"].to(device)
        idx = batch["idx"].long().cpu()
        
        # Get teacher logits (same as gpt_train.py)
        t_logits = train_logits[idx].to(device)
        
        # Forward
        s_logits = model(input_ids, attention_mask)
        loss, parts = kd_loss(s_logits, t_logits, labels_batch, T=T, alpha=alpha, label_smoothing=label_smoothing)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == steps:
            acc = accuracy(model, subset_loader, device)
            print(f"Step {step}/{steps}: loss={loss.item():.4f}, train_acc={acc:.4f}")

    final_acc = accuracy(model, subset_loader, device)
    print(f"\nFinal train accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
    
    if final_acc < 0.95:
        print("⚠️  WARNING: Final accuracy < 95%. Model may not be learning correctly.")
        print("This could indicate:")
        print("  - Model forward pass issue")
        print("  - Gradient flow problem")
        print("  - Data/tokenization mismatch")
    else:
        print("✓ Model successfully overfits - training setup appears correct.")


if __name__ == "__main__":
    main()




