import math
import os
import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Tuple, Iterable
from mamba_student import TinyMambaClassifier


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    T: float = 2.0,
    alpha: float = 0.7,
    label_smoothing: float = 0.0,
    use_margin=True,
    conf_power: float = 1.0,  # 0=no weighting, 1=linear, 2=quadratic
) -> tuple[torch.Tensor, dict[str, float]]:
    # --- hard CE (optionally smoothed) ---
    ce = nn.functional.cross_entropy(
        student_logits, labels, label_smoothing=label_smoothing
    )

    # --- teacher soft targets ---
    if use_margin and student_logits.size(-1) == 2:
        # match margins: s_m = (s1-s0)/T, t_m = (t1-t0)/T
        s_m = (student_logits[:, 1] - student_logits[:, 0]) / T
        t_m = (teacher_logits[:, 1] - teacher_logits[:, 0]) / T
        # confidence from teacher after temperature
        with torch.no_grad():
            pt = torch.sigmoid(t_m)  # prob of class 1 from margin
            w = (pt * (1 - pt)) * 4.0  # peaky near 0.5? invert if desired
            if conf_power != 0:
                w = (pt.clamp(1e-4, 1-1e-4) - 0.5).abs() * 2  # confidence 0..1
                w = w.pow(conf_power).detach()
        kl_or_mse = ((s_m - t_m) ** 2 * (w + 1e-6)).mean() * (T * T)
        loss = (1 - alpha) * ce + alpha * kl_or_mse
        parts = {"ce": ce.item(), "kl": kl_or_mse.item()}
        return loss, parts
    else:
        log_p_s = nn.functional.log_softmax(student_logits / T, dim=-1)
        p_t = nn.functional.softmax(teacher_logits / T, dim=-1)
        kl = nn.functional.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
        loss = (1 - alpha) * ce + alpha * kl
        return loss, {"ce": ce.item(), "kl": kl.item()}


def make_tokenized(
    split: str,
    tokenizer: AutoTokenizer,
    *,
    max_len: int = 128,
    batch_size: int = 128,
    shuffle: bool = False,
) -> Tuple[Any, DataLoader]:
    """Tokenise the SST‑2 split and build a dataloader.

    An ``idx`` column is added if missing and preserved through the
    mapping so that teacher logits can be looked up by index.
    """
    ds = load_dataset("glue", "sst2")[split]
    # ensure an idx column exists
    if "idx" not in ds.column_names:
        ds = ds.add_column("idx", list(range(len(ds))))

    def _tok(batch: Dict[str, Iterable]) -> Dict[str, Any]:
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


def evaluate(
    model: nn.Module,
    val_loader: Iterable,
    device: torch.device,
    ema_shadow: Dict[str, torch.Tensor],
) -> float:
    """Evaluate a model using its EMA weights on the validation set."""
    # store original weights
    original_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # load EMA weights
    model.load_state_dict(ema_shadow, strict=False)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    # restore original weights
    model.load_state_dict(original_state, strict=False)
    return correct / max(1, total)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Starting training…")

    # hyperparameters
    max_len = 128
    epochs = 14
    lr = 2e-4
    weight_decay = 0.01
    warmup_ratio = 0.06
    alpha = 0.9
    T = 4.0
    label_smoothing = 0.00
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # teacher assets
    # use the same id as in dump_teacher.py to ensure consistent tokenisation
    teacher_model_id = "textattack/bert-base-uncased-SST-2"
    logits_path = Path(__file__).resolve().parent / "teacher_sst2_logits.pt"
    teacher_pack = torch.load(str(logits_path), map_location="cpu")
    train_logits = teacher_pack["train"]["logits"].float()
    train_labels = teacher_pack["train"]["labels"].long()
    val_logits = teacher_pack["validation"]["logits"].float()
    val_labels = teacher_pack["validation"]["labels"].long()

    # tokeniser
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id, use_fast=True)
    vocab_size = tokenizer.vocab_size

    # datasets and loaders
    train_ds, train_loader = make_tokenized(
        "train", tokenizer, max_len=max_len, batch_size=128, shuffle=True
    )
    val_ds, val_loader = make_tokenized(
        "validation", tokenizer, max_len=max_len, batch_size=256, shuffle=False
    )

    # model
    model = TinyMambaClassifier(
        vocab_size=vocab_size,
        d_model=192,
        n_layers=8,
        ssm_dim=64,
        expand=2,
        conv_kernel=3,
        num_classes=2,
        tie_embeddings=False,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Train dataset size: {len(train_ds)}, batches: {len(train_loader)}")
    print(f"Val dataset size: {len(val_ds)}, batches: {len(val_loader)}")
    print(
        f"Teacher logits shapes: train={train_logits.shape}, val={val_logits.shape}"
    )

    # --- set up / resume EMA ---
    ema_decay = 0.999
    ckpt_path = Path(__file__).resolve().parent / "ckpts" / "best_ema.pt"
    
    # Initialize optimizer and scheduler first (needed for resume)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98)
    )
    total_steps = math.ceil(len(train_loader) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    checkpoint = None
    if ckpt_path.exists():
        print(f"[Resume] Loading EMA checkpoint from {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location=device)
        ema_state = checkpoint["model"]  # this is an EMA state_dict
        # Load EMA weights into the live model
        model.load_state_dict(ema_state, strict=False)
        # Recreate EMA shadow from the loaded model params
        ema_shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        best_val = float(checkpoint.get("acc", 0.0))
        print(f"[Resume] Resumed from best EMA (val acc={best_val:.4f})")
        
        # Load optimizer/scheduler state if available (for true resume)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("[Resume] Loaded optimizer state")
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
            print("[Resume] Loaded scheduler state")
        if "epoch" in checkpoint:
            start_epoch = int(checkpoint["epoch"]) + 1
            print(f"[Resume] Will continue from epoch {start_epoch}")
        else:
            start_epoch = 1
    else:
        # Fresh EMA from randomly initialized model
        ema_shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        best_val = 0.0
        start_epoch = 1

    os.makedirs(Path(__file__).resolve().parent / "ckpts", exist_ok=True)

    # initial evaluation before any training
    model.eval()
    correct = total = 0
    sample_logits: torch.Tensor | None = None
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            if sample_logits is None:
                sample_logits = logits[:5]
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    init_acc = correct / max(1, total)
    print(f"Initial val accuracy: {init_acc:.4f}")
    if sample_logits is not None:
        print(f"Sample logits (first 5): {sample_logits}")
        print(
            f"Sample predictions: {sample_logits.argmax(-1).tolist()}"
        )

    # Calculate starting step if resuming (checkpoint already loaded above if exists)
    step_idx = 0
    if checkpoint is not None and "step" in checkpoint:
        step_idx = int(checkpoint["step"])
        print(f"[Resume] Resuming from step {step_idx}")
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)
            idx = batch["idx"].long()  # already on CPU
            t_logits = train_logits[idx].to(device)

            # forward
            s_logits = model(input_ids, attention_mask)

            # before computing loss
            progress = step_idx / total_steps
            alpha_now = 0.5 + 0.4 * (1 - progress)  # 0.9 -> 0.5 over training
            T_now = 2.0 + (4.0 - 2.0) * (1 - progress)        # 4 -> 2
            loss, parts = kd_loss(
                s_logits, t_logits, labels_batch,
                T=T_now, alpha=alpha_now, label_smoothing=0.0,
                use_margin=True, conf_power=1.0
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # EMA update
            with torch.no_grad():
                current_state = model.state_dict()
                for k in ema_shadow.keys():
                    if k in current_state and current_state[k].dtype.is_floating_point:
                        ema_shadow[k].mul_(ema_decay).add_(
                            current_state[k].detach(), alpha=1 - ema_decay
                        )

            # update progress bar
            pbar.set_postfix(loss=float(loss.item()), ce=parts["ce"], kl=parts["kl"])

            # optional debug printing
            if epoch == 1 and step_idx in {0, 50, 100, 200}:
                avg_diff = torch.abs(s_logits[:, 0] - s_logits[:, 1]).mean().item()
                print(
                    f"\nStep {step_idx}: Loss={loss.item():.4f}, avg s_logits_diff={avg_diff:.4f}"
                )

            step_idx += 1

        # evaluate current weights
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_eval = batch["label"].to(device)
                logits = model(input_ids, attention_mask)
                preds = logits.argmax(-1)
                correct += (preds == labels_eval).sum().item()
                total += labels_eval.numel()
        eval_acc = correct / max(1, total)

        # evaluate EMA weights
        ema_acc = evaluate(model, val_loader, device, ema_shadow)
        print(
            f"[Val] current acc={eval_acc:.4f}, EMA acc={ema_acc:.4f}"
        )

        if ema_acc > best_val:
            best_val = ema_acc
            # persist EMA state dict with optimizer/scheduler for resume
            torch.save(
                {
                    "model": ema_shadow,
                    "acc": best_val,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "step": step_idx,
                },
                Path(__file__).resolve().parent / "ckpts" / "best_ema.pt",
            )
            print("Saved new best EMA checkpoint")

    print(f"Best EMA val acc: {best_val:.4f}")


if __name__ == "__main__":
    main()
