import math, torch, os
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
from mamba_student import TinyMambaClassifier

# ---------------- KD loss ----------------
def kd_loss(student_logits, teacher_logits, y, T=2.0, alpha=0.7, label_smoothing=0.05):
    # hard CE with smoothing
    if label_smoothing > 0:
        n = student_logits.size(-1)
        with torch.no_grad():
            smoothed = torch.full_like(student_logits, fill_value=label_smoothing/(n-1))
            smoothed.scatter_(1, y.view(-1,1), 1.0 - label_smoothing)
        ce = torch.mean(torch.sum(-smoothed * torch.log_softmax(student_logits, dim=-1), dim=-1))
    else:
        ce = nn.functional.cross_entropy(student_logits, y)

    # soft KL (teacher -> student)
    log_p_s = nn.functional.log_softmax(student_logits / T, dim=-1)
    p_t = nn.functional.softmax(teacher_logits / T, dim=-1)
    kl = nn.functional.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

    # weight them
    return (1 - alpha) * ce + alpha * kl, {"ce": ce.item(), "kl": kl.item()}

def make_tokenized(split, tok, max_len=128, bs=128, shuffle=False):
    ds = load_dataset("glue", "sst2")[split]
    # Add index column to track original positions
    if "idx" not in ds.column_names:
        ds = ds.add_column("idx", list(range(len(ds))))
    def _tok(batch): 
        enc = tok(batch["sentence"], truncation=True, padding="max_length", max_length=max_len)
        # Preserve idx
        enc["idx"] = batch["idx"]
        return enc
    # map preserves idx if we add it to the tokenized columns
    ds = ds.map(_tok, batched=True)
    ds.set_format(type="torch", columns=["input_ids","attention_mask","label","idx"])
    return ds, DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False, num_workers=0)

# # --------------- data --------------------
# def make_tokenized(split, tok, max_len=128, bs=128, shuffle=False):
#     ds = load_dataset("glue", "sst2")[split]
#     def _tok(batch): return tok(batch["sentence"], truncation=True, padding="max_length", max_length=max_len)
#     ds = ds.map(_tok, batched=True)
#     ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
#     return ds, DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False, num_workers=2)

# --------------- training ----------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Starting training...")
    
    max_len = 128
    epochs = 14
    lr = 2e-4
    wd = 0.05
    warmup_ratio = 0.06
    alpha = 0.7
    T = 2.0
    label_smoothing = 0.05
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # teacher assets
    TEACHER_ID = "bert-base-uncased"  # tokenizer only; logits are loaded from file
    TEACHER_LOGITS = "teacher_sst2_logits.pt"  # produced by dump_teacher.py
    teacher_pack = torch.load(TEACHER_LOGITS)
    
    train_logits = torch.as_tensor(teacher_pack["train"]["logits"])      # [N_train, 2]
    train_labels = torch.as_tensor(teacher_pack["train"]["labels"])      # [N_train]
    val_logits   = torch.as_tensor(teacher_pack["validation"]["logits"]) # [N_val, 2]
    val_labels   = torch.as_tensor(teacher_pack["validation"]["labels"]) # [N_val]


    tok = AutoTokenizer.from_pretrained(TEACHER_ID, use_fast=True)
    vocab_size = tok.vocab_size

    # data
    train_ds, train_loader = make_tokenized("train", tok, max_len=max_len, bs=128, shuffle=True)
    val_ds,   val_loader   = make_tokenized("validation", tok, max_len=max_len, bs=256, shuffle=False)

    # model
    model = TinyMambaClassifier(
        vocab_size=vocab_size, d_model=192, n_layers=8, ssm_dim=64, expand=2, conv_kernel=3, num_classes=2, tie_embeddings=False
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Train dataset size: {len(train_ds)}, batches: {len(train_loader)}")
    print(f"Val dataset size: {len(val_ds)}, batches: {len(val_loader)}")
    print(f"Teacher logits shapes: train={train_logits.shape}, val={val_logits.shape}")

    # EMA (simple)
    ema_decay = 0.999
    ema_shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    total_steps = math.ceil(len(train_loader) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    best_val = 0.0
    os.makedirs("ckpts", exist_ok=True)

    # Initial validation (before EMA)
    print("\n=== Initial Evaluation ===")
    model.eval()
    correct, total = 0, 0
    sample_logits = []
    for i, batch in enumerate(val_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)
        logits = model(input_ids, attention_mask)
        if i == 0:
            sample_logits = logits[:5]
        pred = logits.argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    init_acc = correct / max(1, total)
    print(f"Initial val accuracy: {init_acc:.4f}")
    print(f"Sample logits (first 5): {sample_logits}")
    print(f"Sample predictions: {sample_logits.argmax(-1).tolist()}\n")

    step_idx = 0
    for ep in range(1, epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
        for i, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)
            
            # Get teacher logits using the original dataset indices
            idx = batch["idx"].to(torch.long).cpu()  # Get indices from batch
            t_logits = train_logits[idx].to(device)

            s_logits = model(input_ids, attention_mask)
            
            # Debug: check shapes and values on first batch
            if i == 0 and ep == 1:
                print(f"\nDebug first batch:")
                print(f"  idx range: {idx.min().item()} to {idx.max().item()}")
                print(f"  y shape: {y.shape}, sample labels: {y[:5].tolist()}")
                print(f"  t_logits shape: {t_logits.shape}, sample values: {t_logits[0]}")
                print(f"  s_logits shape: {s_logits.shape}, sample values: {s_logits[0]}")

            loss, parts = kd_loss(s_logits, t_logits, y, T=T, alpha=alpha, label_smoothing=label_smoothing)
            
            if i == 0 and ep == 1:
                print(f"  Loss: {loss.item():.4f}, CE: {parts['ce']:.4f}, KL: {parts['kl']:.4f}\n")
                
            opt.zero_grad(set_to_none=True)
            loss.backward()
            
            # Check gradient norms
            if i == 0 and ep == 1:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"  Grad norm: {total_norm:.6f}\n")
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            # EMA update
            with torch.no_grad():
                # Update EMA weights
                current_state = model.state_dict()
                for k in ema_shadow.keys():
                    if k in current_state and current_state[k].dtype.is_floating_point:
                        ema_shadow[k].mul_(ema_decay).add_(current_state[k].detach(), alpha=1-ema_decay)

            pbar.set_postfix(loss=float(loss.item()), ce=parts["ce"], kl=parts["kl"])
            
            # Track loss trajectory
            if ep == 1 and step_idx in [0, 50, 100, 200]:
                print(f"\nStep {step_idx}: Loss={loss.item():.4f}, avg s_logits_diff={torch.abs(s_logits[:, 0] - s_logits[:, 1]).mean().item():.4f}")
            
            step_idx += 1

        # --------- Eval (current weights for debugging) ----------
        model.eval()
        correct, total = 0, 0
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            pred = logits.argmax(-1)
            correct += (pred == y).sum().item()
            total += y.numel()
        eval_acc = correct / max(1, total)
        
        print(f"[Val] acc={eval_acc:.4f}")
        if eval_acc > best_val:
            best_val = eval_acc
            torch.save({"model": ema_shadow, "acc": best_val}, f"ckpts/best_ema.pt")
            print("Saved ckpts/best_ema.pt")

    print(f"Best val acc: {best_val:.4f}")

@torch.no_grad()
def evaluate(model, val_loader, device, ema_shadow):
    # swap in EMA weights
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema_shadow, strict=False)

    model.eval()
    correct, total = 0, 0
    
    # Debug: check if EMA weights are different from original
    if total == 0:  # First call
        diff = sum((ema_shadow[k] - original[k]).abs().sum() for k in ema_shadow.keys())
        print(f"EMA vs original weight diff: {diff.item():.6f}")
    
    for i, batch in enumerate(val_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)
        logits = model(input_ids, attention_mask)
        
        # Debug: check if outputs vary
        if i == 0 and total == 0:
            print(f"First batch logits sample: {logits[:3]}")
        
        pred = logits.argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()

    # restore original weights
    model.load_state_dict(original, strict=False)
    return correct / max(1, total)

if __name__ == "__main__":
    main()
