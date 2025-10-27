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

# --------------- data --------------------
def make_tokenized(split, tok, max_len=128, bs=128, shuffle=False):
    ds = load_dataset("glue", "sst2")[split]
    def _tok(batch): return tok(batch["sentence"], truncation=True, padding="max_length", max_length=max_len)
    ds = ds.map(_tok, batched=True)
    ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
    return ds, DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False, num_workers=2)

# --------------- training ----------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    train_logits = teacher_pack["train"]["logits"]
    train_labels = teacher_pack["train"]["labels"]
    val_logits   = teacher_pack["validation"]["logits"]
    val_labels   = teacher_pack["validation"]["labels"]

    tok = AutoTokenizer.from_pretrained(TEACHER_ID, use_fast=True)
    vocab_size = tok.vocab_size

    # model
    model = TinyMambaClassifier(
        vocab_size=vocab_size, d_model=192, n_layers=8, ssm_dim=64, expand=2, conv_kernel=3, num_classes=2, tie_embeddings=False
    ).to(device)

    # data
    train_ds, train_loader = make_tokenized("train", tok, max_len=max_len, bs=128, shuffle=True)
    val_ds,   val_loader   = make_tokenized("validation", tok, max_len=max_len, bs=256, shuffle=False)

    # EMA (simple)
    ema_decay = 0.999
    ema_shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    total_steps = math.ceil(len(train_loader) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    best_val = 0.0
    os.makedirs("ckpts", exist_ok=True)

    step_idx = 0
    for ep in range(1, epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
        for i, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)

            # align teacher logits by absolute dataset index
            # DataLoader preserves order except for shuffle; so compute global indices via i*bs: i*bs+len
            start = i * train_loader.batch_size
            end = start + input_ids.size(0)
            t_logits = train_logits[start:end].to(device)

            s_logits = model(input_ids, attention_mask)

            loss, parts = kd_loss(s_logits, t_logits, y, T=T, alpha=alpha, label_smoothing=label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            # EMA update
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema_shadow[k].mul_(ema_decay).add_(v.detach(), alpha=1-ema_decay)

            pbar.set_postfix(loss=float(loss.item()), ce=parts["ce"], kl=parts["kl"])
            step_idx += 1

        # --------- Eval (EMA weights) ----------
        eval_acc = evaluate(model, val_loader, val_logits, device, ema_shadow)
        print(f"[Val] acc={eval_acc:.4f}")
        if eval_acc > best_val:
            best_val = eval_acc
            torch.save({"model": ema_shadow, "acc": best_val}, f"ckpts/best_ema.pt")
            print("Saved ckpts/best_ema.pt")

    print(f"Best val acc: {best_val:.4f}")

@torch.no_grad()
def evaluate(model, val_loader, val_logits, device, ema_shadow):
    # swap in EMA weights
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema_shadow, strict=False)

    model.eval()
    correct, total = 0, 0
    for i, batch in enumerate(val_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)
        logits = model(input_ids, attention_mask)
        pred = logits.argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()

    # restore original weights
    model.load_state_dict(original, strict=False)
    return correct / max(1, total)

if __name__ == "__main__":
    main()
