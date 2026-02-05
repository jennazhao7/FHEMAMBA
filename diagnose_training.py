"""
Diagnostic script to identify why val acc isn't improving despite probe showing gradients work.
"""
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path
from mamba_student import TinyMambaClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load same config as gpt_train.py
teacher_model_id = "textattack/bert-base-uncased-SST-2"
tokenizer = AutoTokenizer.from_pretrained(teacher_model_id, use_fast=True)
max_len = 128

# Load teacher logits
logits_path = Path("teacher_sst2_logits.pt")
pack = torch.load(str(logits_path), map_location="cpu")
train_logits = pack["train"]["logits"].float()
val_logits = pack["validation"]["logits"].float()

# Create dataset the same way as gpt_train.py
ds = load_dataset("glue", "sst2")["train"]
if "idx" not in ds.column_names:
    ds = ds.add_column("idx", list(range(len(ds))))

def _tok(batch):
    enc = tokenizer(
        batch["sentence"], truncation=True, padding="max_length", max_length=max_len
    )
    enc["idx"] = batch["idx"]
    return enc

ds = ds.map(_tok, batched=True)
ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "idx"])

# Check alignment: do the idx values match the teacher logits order?
print("\n=== Checking teacher logit alignment ===")
sample_indices = [0, 100, 1000]
for idx in sample_indices:
    sample = ds[idx]
    sentence = sample["sentence"] if "sentence" in ds.column_names else "N/A"
    ds_idx = sample["idx"].item()
    print(f"Dataset[{idx}]: idx={ds_idx}, label={sample['label'].item()}, "
          f"teacher_logit_shape={train_logits[ds_idx].shape if ds_idx < len(train_logits) else 'OUT_OF_BOUNDS'}")

# Verify first few match
print("\n=== Verifying first 10 examples match ===")
for i in range(min(10, len(ds))):
    ds_idx = ds[i]["idx"].item()
    if ds_idx != i:
        print(f"WARNING: Dataset[{i}] has idx={ds_idx} (expected {i})")
    if ds_idx >= len(train_logits):
        print(f"ERROR: idx {ds_idx} is out of bounds for train_logits (len={len(train_logits)})")

# Test model forward
print("\n=== Testing model forward ===")
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

# Test on a single batch
batch = ds[:32]
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)
labels = batch["label"].to(device)
idx_batch = batch["idx"].long().cpu()

model.eval()
with torch.no_grad():
    logits = model(input_ids, attention_mask)
    preds = logits.argmax(-1)
    acc = (preds == labels).float().mean().item()

print(f"Batch accuracy: {acc:.4f}")
print(f"Logits stats: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
print(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
print(f"Preds: {preds[:10].tolist()}")
print(f"Labels: {labels[:10].tolist()}")

# Test KD loss calculation
print("\n=== Testing KD loss ===")
teacher_logits_batch = train_logits[idx_batch].to(device)

alpha = 0.7
T = 2.0
label_smoothing = 0.05

# Hard CE
ce = nn.functional.cross_entropy(logits, labels, label_smoothing=label_smoothing)
# Soft KL
log_p_s = nn.functional.log_softmax(logits / T, dim=-1)
p_t = nn.functional.softmax(teacher_logits_batch / T, dim=-1)
kl = nn.functional.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
loss = (1.0 - alpha) * ce + alpha * kl

print(f"CE: {ce.item():.4f}")
print(f"KL: {kl.item():.4f}")
print(f"Total loss: {loss.item():.4f}")

# Check if model can actually learn (gradient flow test)
print("\n=== Testing gradient flow ===")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.05)
optimizer.zero_grad()
logits = model(input_ids, attention_mask)
loss = nn.functional.cross_entropy(logits, labels)
loss.backward()

grad_norms = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norms[name] = param.grad.norm().item()
    else:
        grad_norms[name] = 0.0

print("Gradient norms:")
for name, norm in sorted(grad_norms.items())[:10]:  # Show first 10
    print(f"  {name}: {norm:.6f}")
print(f"  ... (total {len(grad_norms)} parameters)")

print(f"\nEmbed grad: {grad_norms.get('embed.weight', 0):.6f}")
print(f"Head grad: {grad_norms.get('head.weight', 0):.6f}")

# Check if predictions are changing
print("\n=== Testing if model learns over multiple steps ===")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.05)

initial_preds = None
for step in range(10):
    optimizer.zero_grad()
    logits = model(input_ids, attention_mask)
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if step == 0:
        initial_preds = logits.argmax(-1).clone()
    elif step % 3 == 0:
        current_preds = logits.argmax(-1)
        change_rate = (current_preds != initial_preds).float().mean().item()
        acc = (current_preds == labels).float().mean().item()
        print(f"Step {step}: loss={loss.item():.4f}, acc={acc:.4f}, pred_change_rate={change_rate:.4f}")

final_preds = logits.argmax(-1)
final_acc = (final_preds == labels).float().mean().item()
final_loss = nn.functional.cross_entropy(logits, labels).item()
print(f"After 10 steps: loss={final_loss:.4f}, acc={final_acc:.4f}")




