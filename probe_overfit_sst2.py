import torch, torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from mamba_student import TinyMambaClassifier  # your file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- data: same tokenizer & max_len you use everywhere ---
teacher_model_id = "textattack/bert-base-uncased-SST-2"
tok = AutoTokenizer.from_pretrained(teacher_model_id, use_fast=True)
max_len = 128

ds = load_dataset("glue", "sst2")["train"]
# Keep a fixed, tiny, *balanced-ish* subset to avoid trivial single-class issues
idxs0 = [i for i in range(len(ds)) if ds[i]["label"]==0][:32]
idxs1 = [i for i in range(len(ds)) if ds[i]["label"]==1][:32]
subset_idx = torch.tensor(idxs0 + idxs1)
subset = [ds[i] for i in subset_idx.tolist()]

enc = tok([r["sentence"] for r in subset],
          truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
input_ids = enc["input_ids"].to(device)
attn = enc["attention_mask"].to(device)
labels = torch.tensor([r["label"] for r in subset], dtype=torch.long, device=device)

print("label counts:", torch.bincount(labels).tolist())
print("attn token count per sample (min/max):",
      int(attn.sum(1).min().item()), int(attn.sum(1).max().item()))

# --- model ---
model = TinyMambaClassifier(vocab_size=tok.vocab_size,
                            d_model=192, n_layers=8, ssm_dim=64,
                            expand=2, conv_kernel=3, num_classes=2,
                            tie_embeddings=False).to(device)

# temporarily use masked-mean pooling AFTER stack:
def forward_tokens_only(m, input_ids, attention_mask):
    x = m.embed(input_ids)
    bsz = x.size(0)
    cls_token = m.cls.expand(bsz,1,-1)
    x = torch.cat([x, cls_token], dim=1)
    for layer in m.layers:
        residual = x
        x = layer(x)
        x = residual + x
    x = m.norm(x)
    denom = (attention_mask.sum(1, keepdim=True).clamp_min(1)).to(x.dtype)
    pooled = (x[:, :-1] * attention_mask.unsqueeze(-1)).sum(1) / denom
    return m.head(pooled)

model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# --- forward before update ---
with torch.no_grad():
    logits0 = forward_tokens_only(model, input_ids, attn)
print("logits mean/std BEFORE:", float(logits0.mean()), float(logits0.std()))
print("per-class logit means BEFORE:",
      logits0[labels==0].mean(dim=0).tolist(), logits0[labels==1].mean(dim=0).tolist())
print("emb var pre-stack:", float(model.embed(input_ids).var()))

# --- one step ---
opt.zero_grad(set_to_none=True)
logits = forward_tokens_only(model, input_ids, attn)
loss = nn.functional.cross_entropy(logits, labels)
print("loss BEFORE:", float(loss))
loss.backward()

# gradient norms on key parts
g_head = model.head.weight.grad
g_emb  = model.embed.weight.grad
print("grad ||head||:", 0.0 if g_head is None else float(g_head.norm()))
print("grad ||embed||:", 0.0 if g_emb is None else float(g_emb.norm()))

# param snapshots to prove update happens
w_head_before = model.head.weight.detach().clone()
opt.step()

with torch.no_grad():
    logits1 = forward_tokens_only(model, input_ids, attn)
    loss1   = nn.functional.cross_entropy(logits1, labels)
    delta_head = (model.head.weight - w_head_before).norm().item()

print("loss AFTER one step:", float(loss1))
print("Δ||head.weight|| after opt.step:", delta_head)
print("logits mean/std AFTER:", float(logits1.mean()), float(logits1.std()))
