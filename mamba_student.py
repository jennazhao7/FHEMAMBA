import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm

class MambaBlock(nn.Module):
    def __init__(self, d_model, ssm_dim=64, expand=2, conv_kernel=3):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=ssm_dim, d_conv=conv_kernel, expand=expand)
    def forward(self, x, attn_mask=None):
        return self.mamba(self.norm(x))

class TinyMambaClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=192, n_layers=8, ssm_dim=64, expand=2, conv_kernel=3, num_classes=2, tie_embeddings=True):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1,1,d_model) * 0.02)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model, ssm_dim, expand, conv_kernel) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes, bias=True)
        if tie_embeddings:
            self.head.weight = self.embed.weight[:num_classes]  # lightweight tie (optional); comment out if you prefer untied
    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)  # [B, L, D]
        B = x.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # prepend CLS
        for blk in self.layers:
            x = x + blk(x)
        x = self.norm(x)
        cls_h = x[:,0]  # take CLS
        return self.head(cls_h)
