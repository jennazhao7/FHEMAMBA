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
        # Use larger initialization for CLS token
        self.cls = nn.Parameter(torch.randn(1,1,d_model))
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model, ssm_dim, expand, conv_kernel) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes, bias=True)
        self.tie_embeddings = tie_embeddings
        
        # Initialize properly
        self.apply(self._init_weights)
        
        # Tie embeddings AFTER initialization if needed
        if self.tie_embeddings and num_classes <= vocab_size:
            with torch.no_grad():
                self.head.weight.copy_(self.embed.weight[:num_classes])
        
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)  # [B, L, D]
        B = x.size(0)
        # Append CLS at the END so it can see the full sequence
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([x, cls], dim=1)  # append CLS
        for blk in self.layers:
            residual = x
            x = blk(x)
            x = residual + x
        x = self.norm(x)
        cls_h = x[:, -1]  # take CLS from end
        return self.head(cls_h)
