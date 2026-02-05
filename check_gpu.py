#!/usr/bin/env python3
"""Quick script to check which GPU is being used."""
import torch
import os

print("=" * 60)
print("GPU Information")
print("=" * 60)

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set (all GPUs visible)')}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create a tensor on GPU to verify it works
    x = torch.randn(10, 10).cuda()
    print(f"✓ Successfully created tensor on GPU: {x.device}")
else:
    print("CUDA is not available. Running on CPU.")

print("=" * 60)

