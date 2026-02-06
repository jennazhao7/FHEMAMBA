# Mini S4D — Faithful copy of [state-spaces/s4](https://github.com/state-spaces/s4) S4D

Self-contained S4D implementation aligned with the official **S4D** in `models/s4/s4d.py`: same kernel (ZOH diagonal SSM), same causal convolution, and the **full block** (D skip, dropout, output_linear).

## Block: full vs minimal/FHE-friendly

The **current** `MiniS4D` block matches the official S4D and includes:

- **D (skip)** — `y = y + u * D` (learnable per-channel skip).
- **Dropout** — after GELU (constructor: `dropout=0.0` to disable).
- **output_linear** — `Conv1d(d_model, 2*d_model, 1)` + GLU, i.e. position-wise mix after activation.

An **earlier minimal variant** intentionally dropped D, dropout, and `output_linear` for a smaller, FHE-friendly stack (kernel + conv + GELU + single linear head only). That choice was for pedagogy and FHE export, not for parity with the official block. The codebase now uses the **full block** by default; see `VERIFICATION_vs_official_s4d.md` for a detailed comparison with the official repo.

## How to run

From the `s4d` directory (with PyTorch installed):

```bash
cd s4d
python train_mini_s4d.py
```

This runs a forward pass and a few dummy training steps. **You do not need a GPU** — the default model is tiny (d_model=4, L=64) and runs fine on CPU. If a GPU is available, the script will use it automatically.

## Files

- **`train_mini_s4d.py`** — `S4DKernel` (ZOH kernel) + `MiniS4D` (full block + regression decoder).
- **`VERIFICATION_vs_official_s4d.md`** — Line-by-line verification against official S4D.

## Mini vs Original (structure & sizes)

**MiniS4D in this repo (`train_mini_s4d.py`):**

- **Blocks:** 1 S4D block (no stacking).
- **Default sizes:** `d_model=4`, `d_state=8`, `L=64`, `dropout=0.0`.
- **Block internals:** ZOH S4D kernel → causal `conv1d` → D skip → GELU + dropout → `Conv1d + GLU` → mean pool → `Linear(d_model→1)` head.

**Original S4D (`state-spaces/s4` `models/s4/s4d.py`):**

- **Blocks:** 1 S4D block class intended to be stacked (e.g., 4–6 layers in practice).
- **Typical sizes:** `d_model` usually 128–512, `d_state` usually 64 (depends on experiment).
- **Block internals:** ZOH S4D kernel → FFT convolution → D skip → GELU + dropout → `Conv1d + GLU`. No built‑in head (pooling + classifier are done by the outer model).

## Usage (in code)

```python
from train_mini_s4d import MiniS4D

model = MiniS4D(d_model=4, d_state=8, L=64, dropout=0.1)
u = torch.randn(2, 4, 64)  # (B, d_model, L)
out = model(u)  # (B, 1)
```

To disable dropout: `MiniS4D(..., dropout=0.0)`.

---

## Adding Problem (long-range benchmark, LRA-style)

The **adding problem** is a standard long-range dependency benchmark: the model sees a sequence of (value, mask) pairs; exactly two positions are marked (mask=1); the target is the **sum of the two values** at those positions. Solving it requires finding the two positions across the sequence (long-range).

Setup follows the **Long Range Arena** spirit: fixed train/val/test seeds, long sequences (default `seq_len=1000`), MSE loss, and accuracy = fraction of samples with `|pred − target| < tolerance` (default tolerance `0.04`). (Note: the official LRA suite has ListOps, Text, Retrieval, Image, Pathfinder, Path-X; the adding problem is a classic synthetic benchmark often used alongside LRA-style evaluations.)

### Train from scratch on adding problem

```bash
cd s4d
python adding_problem.py --seq_len 1000 --d_model 64 --d_state 64 --epochs 20
```

Best model is saved as `s4d_adding_best.pt`. Shorter runs for debugging:

```bash
python adding_problem.py --seq_len 100 --train_samples 500 --epochs 5 --d_model 32
```

### Test a trained model (eval only)

- **Full AddingModel checkpoint** (encoder + S4D, saved by this script):

  ```bash
  python adding_problem.py --eval_only --ckpt s4d_adding_best.pt --seq_len 1000 --d_model 64 --d_state 64
  ```

- **Pretrained MiniS4D only** (e.g. from `train_mini_s4d.py` or your own training):  
  Pass the same `--ckpt`; the script loads the state dict into `model.s4d` and keeps the encoder randomly initialized. You can then either train (no `--eval_only`) to learn the encoder and refine the block, or run `--eval_only` to see baseline performance with a frozen S4D.

All options: `--seq_len`, `--d_model`, `--d_state`, `--train_samples`, `--batch_size`, `--epochs`, `--lr`, `--tolerance`, `--ckpt`, `--eval_only`.
