# Verification: `train_mini_s4d.py` vs Official S4D

Comparison of `s4d/train_mini_s4d.py` with the official [state-spaces/s4](https://github.com/state-spaces/s4) implementation under `models/s4/s4d.py`.

## Summary

**Verdict: The mini is a faithful copy of the S4D idea.** The kernel math matches (diagonal SSM + ZOH + same recurrence). The block now includes D (skip), dropout, and output_linear (Conv1d + GLU) to match the official S4D.

---

## 1. S4DKernel – Kernel generation

### What matches (faithful)

| Aspect | Official (`s4d.py`) | Mini (`train_mini_s4d.py`) |
|--------|----------------------|----------------------------|
| **dt** | Log-uniform in `[dt_min, dt_max]` | Same |
| **A** | Diagonal: `-exp(log_A_real) + 1j * A_imag`, with `A_imag = π * arange(N)` | Same |
| **ZOH** | Uses `(exp(dt*A)-1)/A` folded into C | Explicit `A_bar = exp(A*dt)`, `B_bar = (A_bar-1)/A * B` → same recurrence |
| **Kernel** | `K[l] ∝ Σ_n C'_n · A_bar_n^l` (C' = C·(A_bar−1)/A) | `K[l] = Σ_n C_n · A_bar_n^l · B_bar_n` → same when B=1 |

### Intentional differences

- **State size:** Official uses `N//2` + `2*...real` for conjugate pairs; mini uses full `d_state` and `.real`.
- **B:** Official folds B=1 into C; mini has explicit learnable B.
- **Parameter registration:** Official uses `register(..., lr=...)` for SSM params; mini uses plain `nn.Parameter`.

---

## 2. Convolution

- **Official:** FFT-based: `y = irfft(rfft(u) * rfft(k))[..., :L]`.
- **Mini:** Causal conv1d: `conv1d(u, k, padding=L-1)[..., :L]`.

Same causal convolution; FFT vs conv1d is an implementation choice.

---

## 3. S4D block (MiniS4D)

The **current** MiniS4D block matches the official S4D:

- **D (skip):** `y = y + u * self.D.unsqueeze(-1)`
- **GELU** then **dropout** (optional via `dropout=0.0`)
- **output_linear:** `Conv1d(d_model, 2*d_model, 1)` + GLU
- Plus a **decoder** (linear head) for the demo regression task

An earlier minimal variant dropped D, dropout, and output_linear for FHE/minimal use; see README.

---

## 4. Key differences (kernel, convolution, head/pooling)

**Kernel (S4DKernel):**
- **Official:** uses `N//2` conjugate pairs and `2 * ... .real` in the Vandermonde/exp formulation.
- **Mini:** uses full `d_state` and returns `.real` directly; equivalent ZOH kernel, but parameterized without conjugate pairing.

**Convolution:**
- **Official:** FFT‑based convolution (`rfft/irfft`) for speed at long lengths.
- **Mini:** direct causal `conv1d` (simpler, slower at long lengths).

**Head / pooling:**
- **Official:** no built‑in head; pooling + classifier live in the outer model.
- **Mini:** mean‑pool over time with a built‑in `Linear(d_model→1)` head for the demo.

---

## 5. Dependencies

- **Official:** `einops`, `src.models.nn.DropoutNd`.
- **Mini:** only `torch`, `numpy`. Self-contained.

---

## 6. Conclusion

- **Kernel:** Faithful (same diagonal SSM, ZOH, kernel recurrence).
- **Convolution:** Faithful (causal; FFT vs conv1d).
- **Block:** Faithful (D, dropout, GELU, output_linear). README documents the minimal/FHE variant that omitted these.
