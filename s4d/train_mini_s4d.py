import torch
import torch.nn as nn
import numpy as np

class S4DKernel(nn.Module):
    """
    100% Faithful S4D Kernel Generation.
    Uses ZOH discretization and standard S4D initialization.
    """
    def __init__(self, d_model, d_state=8, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.N = d_state

        # 1. Initialize dt (Timescale)
        # Log-uniform initialization (Standard S4)
        log_dt = torch.rand(d_model) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # 2. Initialize A (State Matrix) - S4D-Lin Initialization
        # A is diagonal. Real part is -0.5, Imag part is spaced by pi.
        # This is the "HiPPO" approximation for diagonal matrices.
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))
        self.A_imag = nn.Parameter(torch.pi * torch.arange(d_state).float().repeat(d_model, 1))

        # 3. Initialize B (Input Projection)
        # Standard S4D fixes B to ones (redundant with C), but we make it learnable to be rigorous
        self.B = nn.Parameter(torch.ones(d_model, d_state, dtype=torch.cfloat))

        # 4. Initialize C (Output Projection)
        # Random normal initialization
        self.C = nn.Parameter(torch.randn(d_model, d_state, dtype=torch.cfloat))

    def forward(self, L):
        """
        Returns the Discrete Convolution Kernel (K) using ZOH.
        Shape: (d_model, L)
        """
        # Materialize parameters
        dt = torch.exp(self.log_dt)          # (H)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H, N)
        
        # --- The Faithful ZOH Discretization ---
        # A_bar = exp(A * dt)
        # B_bar = (A_bar - I) * A^{-1} * B * dt (approx) or exact ZOH integration
        
        # Since A is diagonal, we can do element-wise ops.
        # dt must be broadcast to (H, N)
        dt_broad = dt.unsqueeze(-1)
        
        # 1. Discretize A -> A_bar (State transition)
        A_bar = torch.exp(A * dt_broad)  # (H, N)

        # 2. Discretize B -> B_bar (Input mixing)
        # Exact ZOH: B_bar = A^{-1} (exp(A*dt) - I) * B
        B_bar = (1/A) * (A_bar - 1.0) * self.B # (H, N)

        # --- Compute Kernel (Power Series) ---
        # K = [C B_bar, C A_bar B_bar, C A_bar^2 B_bar, ...]
        # We compute this explicitly for small L (Faithful to the math, distinct from FFT optimization)
        
        # Create vandermonde-like powers: A_bar^t
        # Range t = [0, ..., L-1]
        t = torch.arange(L, device=A.device).unsqueeze(0).unsqueeze(0) # (1, 1, L)
        A_bar_pow = torch.pow(A_bar.unsqueeze(-1), t) # (H, N, L)
        
        # Combine: C * (A_bar^t) * B_bar
        # Sum over state dimension N (since y = C x)
        # Shape: (H, N, L) -> (H, L)
        K = (self.C.unsqueeze(-1) * A_bar_pow * B_bar.unsqueeze(-1)).sum(dim=1)
        
        return K.real # Return real part (SISO system assumption)

class MiniS4D(nn.Module):
    """
    S4D block faithful to official state-spaces/s4: includes D (skip), dropout,
    and output_linear (Conv1d + GLU). For a minimal/FHE-friendly variant that
    omits these, see README.
    """
    def __init__(self, d_model=4, d_state=8, L=64, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.L = L

        # D term (skip connection) — official S4D
        self.D = nn.Parameter(torch.randn(d_model))

        # Core S4 Layer
        self.kernel_gen = S4DKernel(d_model, d_state)

        # Pointwise operations (match official S4D block)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        # position-wise output transform: Conv1d + GLU (official output_linear)
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=-2),
        )
        self.decoder = nn.Linear(d_model, 1)  # Simple regression head for demo

    def forward(self, u):
        """
        u: (Batch, d_model, L)
        """
        # 1. Generate Kernel (ZOH based)
        K = self.kernel_gen(self.L)  # (d_model, L)

        # 2. Convolution (causal)
        k_conv = K.unsqueeze(1)  # (H, 1, L)
        y = torch.nn.functional.conv1d(u, k_conv, padding=self.L - 1, groups=self.d_model)
        y = y[:, :, : self.L]  # Causal slice

        # 3. D term (skip connection) — official S4D
        y = y + u * self.D.unsqueeze(-1)

        # 4. Activation + dropout + output_linear (Conv1d + GLU)
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)  # (B, d_model, L)

        # Pooling / prediction for demo
        return self.decoder(y.mean(dim=-1))

    def export_toeplitz(self, head_idx=0):
        """
        Exports the EXACT kernel matrix for FHE usage.
        """
        with torch.no_grad():
            K = self.kernel_gen(self.L)
            k_np = K[head_idx].cpu().numpy()
            
            # Construct Toeplitz
            T = np.zeros((self.L, self.L))
            for i in range(self.L):
                for j in range(i + 1):
                    T[i, j] = k_np[i-j]
            return T


if __name__ == "__main__":
    # CPU is enough for this tiny model; GPU used automatically if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MiniS4D(d_model=4, d_state=8, L=64, dropout=0.1).to(device)
    u = torch.randn(4, 4, 64, device=device)  # (B, d_model, L)

    # Forward
    out = model(u)
    print(f"Input shape: {u.shape} -> Output shape: {out.shape}")

    # Optional: a few training steps on random targets
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(3):
        opt.zero_grad()
        out = model(u)
        loss = out.squeeze(-1).pow(2).mean()  # dummy loss
        loss.backward()
        opt.step()
        print(f"Step {step + 1}, loss: {loss.item():.4f}")
    print("Done.")