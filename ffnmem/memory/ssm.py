"""Non-selective diagonal SSM over token embeddings (SD-C).

    u[t] = W_in RMSNorm(E[x[t]])            W_in: d_model -> state_dim
    a    = sigmoid(theta)                    per channel, init decays span 0.5 .. 0.999
    s[t] = a * s[t-1] + (1 - a) * u[t],      s[-1] = 0

Pure PyTorch, fp32, no custom kernels (runs on sm_70/75 and on the CPU drafter).
Training uses a chunked parallel scan: inside a chunk of L positions the state is
a matmul with the [L, L] decay matrix a^(t-k) (shared by every chunk), and the
carried state crosses chunk boundaries sequentially (T / L steps).  Inference
uses ``step`` (the plain recurrence).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class DiagonalSSM(nn.Module):
    def __init__(self, d_model: int, state_dim: int = 256, chunk: int = 128, eps: float = 1e-5,
                 init_std: float = 0.02, decay_min: float = 0.5, decay_max: float = 0.999):
        super().__init__()
        from ..model import RMSNorm  # model imports memory modules; avoid the cycle at import time

        self.state_dim, self.chunk = int(state_dim), int(chunk)
        self.norm = RMSNorm(d_model, eps)
        self.w_in = nn.Linear(d_model, self.state_dim, bias=False)
        nn.init.normal_(self.w_in.weight, mean=0.0, std=init_std)
        # Geometric spacing of the time constants 1/(1-a) between the two decays.
        one_minus = torch.exp(torch.linspace(math.log(1 - decay_min), math.log(1 - decay_max), self.state_dim))
        a = 1.0 - one_minus
        self.theta = nn.Parameter(torch.log(a) - torch.log1p(-a))  # logit(a)

    def decay(self) -> Tensor:
        return torch.sigmoid(self.theta.float())

    def inputs(self, emb: Tensor) -> Tensor:
        """u[t] in fp32: [B, T, d_model] -> [B, T, N]."""
        with torch.autocast(device_type=emb.device.type, enabled=False):
            return nn.functional.linear(self.norm(emb.float()), self.w_in.weight.float())

    def forward(self, emb: Tensor) -> Tensor:
        """States s[0..T-1]: [B, T, N] fp32."""
        return chunked_scan(self.decay(), self.inputs(emb), self.chunk)

    def step(self, state: Tensor | None, emb_t: Tensor) -> Tensor:
        """One recurrence step: emb_t [B, d_model], state [B, N] or None -> new state."""
        u = self.inputs(emb_t[:, None])[:, 0]
        a = self.decay()
        if state is None:
            state = torch.zeros_like(u)
        return a * state + (1 - a) * u


def sequential_scan(a: Tensor, u: Tensor) -> Tensor:
    """Reference loop: a [N], u [B, T, N] -> s [B, T, N]."""
    s = torch.zeros_like(u[:, 0])
    out = []
    for t in range(u.shape[1]):
        s = a * s + (1 - a) * u[:, t]
        out.append(s)
    return torch.stack(out, 1)


def chunked_scan(a: Tensor, u: Tensor, chunk: int = 128) -> Tensor:
    """Same result as ``sequential_scan`` (fp32), parallel inside each chunk."""
    b, t, n = u.shape
    a, u = a.float(), u.float()
    L = min(chunk, t)
    pad = (-t) % L
    if pad:
        u = torch.nn.functional.pad(u, (0, 0, 0, pad))
    c = u.shape[1] // L
    x = ((1 - a) * u).view(b, c, L, n)
    idx = torch.arange(L, device=u.device)
    diff = (idx[:, None] - idx[None, :]).to(torch.float32)  # t - k
    log_a = torch.log(a)
    # decay[t, k, n] = a^(t-k) for k <= t else 0; exponent clamped so masked entries stay finite
    decay = torch.exp(diff.clamp_min(0)[..., None] * log_a) * (diff >= 0)[..., None]
    local = torch.einsum("tkn,bckn->bctn", decay, x)  # zero-initial-state scan of each chunk
    carry_scale = torch.exp((idx + 1).to(torch.float32)[:, None] * log_a)  # a^(t+1), [L, N]
    outs = []
    state = u.new_zeros(b, n)
    for ci in range(c):
        out = local[:, ci] + carry_scale[None] * state[:, None]
        outs.append(out)
        state = out[:, -1]
    return torch.cat(outs, 1)[:, :t]
