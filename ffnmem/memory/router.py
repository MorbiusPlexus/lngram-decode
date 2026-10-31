"""Router inputs that depend only on tokens (Wave 3 self-drafting).

``router_input`` (memory config key, default ``hidden``):

- ``hidden``     : v2 / Wave 1 behaviour -- the router reads the block's pre-attention RMSNorm output.
- ``embed2``     : concat(RMSNorm(E[x[t-1]]), RMSNorm(E[x[t]]))              -> 2 * d_model
- ``embed2_ssm`` : concat(RMSNorm(E[x[t-1]]), RMSNorm(E[x[t]]), s[t])        -> 2 * d_model + state_dim

For t = 0 a learned start vector stands in for E[x[-1]] (no document-boundary
masking, as in Wave 1).  A router Linear over the concatenation is the same as
``W_e . embeddings + W_s . s[t]`` in the plan.

The codes of ``embed2`` depend only on (x[t-1], x[t]); with the SSM they also
depend on the whole prefix through s[t], but never on backbone activations.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

ROUTER_INPUTS = ("hidden", "embed2", "embed2_ssm")


def router_input_dim(kind: str, d_model: int, ssm_dim: int) -> int:
    if kind == "hidden":
        return d_model
    if kind == "embed2":
        return 2 * d_model
    if kind == "embed2_ssm":
        if ssm_dim <= 0:
            raise ValueError("router_input embed2_ssm needs model.ssm")
        return 2 * d_model + ssm_dim
    raise ValueError(f"router_input must be one of {ROUTER_INPUTS}, got {kind!r}")


class TokenRouterFeatures(nn.Module):
    def __init__(self, d_model: int, kind: str, eps: float = 1e-5, init_std: float = 0.02):
        super().__init__()
        from ..model import RMSNorm

        if kind not in ("embed2", "embed2_ssm"):
            raise ValueError(f"TokenRouterFeatures does not handle {kind!r}")
        self.kind = kind
        self.prev_norm = RMSNorm(d_model, eps)
        self.cur_norm = RMSNorm(d_model, eps)
        self.start = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.start, mean=0.0, std=init_std)

    def forward(self, emb: Tensor, ssm_state: Tensor | None = None) -> Tensor:
        """emb: [B, T, d] token embeddings (E[x]); ssm_state: [B, T, N] for embed2_ssm."""
        b = emb.shape[0]
        prev = torch.cat((self.start.to(emb.dtype).expand(b, 1, -1), emb[:, :-1]), dim=1)
        parts = [self.prev_norm(prev), self.cur_norm(emb)]
        if self.kind == "embed2_ssm":
            if ssm_state is None:
                raise ValueError("embed2_ssm router needs the SSM state")
            parts.append(ssm_state.to(parts[0].dtype))
        return torch.cat(parts, dim=-1)

    def step(self, emb_prev: Tensor | None, emb_cur: Tensor, ssm_state: Tensor | None = None) -> Tensor:
        """Single position (decode / CPU drafter): emb_prev None at t = 0. Shapes [B, d]."""
        if emb_prev is None:
            emb_prev = self.start.to(emb_cur.dtype).expand_as(emb_cur)
        parts = [self.prev_norm(emb_prev), self.cur_norm(emb_cur)]
        if self.kind == "embed2_ssm":
            parts.append(ssm_state.to(parts[0].dtype))
        return torch.cat(parts, dim=-1)


class TokenRouter(nn.Module):
    """SD-A imitating router: embed2 features -> R*bits logits (fp32), trained by BCE
    against the main (hidden-state) router's bits."""

    def __init__(self, d_model: int, out_dim: int, eps: float = 1e-5, init_std: float = 0.02):
        super().__init__()
        self.features = TokenRouterFeatures(d_model, "embed2", eps, init_std)
        self.proj = nn.Linear(2 * d_model, out_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=init_std)

    def forward(self, emb: Tensor) -> Tensor:
        x = self.features(emb)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return nn.functional.linear(x.float(), self.proj.weight.float())
