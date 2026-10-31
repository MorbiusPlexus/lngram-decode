"""Post-hoc drafter (RQ4): a token router + draft readout trained on a FROZEN target.

    drafter = PosthocDrafter(target, ssm_dim=0 | 256)
    logits = drafter(idx, b_idx, t_idx)                # draft logits at those positions
    bce = drafter.route_bce(idx, target_route_logits)  # imitation of the target router's bits

Only the drafter's own parameters are trainable / saved: the token router
(embed2 or embed2_ssm features -> R*bits), the optional SSM, and a
``DraftReadout``.  It reads the target's frozen memory table, memory norm + KV
projection and (detached) embeddings; the target is held outside the module tree
so ``state_dict`` / ``parameters`` contain only drafter weights.

Codes for the draft come from the drafter's own router at every position (live
rule: no target activations).  For unmodified targets the offline harness may
substitute the target's exact codes at verified positions (``codes`` argument).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ffnmem.memory.draft import DraftReadout, lookup_rows_at
from ffnmem.memory.router import TokenRouterFeatures
from ffnmem.memory.ssm import DiagonalSSM
from ffnmem.model import Transformer


class PosthocDrafter(nn.Module):
    def __init__(self, target: Transformer, layer: int | None = None, q_heads: int = 4, head_dim: int = 64,
                 ssm_dim: int = 0, ssm_chunk: int = 128):
        super().__init__()
        self._target = [target]  # not a submodule
        mems = [i for i, b in enumerate(target.blocks) if b.memory is not None]
        self.layer = mems[0] if layer is None else int(layer)
        mem = target.blocks[self.layer].memory
        d, eps, std = target.cfg["d_model"], target.cfg["norm_eps"], target.cfg["init_std"]
        self.config = dict(layer=self.layer, q_heads=q_heads, head_dim=head_dim, ssm_dim=ssm_dim, ssm_chunk=ssm_chunk)
        self.ssm = DiagonalSSM(d, ssm_dim, ssm_chunk, eps, std) if ssm_dim else None
        self.features = TokenRouterFeatures(d, "embed2_ssm" if ssm_dim else "embed2", eps, std)
        self.router = nn.Linear(2 * d + ssm_dim, mem.routes * mem.bits, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=std)
        self.draft = DraftReadout(d, mem.module, q_heads, head_dim, ssm_dim, eps, std)

    @property
    def target(self) -> Transformer:
        return self._target[0]

    @property
    def memory_module(self):
        return self.target.blocks[self.layer].memory.module

    def embed(self, idx: Tensor) -> Tensor:
        return self.target.tok_emb(idx).detach()

    def states(self, emb: Tensor) -> Tensor | None:
        return self.ssm(emb) if self.ssm is not None else None

    def route_logits(self, emb: Tensor, s: Tensor | None) -> Tensor:
        x = self.features(emb, s)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.linear(x.float(), self.router.weight.float())

    def codes(self, idx: Tensor) -> Tensor:
        emb = self.embed(idx)
        return self.memory_module.quantize_route_logits(self.route_logits(emb, self.states(emb)))

    def forward(self, idx: Tensor, b_idx: Tensor, t_idx: Tensor, codes: Tensor | None = None) -> Tensor:
        """Draft logits [N, V] at (b_idx, t_idx). ``codes`` [B, T, R] override the drafter's own codes."""
        emb = self.embed(idx)
        s = self.states(emb)
        m = self.memory_module
        if codes is None:
            codes = m.quantize_route_logits(self.route_logits(emb, s))
        rows = lookup_rows_at(m, codes, None, b_idx, t_idx)
        s_sel = s[b_idx, t_idx] if s is not None else None
        return self.draft(m, emb[b_idx, t_idx], rows, self.target.tok_emb.weight.detach(), s_sel)

    def route_bce(self, idx: Tensor, target_route_logits: Tensor) -> Tensor:
        emb = self.embed(idx)
        q = self.route_logits(emb, self.states(emb))
        return F.binary_cross_entropy_with_logits(q, (target_route_logits > 0).float())

    def save(self, path) -> None:
        torch.save({"state": self.state_dict(), "config": self.config}, path)

    @classmethod
    def load(cls, path, target: Transformer) -> "PosthocDrafter":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        d = cls(target, **blob["config"])
        d.load_state_dict(blob["state"])
        return d.to(target.tok_emb.weight.device)


@torch.no_grad()
def target_route_logits(target: Transformer, idx: Tensor, layer: int) -> Tensor:
    """The target's real router logits at memory block ``layer`` (one forward)."""
    mem = target.blocks[layer].memory
    mem.capture_route_logits = True
    try:
        target(idx)
        return mem.last_route_logits
    finally:
        mem.capture_route_logits = False
