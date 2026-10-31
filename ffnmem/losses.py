"""Wave 3 draft loss pieces (M1.T4).

    L = L_LM + lambda * KL(sg(p) || q)  [+ mu * BCE(token router, sign(main router)) for SD-A]

p is the model's next-token distribution (detached), q the draft distribution.
The KL is computed on a random ``loss_position_frac`` of positions per sequence,
in chunks of ``loss_chunk`` positions, softmax/KL in fp32.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def draft_kl_chunk(q_logits: Tensor, p_logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Sums over the chunk's positions: KL(p || q) (with grad into q), top-1 agreement
    count and sum_v min(p, q) (both detached).  Inputs [n, V], any float dtype."""
    lp = F.log_softmax(p_logits.float(), dim=-1)
    lq = F.log_softmax(q_logits.float(), dim=-1)
    pp = lp.exp()
    kl = (pp * (lp - lq)).sum()
    with torch.no_grad():
        agree = (p_logits.argmax(-1) == q_logits.argmax(-1)).sum().float()
        accept = torch.minimum(pp, lq.exp()).sum()
    return kl, agree, accept


def position_seed(data_seed: int, step: int, window_id: int) -> int:
    """Per-sequence seed: depends on the optimizer step and the global window id only,
    so the chosen positions do not depend on GPU count, micro-batch or resume."""
    return int(np.random.SeedSequence([int(data_seed), int(step), int(window_id)]).generate_state(1)[0])


def select_positions(seeds: list[int], seq_len: int, frac: float, device: torch.device) -> tuple[Tensor, Tensor]:
    """``round(frac * seq_len)`` distinct positions per sequence, sorted. Returns (b_idx, t_idx)."""
    k = max(1, round(frac * seq_len))
    b_parts, t_parts = [], []
    for b, seed in enumerate(seeds):
        g = torch.Generator().manual_seed(seed)
        t = torch.randperm(seq_len, generator=g)[:k].sort().values
        t_parts.append(t)
        b_parts.append(torch.full((k,), b, dtype=torch.long))
    return torch.cat(b_parts).to(device), torch.cat(t_parts).to(device)
