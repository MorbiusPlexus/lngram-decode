"""Row layout for serving the memory table from host RAM (M4.T1, gate G1).

Vendored Lngram v2 addresses an n-gram ending at t with the OLDEST code in the
least-significant digit (docs/VERIFIED.md §1):

    v2 row(order n, route r) = r * 16^n + sum_{k=0}^{n-1} c[t-n+1+k] * 16^k

so the 16 rows that step t+1 may read (all values of c[t+1]) are 16 or 256 rows
apart.  ``permute_tables`` reorders each route's block once at load time so the
CURRENT code is least significant:

    serving row = r * 16^n + sum_{k=0}^{n-1} c[t-k] * 16^k

Then the candidates for step t+1 are the contiguous block
``[base, base + 16)`` with ``base = r*16^n + 16 * (c[t] + 16 * c[t-1] + ...)``.
Only the row order changes; ``serving_route_tokens`` gives the same route tokens
as the v2 lookup (tests/test_index_layout.py).
"""

from __future__ import annotations

import torch
from torch import Tensor


def reverse_digits(n_digits: int, bits: int, device=None) -> Tensor:
    """Index map j -> j with its n base-2^bits digits reversed (an involution)."""
    alphabet = 1 << bits
    j = torch.arange(alphabet ** n_digits, device=device, dtype=torch.long)
    out = torch.zeros_like(j)
    for k in range(n_digits):
        digit = (j >> (k * bits)) & (alphabet - 1)
        out |= digit << ((n_digits - 1 - k) * bits)
    return out


def permute_table(weight: Tensor, routes: int, order: int, bits: int) -> Tensor:
    """v2 table [R * 16^n, D] -> serving layout (same shape). Row j of a route block
    holds the v2 row reverse_digits(j)."""
    block = (1 << bits) ** order
    rev = reverse_digits(order, bits, weight.device)
    return weight.view(routes, block, -1)[:, rev].reshape_as(weight)


def permute_tables(module) -> dict[int, Tensor]:
    """Serving-layout copies of every order's table of a ``RouteNgramMemoryV2``."""
    return {order: permute_table(module.tables[f"ngram_{order}"].weight.detach(), module.num_routes,
                                 order, module.bits_per_route)
            for order in module.ngrams}


def serving_address(codes: Tensor, order: int, bits: int) -> Tensor:
    """Addresses [B, T-order+1, R] of the n-grams ending at t = order-1..T-1, current code least significant."""
    b, t, r = codes.shape
    valid = t - order + 1
    if valid <= 0:
        return codes.new_empty(b, 0, r, dtype=torch.long)
    c = codes.to(torch.long)
    addr = c[:, order - 1:, :].clone()  # current code
    for k in range(1, order):
        addr |= c[:, order - 1 - k: order - 1 - k + valid, :] << (k * bits)
    return addr


def serving_route_tokens(tables: dict[int, Tensor], codes: Tensor, ngrams, bits: int) -> Tensor:
    """Route tokens [B, T, R, len(ngrams) * D] from serving-layout tables (zeros where
    the n-gram does not fit), equal to v2's ``_lookup_route_chunk`` without masking."""
    b, t, r = codes.shape
    parts = []
    for order in ngrams:
        w = tables[order]
        block = (1 << bits) ** order
        offs = (torch.arange(r, device=codes.device) * block).view(1, 1, r)
        out = w.new_zeros(b, t, r, w.shape[-1])
        addr = serving_address(codes, order, bits)
        if addr.shape[1]:
            out[:, order - 1:] = w[offs + addr]
        parts.append(out)
    return torch.cat(parts, dim=-1)


def candidate_base(history: Tensor, order: int, bits: int) -> Tensor:
    """First row of the 16 candidates for the NEXT step, per route.

    history: [B, order-1, R] codes, oldest first, ending with the current step's codes
    (the codes c[t-order+2 .. t]).  Returns [B, R] row indices into the serving table;
    the candidates are rows base .. base + 15, indexed by c[t+1].
    """
    b, h, r = history.shape
    if h != order - 1:
        raise ValueError(f"need {order - 1} history codes for order {order}, got {h}")
    block = (1 << bits) ** order
    suffix = torch.zeros(b, r, dtype=torch.long, device=history.device)
    for k in range(order - 1):  # c[t-k] gets digit k+1
        suffix |= history[:, h - 1 - k].to(torch.long) << ((k + 1) * bits)
    return torch.arange(r, device=history.device).view(1, r) * block + suffix


def candidate_rows(tables: dict[int, Tensor], history: Tensor, ngrams, bits: int) -> dict[int, Tensor]:
    """Contiguous 16-row candidate blocks for step t+1: {order: [B, R, 16, D]}.

    ``history`` holds the last max(ngrams)-1 codes [B, H, R] (oldest first)."""
    alphabet = 1 << bits
    out = {}
    for order in ngrams:
        base = candidate_base(history[:, history.shape[1] - (order - 1):], order, bits)
        rows = base[..., None] + torch.arange(alphabet, device=base.device)
        out[order] = tables[order][rows]
    return out


def select_candidates(cands: dict[int, Tensor], next_codes: Tensor, ngrams) -> Tensor:
    """Pick the rows for the actual next codes [B, R] from candidate blocks -> [B, R, len(ngrams) * D]."""
    idx = next_codes.to(torch.long)[..., None, None]
    parts = [cands[o].gather(2, idx.expand(-1, -1, 1, cands[o].shape[-1]))[:, :, 0] for o in ngrams]
    return torch.cat(parts, dim=-1)
