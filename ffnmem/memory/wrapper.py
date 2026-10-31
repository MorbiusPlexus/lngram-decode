"""Adapter around the vendored Lngram v2 module.

The vendored ``RouteNgramMemoryV2`` does not own the router: the caller supplies
route logits and codes.  This wrapper owns the router (``d_model -> R * bits``),
mirrors the v1 integration (router reads the pre-attention RMSNorm output, the
memory module reads the raw residual stream, output is added to the residual
before attention), applies the Lngram v2 Appendix E initialisation, keeps the
routing logits in fp32, and exposes diagnostics.

Document-boundary masking is deliberately off: training windows are plain
contiguous chunks, so ``segment_ids`` is always ``None`` (one segment/window).

Wave 3 options (absent -> exactly the Wave 1 module):
- ``router_input: embed2 | embed2_ssm`` -- the router reads token-only features
  (``router.TokenRouterFeatures``), passed in by the model as ``router_in``.
- ``token_router: {enabled, loss_weight}`` -- SD-A imitating router; the main router's
  logits of the last forward are kept (detached) in ``last_route_logits`` for its BCE.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .lngram_v2 import RouteNgramMemoryV2, _aligned_ngram_address
from .router import TokenRouter, TokenRouterFeatures, router_input_dim

# Wrapper-level keys (never passed to the vendored module).
WRAPPER_KEYS = ("routes", "mem_dim", "kv_heads", "q_heads", "head_dim", "router_init_std", "rank1",
                "router_input", "token_router")

# Appendix E recipe (Lngram v2) -- overrides the code defaults, which are the
# opposite initialisation (table std 0, output projection std 1).
APPENDIX_E = dict(
    bits_per_route=4,
    ngrams=(2, 3),
    table_init_std_scale=1.0,
    output_proj_init_std_scale=0.0,
    conv_zero_init=True,
    conv_kernel_size=4,
    q_surrogate_enable=True,
    q_surrogate_mode="approximate",
    sink_initial_active_mass=0.5,
)


class LngramMemory(nn.Module):
    def __init__(self, d_model: int, mem_cfg: dict, init_std: float = 0.02, norm_eps: float = 1e-5,
                 ssm_dim: int = 0):
        super().__init__()
        cfg = dict(APPENDIX_E)
        cfg.update({k: v for k, v in mem_cfg.items() if k not in WRAPPER_KEYS})
        self.routes = int(mem_cfg["routes"])
        self.bits = int(cfg["bits_per_route"])
        head_dim = int(mem_cfg.get("head_dim", 64))
        q_heads = int(mem_cfg.get("q_heads", d_model // head_dim))
        if "ngrams" in cfg:
            cfg["ngrams"] = tuple(int(n) for n in cfg["ngrams"])
        self.module = RouteNgramMemoryV2(
            hidden_size=d_model,
            num_routes=self.routes,
            memory_dim=int(mem_cfg["mem_dim"]),
            readout_num_heads=q_heads,
            readout_num_kv_heads=int(mem_cfg.get("kv_heads", 4)),
            readout_head_dim=head_dim,
            initializer_range=init_std,
            rmsnorm_eps=norm_eps,
            **cfg,
        )
        self.router_input = str(mem_cfg.get("router_input", "hidden"))
        in_dim = router_input_dim(self.router_input, d_model, ssm_dim)
        self.router = nn.Linear(in_dim, self.routes * self.bits, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=float(mem_cfg.get("router_init_std", init_std)))
        if self.router_input != "hidden":
            self.router_features = TokenRouterFeatures(d_model, self.router_input, norm_eps, init_std)
        tr = mem_cfg.get("token_router") or {}
        self.cfg_token_router = dict(tr)
        self.token_router = None
        if tr.get("enabled", False):
            if self.router_input != "hidden":
                raise ValueError("token_router (SD-A) imitates a hidden-state router; use router_input: hidden")
            self.token_router = TokenRouter(d_model, self.routes * self.bits, norm_eps, init_std)
        self.last_route_logits: Tensor | None = None
        self.capture_route_logits = False  # analysis: keep the last forward's router logits
        # Optional per-entry rank-1 path (idea A), added to the v2 readout. Absent -> exact v2.
        self.rank1_dim = 0
        r1 = mem_cfg.get("rank1")
        if r1:
            self._build_rank1(d_model, int(r1["dim"]), init_std)
        # Diagnostics (filled only while ``collect_diagnostics`` is True, in eval).
        self.collect_diagnostics = False
        self.reset_diagnostics()

    # ------------------------------------------------------------------ forward
    def route_logits(self, h_norm: Tensor) -> Tensor:
        """Router in fp32 regardless of autocast (sign thresholding is precision-sensitive)."""
        with torch.autocast(device_type=h_norm.device.type, enabled=False):
            return nn.functional.linear(h_norm.float(), self.router.weight.float())

    def forward(self, h: Tensor, h_norm: Tensor, router_in: Tensor | None = None) -> Tensor:
        if self.router_input == "hidden":
            q_logits = self.route_logits(h_norm)
        else:
            if router_in is None:
                raise ValueError(f"router_input={self.router_input} needs router_in from the model")
            q_logits = self.route_logits(router_in)
        if self.token_router is not None or self.capture_route_logits:
            self.last_route_logits = q_logits.detach()
        codes = self.module.quantize_route_logits(q_logits)
        if self.collect_diagnostics and not self.training:
            self._accumulate_diagnostics(h, codes)
        out = self.module(h, codes, q_logits=q_logits if self.training else None)
        if self.rank1_dim:
            extra = self._rank1(h_norm, codes)
            if self.collect_diagnostics and not self.training:
                with torch.no_grad():
                    hn = h.float().norm(dim=-1).clamp_min(1e-6)
                    self._r1_ratio_sum += float((extra.float().norm(dim=-1) / hn).mean())
                    self._v2_ratio_sum += float((out.float().norm(dim=-1) / hn).mean())
                    self._r1_count += 1
            out = out + extra
        return out

    # ------------------------------------------------------- rank-1 path (idea A)
    def _build_rank1(self, d_model: int, dim: int, init_std: float) -> None:
        """Each table row also stores (w_i, u_i) in R^dim x R^dim: a per-entry rank-1 map.

        out = U_out( sum_{routes, orders} u_i * SiLU(w_i . P_in h_norm) / sqrt(R * n_orders) )
        Same n-gram addresses as the v2 tables. U_out is zero-initialised, so at init the
        layer output equals the v2 readout. The rank-1 tables get no router gradient (hard
        lookup); the router keeps training through the v2 surrogate.
        """
        m = self.module
        self.rank1_dim = dim
        self.rank1_in = nn.Linear(d_model, dim, bias=False)
        nn.init.normal_(self.rank1_in.weight, mean=0.0, std=init_std)
        self.rank1_out = nn.Linear(dim, d_model, bias=False)
        nn.init.zeros_(self.rank1_out.weight)
        self.rank1_tables = nn.ModuleDict()
        for order in m.ngrams:
            rows = self.routes * (m.alphabet_size ** order)
            table = nn.Embedding(rows, 2 * dim)
            nn.init.normal_(table.weight, mean=0.0, std=1.0 / math.sqrt(dim))
            self.rank1_tables[f"ngram_{order}"] = table

    def _rank1(self, h_norm: Tensor, codes: Tensor) -> Tensor:
        m = self.module
        x = self.rank1_in(h_norm)  # [B, T, dim]
        t = x.shape[1]
        codes = codes.to(m.index_dtype)
        acc = None
        for order in m.ngrams:
            valid = t - order + 1
            if valid <= 0:
                continue
            address = _aligned_ngram_address(codes, order, self.bits)  # [B, valid, R]
            offsets = (torch.arange(self.routes, device=codes.device, dtype=m.index_dtype)
                       * (m.alphabet_size ** order)).view(1, 1, -1)
            rows = self.rank1_tables[f"ngram_{order}"](offsets + address)  # [B, valid, R, 2*dim]
            w, u = rows.split(self.rank1_dim, dim=-1)
            xs = x[:, order - 1:, None, :].to(w.dtype)  # entry ending at token t reads x_t
            gate = F.silu((w * xs).sum(-1, keepdim=True))  # [B, valid, R, 1]
            c = F.pad((gate * u).sum(2), (0, 0, order - 1, 0))  # [B, T, dim]
            acc = c if acc is None else acc + c
        if acc is None:
            return torch.zeros_like(h_norm)
        acc = acc / math.sqrt(self.routes * len(m.ngrams))
        return self.rank1_out(acc.to(x.dtype))

    # -------------------------------------------------------------- diagnostics
    def reset_diagnostics(self) -> None:
        self._code_hist = None  # [R, 2^bits] counts
        self._sink_sum = 0.0
        self._sink_count = 0
        self._r1_ratio_sum = 0.0
        self._v2_ratio_sum = 0.0
        self._r1_count = 0

    @torch.no_grad()
    def _accumulate_diagnostics(self, h: Tensor, codes: Tensor) -> None:
        m = self.module
        n_codes = 1 << self.bits
        flat = codes.reshape(-1, self.routes).long()
        onehot_counts = torch.zeros(self.routes, n_codes, device=codes.device, dtype=torch.float64)
        onehot_counts.scatter_add_(
            1, flat.t(), torch.ones_like(flat.t(), dtype=torch.float64)
        )
        self._code_hist = onehot_counts if self._code_hist is None else self._code_hist + onehot_counts
        # Sink mass: recompute the readout scores (same path as the module's full mode).
        query = m._project_query(h)
        tokens = m._lookup_route_chunk(codes, None, 0, m.num_routes, None, None)
        key, _ = m._project_memory(tokens)
        b, t = query.shape[:2]
        grouped = query.view(b, t, m.readout_num_kv_heads, m.readout_groups, m.readout_head_dim)
        scores = torch.matmul(grouped, key.transpose(-1, -2)).float() * m.softmax_scale
        real = m._real_route_probabilities(scores)  # [B,T,Hkv,G,R]
        sink = 1.0 - real.sum(-1)
        self._sink_sum += float(sink.sum())
        self._sink_count += sink.numel()

    def diagnostics(self) -> dict:
        out: dict = {}
        if self._sink_count:
            out["sink_mass"] = self._sink_sum / self._sink_count
        if self._code_hist is not None:
            p = self._code_hist / self._code_hist.sum(-1, keepdim=True).clamp_min(1)
            entropy = -(p * p.clamp_min(1e-30).log()).sum(-1)
            eff = entropy.exp()  # effective number of codes per route
            out["eff_codes_mean"] = float(eff.mean())
            out["frac_routes_eff_lt4"] = float((eff < 4).double().mean())
            out["frac_routes_eff_lt2"] = float((eff < 2).double().mean())
            out["dead_code_frac"] = float((self._code_hist == 0).double().mean())
            out["top_code_share"] = float(p.max(-1).values.mean())
        if self._r1_count:
            out["rank1_out_to_residual"] = self._r1_ratio_sum / self._r1_count
            out["v2_out_to_residual"] = self._v2_ratio_sum / self._r1_count
        return out

    # ------------------------------------------------------------ bookkeeping
    @property
    def sink_initial_mass(self) -> float:
        # The module's sink logit gives (1 - mass) to the sink at zero real logits.
        return 1.0 - self.module.sink_initial_active_mass

    def table_parameters(self):
        params = list(self.module.tables.parameters())
        if self.rank1_dim:
            params += list(self.rank1_tables.parameters())
        return params

    def active_table_params_per_token(self) -> int:
        active = self.routes * len(self.module.ngrams) * self.module.memory_dim
        if self.rank1_dim:
            active += self.routes * len(self.module.ngrams) * 2 * self.rank1_dim
        return active


def param_group_of(name: str, param: nn.Parameter) -> str:
    """Optimizer group for a parameter name of the full model.

    tables -> 4x LR, wd 0; router -> base LR, wd 0; other memory params (readout
    Q/KV/O, conv) -> base LR, wd 0.1; backbone matrices -> wd 0.1; every 1-D
    parameter (norm weights) -> wd 0.
    """
    if ".memory.module.tables." in name or ".memory.rank1_tables." in name:
        return "tables"
    if ".memory.router." in name or ".memory.token_router.proj." in name:
        return "router"
    if param.ndim < 2:
        return "no_decay"
    return "decay"

