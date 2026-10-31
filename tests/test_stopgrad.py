"""M1.T4: where the draft loss sends gradient."""

from __future__ import annotations

import pytest
import torch

from tests.test_draft_isolation import VARIANTS, sd_model


def draft_only_grads(model, act_ckpt: bool):
    model.train()
    model.act_ckpt = act_ckpt
    x = torch.randint(0, 256, (2, 33))
    bi = torch.tensor([0, 0, 1, 1, 1])
    ti = torch.tensor([1, 9, 2, 17, 31])
    model.zero_grad()
    model(x[:, :-1], x[:, 1:], draft_sel=(bi, ti))
    # backward the draft KL alone: rebuild it with grad (last_aux is detached)
    from ffnmem.losses import draft_kl_chunk

    logits = model(x[:, :-1]).detach()
    model.zero_grad()
    q = model.draft_logits(x[:, :-1], bi, ti)
    kl = draft_kl_chunk(q, logits[bi, ti])[0]
    kl.backward()
    return {n: p.grad for n, p in model.named_parameters()}


def nonzero(g) -> bool:
    return g is not None and float(g.abs().sum()) > 0


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_draft_gradient_destinations(variant):
    model = sd_model(variant)
    grads = draft_only_grads(model, act_ckpt=False)
    must = ["blocks.1.memory.module.tables.ngram_2.weight", "blocks.1.memory.module.tables.ngram_3.weight",
            "blocks.1.memory.module.readout_kv_proj.weight", "blocks.1.memory.module.readout_memory_norm.weight",
            "draft.w_q.weight", "draft.w_o.weight", "draft.out_norm.weight"]
    if variant == "sda":
        must.append("blocks.1.memory.token_router.proj.weight")
    else:
        must.append("blocks.1.memory.router.weight")
    if variant == "sdc":
        must += ["ssm.w_in.weight", "ssm.theta", "draft.w_sq.weight"]
    for n in must:
        assert nonzero(grads[n]), n
    for n, g in grads.items():
        if ".attn." in n or ".ffn." in n or n.startswith("tok_emb") or n == "norm.weight":
            assert not nonzero(g), n
        if ".readout_q_proj." in n or ".readout_o_proj." in n or ".short_conv." in n:
            assert not nonzero(g), n  # backbone-side readout pieces are not on the draft path


def test_draft_loss_in_training_forward_matches_lambda():
    model = sd_model().train()
    x = torch.randint(0, 256, (2, 33))
    sel = (torch.tensor([0, 1, 1]), torch.tensor([3, 4, 30]))
    _, total = model(x[:, :-1], x[:, 1:], draft_sel=sel)
    aux = model.last_aux
    torch.testing.assert_close(total.detach(), aux["lm"] + 0.1 * aux["draft_kl"])
    # same total gradient with and without activation checkpointing
    grads = []
    for ck in (False, True):
        model.zero_grad()
        model.act_ckpt = ck
        _, total = model(x[:, :-1], x[:, 1:], draft_sel=sel)
        total.backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    for n in grads[0]:
        torch.testing.assert_close(grads[0][n], grads[1][n], rtol=1e-5, atol=1e-7)


def test_sda_route_bce_trains_token_router_only_toward_main_bits():
    model = sd_model("sda").train()
    x = torch.randint(0, 256, (2, 33))
    _, total = model(x[:, :-1], x[:, 1:], draft_sel=(torch.tensor([0]), torch.tensor([5])))
    aux = model.last_aux
    assert "route_bce" in aux
    torch.testing.assert_close(total.detach(), aux["lm"] + 0.1 * aux["draft_kl"] + 0.1 * aux["route_bce"])
