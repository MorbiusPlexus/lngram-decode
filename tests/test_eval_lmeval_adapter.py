"""lm-eval adapter: scores equal direct sums of target log-probs from ``model(idx)``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from eval.lm_eval_adapter import EOS_ID, MorbiusScorer, compact_sample, write_outputs
from ffnmem import checkpoint as ckpt
from ffnmem.config import load_config
from ffnmem.model import Transformer

TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny_cpu.yaml"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Transformer(load_config(TINY)["model"]).eval()  # vocab 256, seq_len 32, one memory layer


def direct(model, ctx: list[int], cont: list[int]) -> tuple[float, bool]:
    full = torch.tensor([(ctx or [EOS_ID]) + cont])
    with torch.no_grad():
        lp = F.log_softmax(model(full[:, :-1])[0].float(), -1)
    pos = torch.arange(full.shape[1] - 1 - len(cont), full.shape[1] - 1)
    tgt = full[0, -len(cont):]
    return float(lp[pos, tgt].sum()), bool((lp[pos].argmax(-1) == tgt).all())


class ByteTok:
    """Stand-in for ``tokenizers.Tokenizer``: one id per UTF-8 byte (vocab 256)."""

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, s: str, add_special_tokens: bool = False):
        return self._Enc(list(s.encode("utf-8")))

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return bytes(i for i in ids if i >= 4).decode("utf-8", errors="replace")


def test_loglikelihood_matches_direct_sum(model):
    g = torch.Generator().manual_seed(1)
    reqs = []
    for n_ctx, n_cont in [(5, 1), (12, 3), (1, 6), (20, 2), (0, 4), (8, 8), (3, 1)]:
        reqs.append((torch.randint(4, 256, (n_ctx,), generator=g).tolist(),
                     torch.randint(4, 256, (n_cont,), generator=g).tolist()))
    scorer = MorbiusScorer(model, seq_len=32, batch_size=3, precision="fp32")  # mixed lengths -> padding
    got = scorer.loglikelihood_tokens(reqs)
    for (c, t), (ll, greedy) in zip(reqs, got):
        ref_ll, ref_greedy = direct(model, c, t)
        assert ll == pytest.approx(ref_ll, abs=1e-4)
        assert greedy == ref_greedy


def test_greedy_flag_true_for_argmax_continuation(model):
    ctx = [7, 8, 9, 10]
    with torch.no_grad():
        nxt = int(model(torch.tensor([ctx]))[0, -1].argmax())
    (_, greedy), = MorbiusScorer(model, 32, precision="fp32").loglikelihood_tokens([(ctx, [nxt])])
    assert greedy


def test_left_truncation(model):
    g = torch.Generator().manual_seed(2)
    ctx = torch.randint(4, 256, (50,), generator=g).tolist()
    cont = torch.randint(4, 256, (3,), generator=g).tolist()
    scorer = MorbiusScorer(model, seq_len=32, batch_size=4, precision="fp32")
    inp, tgt = scorer.window(ctx, cont)
    assert len(inp) == 32 and tgt == cont and inp == (ctx + cont)[-33:-1]
    (ll, _), = scorer.loglikelihood_tokens([(ctx, cont)])
    keep = (ctx + cont)[-33:]
    assert ll == pytest.approx(direct(model, keep[:-3], cont)[0], abs=1e-4)


def test_rolling_predicts_each_token_once(model):
    scorer = MorbiusScorer(model, seq_len=32, batch_size=4, precision="fp32")
    toks = torch.randint(4, 256, (75,), generator=torch.Generator().manual_seed(3)).tolist()
    chunks = scorer.rolling_requests(toks)
    assert sum(len(t) for _, t in chunks) == len(toks)
    assert [x for _, t in chunks for x in t] == toks
    short = toks[:20]
    (ll,) = scorer.loglikelihood_rolling_tokens([short])
    assert ll == pytest.approx(direct(model, [EOS_ID], short)[0], abs=1e-4)


def test_generate_tokens_greedy(model):
    scorer = MorbiusScorer(model, 32, precision="fp32")
    ctx = [5, 6, 7, 8, 9]
    out = scorer.generate_tokens(ctx, 6)
    seq = list(ctx)
    with torch.no_grad():
        for _ in range(len(out)):
            seq.append(int(model(torch.tensor([seq]))[0, -1].argmax()))
    assert out == seq[len(ctx):]


def test_lm_eval_instances(model, tmp_path):
    pytest.importorskip("lm_eval")
    from lm_eval.api.instance import Instance

    from eval.lm_eval_adapter import MorbiusLM

    path = tmp_path / "checkpoints" / "step_0000003.pt"
    ckpt.save_atomic({"model": model.state_dict(), "config": load_config(TINY), "step": 3}, path)
    lm = MorbiusLM(checkpoint=str(tmp_path), device="cpu", batch_size="2", precision="fp32", tok=ByteTok())
    pairs = [("Question: what is red?\nAnswer:", " apple"), ("Hi there", " you"), ("", "abc"),
             ("x" * 60, " long context")]
    reqs = [Instance("loglikelihood", {}, p, i, ("t", i, 1)) for i, p in enumerate(pairs)]
    got = lm.loglikelihood(reqs)
    for (ctx, cont), (ll, _) in zip(pairs, got):
        if ctx:
            c, t = list(ctx.encode()), list(cont.encode())
            keep = (c + t)[-33:]
            ref = direct(model, keep[:-len(t)], t)[0]
        else:
            ref = direct(model, [EOS_ID], list(cont.encode()))[0]
        assert ll == pytest.approx(ref, abs=1e-4)
    roll = lm.loglikelihood_rolling([Instance("loglikelihood_rolling", {}, ("hello world",), 0, ("t", 0, 1))])
    assert roll[0] == pytest.approx(direct(model, [EOS_ID], list(b"hello world"))[0], abs=1e-4)
    gen = lm.generate_until([Instance("generate_until", {}, ("abc", {"until": ["\n"], "max_gen_toks": 4}), 0,
                                      ("t", 0, 1))])
    assert isinstance(gen[0], str) and "\n" not in gen[0]


def test_write_outputs(tmp_path):
    sample = {"doc_id": 3, "target": 1, "metrics": ["acc", "acc_norm"], "acc": 1.0, "acc_norm": 0.0,
              "filtered_resps": [[-1.5, False], [-0.5, True]]}
    assert compact_sample("piqa", sample)["lls"] == [-1.5, -0.5]
    write_outputs({"samples": {"piqa": [sample]}, "results": {"piqa": {"acc,none": 1.0}}}, tmp_path, {"name": "x"})
    rec = json.loads((tmp_path / "samples_piqa.jsonl").read_text().strip())
    assert rec["acc"] == 1.0 and rec["doc_id"] == 3
    assert json.loads((tmp_path / "results.json").read_text())["meta"]["name"] == "x"
