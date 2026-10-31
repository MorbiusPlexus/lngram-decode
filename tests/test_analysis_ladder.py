"""M2.T4 decodability ladder on tiny CPU models."""

from __future__ import annotations

import json
import math

import numpy as np
import torch

from analysis import decodability_ladder as dl
from drafting.posthoc_model import PosthocDrafter
from drafting.sketches import Sketch
from ffnmem.infer import load_model

from .test_analysis_dump import tiny_checkpoint


def test_sketch_decoder_on_hand_filled_sketch():
    sk = Sketch.empty(routes=2, bits=4, ngrams=(2, 3), layer=1, k=4, vocab=256)
    codes = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], np.uint8)  # [T, R]
    rows = dl.code_rows(sk, codes[None])
    # v2 convention: oldest code least significant (docs/VERIFIED.md section 1)
    assert rows[2][2, 0] == 0 * 256 + 3 + 16 * 5 and rows[2][2, 1] == 1 * 256 + 4 + 16 * 6
    assert rows[3][2, 1] == 1 * 4096 + 2 + 16 * 4 + 256 * 6 and rows[3][1, 0] == -1 and rows[2][0, 0] == -1
    sk.ids[2][rows[2][2, 0]] = [7, 0, 0, 0]
    sk.counts[2][rows[2][2, 0]] = [5, 0, 0, 0]
    sk.ids[2][rows[2][2, 1]] = [9, 0, 0, 0]
    sk.counts[2][rows[2][2, 1]] = [2, 0, 0, 0]
    sk.ids[3][rows[3][2, 0]] = [9, 7, 0, 0]
    sk.counts[3][rows[3][2, 0]] = [4, 0, 0, 0]
    dec = dl.SketchDecoder(sk)
    p = dec.probs(rows, smoothing=0.0)
    torch.testing.assert_close(p, sk.decode_dense(rows, 0.0))
    assert math.isclose(float(p[2, 9]), 6 / 11, rel_tol=1e-6) and math.isclose(float(p[2, 7]), 5 / 11, rel_tol=1e-6)
    assert torch.allclose(p[0], torch.full((256,), 1 / 256))  # nothing known at t = 0 -> uniform
    torch.testing.assert_close(dec.probs(rows, 1e-3), sk.decode_dense(rows, 1e-3))


def test_token_ngram_sketch_counts(tiny_data):
    n_tok = 20_000
    sk = dl.build_token_ngram_sketch(tiny_data, n_tok, rows_log2=16, k=16)  # bigrams exact, trigrams hashed
    assert sk.tokens_seen == n_tok and sk.ids[2].shape == (256 ** 2, 16) and sk.ids[3].shape == (1 << 16, 16)
    assert int(sk.counts[3].sum()) == n_tok - 3 and int(sk.counts[2].sum()) <= n_tok - 2
    toks = np.fromfile(tiny_data / "shard_00000.bin", dtype="<u2").astype(np.int64)[:n_tok]
    key = toks[:-2] + 256 * toks[1:-1]  # bigram rows are exact keys here
    nxt = toks[2:]
    top_key = np.bincount(key).argmax()
    exact = np.bincount(nxt[key == top_key], minlength=256)
    row = dict(zip(sk.ids[2][top_key].tolist(), sk.counts[2][top_key].tolist()))
    for t, c in row.items():
        if c:
            assert exact[t] == c
    assert sum(row.values()) == min(exact.sum(), sum(sorted(exact)[-16:]))
    # merging two halves equals counting at once while nothing is truncated
    a = {n: np.zeros((50, 64), np.uint16) for n in (0, 1)}
    c = {n: np.zeros((50, 64), np.uint32) for n in (0, 1)}
    rng = np.random.default_rng(0)
    rows, tk = rng.integers(0, 50, 3000), rng.integers(0, 40, 3000)
    dl.merge_topk(a[0], c[0], rows, tk, 256)
    dl.merge_topk(a[1], c[1], rows[:1000], tk[:1000], 256)
    dl.merge_topk(a[1], c[1], rows[1000:], tk[1000:], 256)
    for r in range(50):
        d0 = {i: x for i, x in zip(a[0][r], c[0][r]) if x}
        d1 = {i: x for i, x in zip(a[1][r], c[1][r]) if x}
        assert d0 == d1


def test_level1_is_uniform_rows_plus_sink(tmp_path):
    model, _ = load_model(tiny_checkpoint(tmp_path / "run"))
    m = model.blocks[1].memory.module
    codes = torch.randint(0, 16, (2, 12, m.num_routes), dtype=torch.int32)
    with torch.no_grad():
        got = dl.zero_query_readout(m, codes)
        _, value = m._project_memory(m._lookup_route_chunk(codes, None, 0, m.num_routes, None, None))
        w = 1.0 / (m.num_routes + math.exp(m.readout_sink_logit))  # each real row; the Sink takes the rest
        assert math.isclose(w, 1 / (2 * m.num_routes))  # sink_initial_active_mass 0.5 -> b0 = log R
        head = (value.sum(-2) * w)[:, :, :, None].expand(-1, -1, -1, m.readout_groups, -1)  # [B,T,Hkv,G,hd]
        want = m.readout_o_proj(head.reshape(2, 12, -1))
        torch.testing.assert_close(got, want)
        conv = dl.zero_query_hidden(m, codes, conv=True)
        torch.testing.assert_close(conv, want + m.short_conv(want))


def test_ladder_eval_levels(tmp_path, tiny_data):
    run = tiny_checkpoint(tmp_path / "run", "wave1")
    model, _ = load_model(run)
    sk = Sketch.empty(routes=4, bits=4, ngrams=(2, 3), layer=1, k=4, vocab=256)
    sk.ids[2][:, 0] = 5
    sk.counts[2][:, 0] = 1
    sk.save(tmp_path / "sk.npz")
    rnd = Sketch.empty(routes=4, bits=4, ngrams=(2, 3), layer=1, k=4, vocab=256)
    rnd.save(tmp_path / "rnd.npz")
    z = dict(np.load(tmp_path / "rnd.npz"))
    z["router_weight"] = dl.random_router_weight(model, 1, seed=3).numpy()
    np.savez(tmp_path / "rnd.npz", **z)
    tok = dl.build_token_ngram_sketch(tiny_data, 30_000, rows_log2=12)
    dl.save_token_sketch(tok, tmp_path / "tok.npz")
    dl.TunedLens(64, 1).save(tmp_path / "lens.pt")
    out = tmp_path / "res"
    res = dl.main(["eval", "--ckpt", str(run), "--data", str(tiny_data), "--run", "t", "--sketch", str(tmp_path / "sk.npz"),
                   "--tok_sketch", str(tmp_path / "tok.npz"), "--random_sketch", str(tmp_path / "rnd.npz"),
                   "--lens", str(tmp_path / "lens.pt"), "--max_tokens", "320", "--chunk", "100",
                   "--n_boot", "50", "--device", "cpu", "--out_dir", str(out)])
    L = res["levels"]
    assert set(L) == set(dl.LEVELS) - {"L2_draft"} and res["skipped"] == ["L2_draft"]
    assert L["L4_model"]["top1_agree"]["value"] == 1.0 and abs(L["L4_model"]["kl"]["value"]) < 1e-6
    assert abs(L["L4_model"]["sum_min"]["value"] - 1.0) < 1e-5
    # the hand-filled sketch predicts token 5 wherever a 2-gram exists (t >= 1)
    assert L["L0_sketch"]["acc_true"]["value"] <= 0.1
    for lv, r in L.items():
        assert 0.0 <= r["top1_agree"]["value"] <= 1.0 and r["kl"]["value"] >= -1e-6
        assert 0.0 < r["sum_min"]["value"] <= 1.0 + 1e-6
        for s in r.values():
            assert s["lo"] - 1e-9 <= s["value"] <= s["hi"] + 1e-9
    # the random-router sketch is empty, so q is uniform: KL(p || u) = log V - H(p) <= log V
    assert 0 < L["L0_random_router"]["kl"]["value"] <= math.log(256) + 1e-6
    assert json.loads((out / "ladder_t.json").read_text())["n_positions"] == 320
    assert "| L4_model |" in (out / "ladder_t.md").read_text(encoding="utf-8")


def test_level2_joint_and_posthoc(tmp_path, tiny_data):
    run_sd = tiny_checkpoint(tmp_path / "sd", "sdb")
    sd, _ = load_model(run_sd)
    res = dl.ladder(sd, tiny_data, levels=("L2_draft", "L4_model"), max_tokens=160, n_boot=20, device=torch.device("cpu"))
    assert res["level2"] == "joint draft head" and 0 <= res["levels"]["L2_draft"]["top1_agree"]["value"] <= 1
    # equals the draft head called directly
    from ffnmem.data.loader import ValWindows

    x = next(ValWindows(tiny_data, 32, 10_000).batches(5))[0]
    with torch.no_grad():
        p = sd(x).float().log_softmax(-1).reshape(160, -1)
        b, t = torch.arange(5).repeat_interleave(32), torch.arange(32).repeat(5)
        q = sd.draft_logits(x, b, t).float().log_softmax(-1)
    agree = (p.argmax(-1) == q.argmax(-1)).float().mean()
    assert math.isclose(res["levels"]["L2_draft"]["top1_agree"]["value"], float(agree), abs_tol=1e-9)

    run = tiny_checkpoint(tmp_path / "w1", "wave1")
    model, _ = load_model(run)
    PosthocDrafter(model, q_heads=2, head_dim=16).save(tmp_path / "drafter.pt")
    res = dl.main(["eval", "--ckpt", str(run), "--data", str(tiny_data), "--run", "p", "--drafter",
                   str(tmp_path / "drafter.pt"), "--levels", "L2_draft", "L1_zero_query", "--max_tokens", "96",
                   "--n_boot", "10", "--device", "cpu", "--out_dir", str(tmp_path / "res")])
    assert res["level2"] == "post-hoc drafter" and set(res["levels"]) == {"L2_draft", "L1_zero_query"}


def test_tuned_lens_training_reduces_kl(tmp_path, tiny_data):
    run = tiny_checkpoint(tmp_path / "run")
    model, _ = load_model(run)
    with torch.no_grad():  # sharpen the untrained model's next-token distributions
        model.tok_emb.weight.mul_(40)
    kw = dict(levels=("L3_tuned_lens",), max_tokens=640, n_boot=10, device=torch.device("cpu"))
    before = dl.ladder(model, tiny_data, lens=dl.TunedLens(64, 1), **kw)["levels"]["L3_tuned_lens"]["kl"]["value"]
    lens, hist = dl.train_lens(model, tiny_data, layer=1, train_tokens=32 * 4 * 60, batch=4, positions=16, lr=3e-3,
                               device=torch.device("cpu"), log_every=0)
    assert len(hist) == 60 and np.mean(hist[-10:]) < np.mean(hist[:5])
    after = dl.ladder(model, tiny_data, lens=lens, **kw)["levels"]["L3_tuned_lens"]["kl"]["value"]
    assert after < 0.7 * before
    lens.save(tmp_path / "lens.pt")
    again = dl.TunedLens.load(tmp_path / "lens.pt")
    assert torch.equal(again.proj.weight, lens.proj.weight) and again.layer == 1
