"""Interpretability studies on the Wave 3 memory models (CPU; reads code dumps and analysis/interp_gpu.py outputs).

    python -m analysis.interp_studies all --dumps ~/morbius_data/dumps --interp results/interp \
        --train_shard ~/morbius_data/climbmix_clean/shard_00000.bin --out results/interp

Studies (each writes results/interp/<study>.json; missing inputs are skipped):
  utilization   rows of the table touched on held-out text; effective rows; hit concentration
  redundancy    normalised mutual information between routes' codes
  semantics     what codes encode: token classes, and next-token information beyond the current token
  benefit       per-token loss gain from the memory (SB), and SB vs dense D270, by token frequency / class
  draftacc      draft-head agreement with the model by model confidence, token frequency and sink mass
  accept        per-row acceptance: complementarity of prompt lookup and the self-drafting head, hybrids,
                acceptance along the continuation
  replicate     code agreement between the two SB training runs (replicates)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.dump_codes import load_dump

MODELS = {"sb": "SB", "sdb": "SD-B", "sdc": "SD-C"}
RNG = np.random.default_rng(0)


def boot_mean(v: np.ndarray, groups: np.ndarray | None = None, n: int = 1000) -> list[float]:
    """95% CI of the mean; resample groups (e.g. documents/prompts) if given."""
    v = np.asarray(v, np.float64)
    if groups is None:
        idx = RNG.integers(0, len(v), (n, len(v)))
        m = v[idx].mean(1)
    else:
        u, inv = np.unique(groups, return_inverse=True)
        s = np.bincount(inv, v, len(u))
        c = np.bincount(inv, None, len(u))
        idx = RNG.integers(0, len(u), (n, len(u)))
        m = s[idx].sum(1) / c[idx].sum(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


# ------------------------------------------------------------------ token classes
def token_classes(vocab: int = 32768) -> tuple[np.ndarray, list[str]]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(__file__).resolve().parents[1] / "data/tokenizer/tokenizer.json"))
    names = ["word-start", "word-cont", "punct", "digit", "whitespace/newline", "special"]
    cls = np.zeros(vocab, np.int8)
    strs = []
    for i in range(vocab):
        s = tok.id_to_token(i) or ""
        d = tok.decode([i]) if s else ""
        strs.append(d)
        if s.startswith("<") and s.endswith(">") or not s:
            cls[i] = 5
        elif d.strip() == "":
            cls[i] = 4
        elif d.strip().isdigit():
            cls[i] = 3
        elif all(not ch.isalnum() for ch in d.strip()):
            cls[i] = 2
        elif d.startswith(" "):
            cls[i] = 0
        else:
            cls[i] = 1
    return cls, names, strs


def freq_buckets(tokens: np.ndarray, train_counts: np.ndarray) -> tuple[np.ndarray, list[str]]:
    f = train_counts[tokens]
    edges = [0, 10, 100, 1_000, 10_000, 100_000, 1_000_000, np.inf]
    b = np.digitize(f, edges[1:-1])
    labels = ["<10", "10-100", "100-1k", "1k-10k", "10k-100k", "100k-1M", ">1M"]
    return b, labels


# ------------------------------------------------------------------ studies
def utilization(a) -> dict:
    out = {}
    for m in MODELS:
        if not (Path(a.dumps) / m).exists():
            continue
        meta, d = load_dump(Path(a.dumps) / m, ["codes_L1", "pos"])
        codes = np.asarray(d["codes_L1"]).astype(np.int64)
        pos = np.asarray(d["pos"])
        R = codes.shape[1]
        res = {"routes": R}
        for n in (2, 3):
            valid = pos >= n - 1
            addr = codes[n - 1:].copy() * 0
            # v2 convention: oldest code least significant; position t reads codes t-n+1..t
            T = len(codes)
            addr = np.zeros((T - n + 1, R), np.int64)
            for e in range(n):
                addr |= codes[e: e + T - n + 1] << (4 * e)
            addr = addr[valid[n - 1:]]
            rows = 16 ** n
            touched, eff, top1share = [], [], []
            for r in range(R):
                c = np.bincount(addr[:, r], minlength=rows)
                touched.append((c > 0).mean())
                eff.append(2 ** entropy(c) / rows)
                k = max(1, rows // 100)
                top1share.append(np.sort(c)[::-1][:k].sum() / c.sum())
            res[f"order{n}"] = {"rows_per_route": rows, "frac_rows_touched": float(np.mean(touched)),
                                "frac_rows_touched_min": float(np.min(touched)),
                                "effective_rows_frac": float(np.mean(eff)),
                                "hits_in_top1pct_rows": float(np.mean(top1share)), "positions": int(len(addr))}
        out[MODELS[m]] = res
        print(f"[utilization] {MODELS[m]}: " + json.dumps({k: v for k, v in res.items() if k != 'routes'}), flush=True)
    return out


def nmi(x: np.ndarray, y: np.ndarray, k: int = 16) -> float:
    j = np.bincount(x * k + y, minlength=k * k).reshape(k, k).astype(np.float64)
    hx, hy, hxy = entropy(j.sum(1)), entropy(j.sum(0)), entropy(j.ravel())
    return (hx + hy - hxy) / max(min(hx, hy), 1e-9)


def redundancy(a) -> dict:
    out = {}
    for m in MODELS:
        if not (Path(a.dumps) / m).exists():
            continue
        _, d = load_dump(Path(a.dumps) / m, ["codes_L1"])
        codes = np.asarray(d["codes_L1"][:500_000]).astype(np.int64)
        R = codes.shape[1]
        pairs = RNG.choice(R, (3000, 2))
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        vals = np.array([nmi(codes[:, i], codes[:, j]) for i, j in pairs])
        sub = RNG.choice(R, 24, replace=False)
        best = []
        for i in sub:
            best.append(max(nmi(codes[:, i], codes[:, j]) for j in range(R) if j != i))
        res = {"pairs": int(len(vals)), "nmi_mean": float(vals.mean()), "nmi_median": float(np.median(vals)),
               "nmi_p90": float(np.percentile(vals, 90)), "frac_pairs_nmi_gt_0.3": float((vals > 0.3).mean()),
               "max_nmi_per_route_mean": float(np.mean(best)), "max_nmi_per_route_min": float(np.min(best)),
               "hist": np.histogram(vals, bins=20, range=(0, 1))[0].tolist()}
        out[MODELS[m]] = res
        print(f"[redundancy] {MODELS[m]}: " + json.dumps({k: v for k, v in res.items() if k != 'hist'}), flush=True)
    return out


def _cond_ce(key_fit, y_fit, key_ev, y_ev, back_fit_p, back_ev_p, beta: float = 1.0) -> float:
    """Held-out CE (bits) of y given a discrete key, interpolated with a back-off distribution:
    P(y|k) = (n(k,y) + beta * Pb(y)) / (n(k) + beta).  back_*_p = Pb(y_true) at each position."""
    ky_fit = key_fit.astype(np.int64) * 40000 + y_fit
    u, cnt = np.unique(ky_fit, return_counts=True)
    ku, kcnt = np.unique(key_fit, return_counts=True)
    ky_ev = key_ev.astype(np.int64) * 40000 + y_ev
    i = np.searchsorted(u, ky_ev)
    i = np.clip(i, 0, len(u) - 1)
    n_ky = np.where(u[i] == ky_ev, cnt[i], 0)
    j = np.clip(np.searchsorted(ku, key_ev), 0, len(ku) - 1)
    n_k = np.where(ku[j] == key_ev, kcnt[j], 0)
    p = (n_ky + beta * back_ev_p) / (n_k + beta)
    return float(-np.log2(np.clip(p, 1e-12, 1)).mean())


def semantics(a) -> dict:
    cls, names, strs = token_classes()
    out = {"classes": names}
    for m in MODELS:
        if not (Path(a.dumps) / m).exists():
            continue
        _, d = load_dump(Path(a.dumps) / m, ["codes_L1", "tokens", "next", "doc"])
        codes = np.asarray(d["codes_L1"]).astype(np.int64)
        tok, nxt, doc = (np.asarray(d[k]).astype(np.int64) for k in ("tokens", "next", "doc"))
        R = codes.shape[1]
        fit = doc < np.median(doc)
        ev = ~fit
        # class information in the codes
        c_cls = cls[tok]
        h_cls = entropy(np.bincount(c_cls[ev], minlength=6))
        mi_cls = []
        for r in range(R):
            j = np.bincount(codes[ev, r] * 6 + c_cls[ev], minlength=16 * 6).reshape(16, 6).astype(float)
            mi_cls.append(entropy(j.sum(1)) + entropy(j.sum(0)) - entropy(j.ravel()))
        mi_cls = np.array(mi_cls)
        # next-token information beyond the current token: CE(y | x) vs CE(y | x, code_r)
        uni = np.bincount(nxt[fit], minlength=40000).astype(float) + 0.5
        uni /= uni.sum()
        ce_x_fit_back = uni[nxt[fit]]
        # P(y|x) evaluated at the eval positions and at the fit positions (for the next level's back-off)
        def p_given(key_fit, key_ev, back_fit, back_ev, beta=1.0):
            ky = key_fit * 40000 + nxt[fit]
            u, cnt = np.unique(ky, return_counts=True)
            ku, kc = np.unique(key_fit, return_counts=True)

            def at(key, y, back):
                kk = key * 40000 + y
                i = np.clip(np.searchsorted(u, kk), 0, len(u) - 1)
                nky = np.where(u[i] == kk, cnt[i], 0)
                jj = np.clip(np.searchsorted(ku, key), 0, len(ku) - 1)
                nk = np.where(ku[jj] == key, kc[jj], 0)
                return (nky + beta * back) / (nk + beta)
            return at(key_fit, nxt[fit], back_fit), at(key_ev, nxt[ev], back_ev)

        px_fit, px_ev = p_given(tok[fit], tok[ev], uni[nxt[fit]], uni[nxt[ev]])
        ce_x = float(-np.log2(px_ev).mean())
        prev = np.concatenate(([0], tok[:-1]))
        _, pbi_ev = p_given(prev[fit] * 32768 + tok[fit], prev[ev] * 32768 + tok[ev], px_fit, px_ev)
        ce_bigram = float(-np.log2(pbi_ev).mean())
        gains = []
        for r in range(R):
            _, pc_ev = p_given(tok[fit] * 16 + codes[fit, r], tok[ev] * 16 + codes[ev, r], px_fit, px_ev)
            gains.append(ce_x - float(-np.log2(pc_ev).mean()))
        gains = np.array(gains)
        res = {"H_class_bits": h_cls, "mi_code_class_bits_mean": float(mi_cls.mean()),
               "mi_code_class_frac_mean": float(mi_cls.mean() / h_cls), "mi_code_class_frac_max": float(mi_cls.max() / h_cls),
               "ce_next_given_x_bits": ce_x, "ce_next_given_prev_x_bits": ce_bigram,
               "bigram_gain_bits": ce_x - ce_bigram,
               "code_gain_bits_mean": float(gains.mean()), "code_gain_bits_max": float(gains.max()),
               "code_gain_bits_p90": float(np.percentile(gains, 90)),
               "top_class_route": int(mi_cls.argmax()), "top_gain_route": int(gains.argmax())}
        # qualitative: for the most class-informative route, top tokens per code
        r = int(mi_cls.argmax())
        ex = {}
        for c in range(16):
            m_ = codes[ev, r] == c
            if m_.sum() == 0:
                continue
            t, n = np.unique(tok[ev][m_], return_counts=True)
            top = t[np.argsort(-n)[:8]]
            ex[str(c)] = {"share": float(m_.mean()), "top_tokens": [strs[i] for i in top],
                          "class_mix": np.bincount(c_cls[ev][m_], minlength=6).tolist()}
        res["examples_route"] = r
        res["examples"] = ex
        out[MODELS[m]] = res
        print(f"[semantics] {MODELS[m]}: " + json.dumps({k: v for k, v in res.items() if k != 'examples'}), flush=True)
    return out


def train_counts(a) -> np.ndarray:
    path = Path(a.out) / "train_token_counts.npy"
    if path.exists():
        return np.load(path)
    sh = np.memmap(a.train_shard, dtype="<u2", mode="r")
    c = np.zeros(32768, np.int64)
    for s in range(0, len(sh), 50_000_000):
        c += np.bincount(sh[s:s + 50_000_000], minlength=32768)
    np.save(path, c)
    return c


def benefit(a) -> dict:
    p = Path(a.interp) / "pertoken_sb.npz"
    if not p.exists():
        return {}
    z = np.load(p)
    ce, cez = z["ce"].astype(np.float64), z["ce_zero"].astype(np.float64)
    _, d = load_dump(Path(a.dumps) / "sb", ["tokens", "next", "doc", "sink_L1", "memnorm_L1", "pos"])
    tok, nxt, doc, sink, pos = (np.asarray(d[k]) for k in ("tokens", "next", "doc", "sink_L1", "pos"))
    cls, names, _ = token_classes()
    tc = train_counts(a)
    gain = cez - ce
    out = {"mean_gain_nats": float(gain.mean()), "ci": boot_mean(gain, doc)}
    srt = np.sort(gain)[::-1]
    out["share_of_gain_from_top10pct_tokens"] = float(srt[: len(srt) // 10].sum() / gain.sum())
    out["frac_tokens_hurt_by_memory"] = float((gain < 0).mean())
    fb, fl = freq_buckets(tok, tc)
    nb, _ = freq_buckets(nxt, tc)
    out["by_current_token_freq"] = {fl[i]: {"gain": float(gain[fb == i].mean()), "n": int((fb == i).sum())} for i in range(7) if (fb == i).any()}
    out["by_next_token_freq"] = {fl[i]: {"gain": float(gain[nb == i].mean()), "n": int((nb == i).sum())} for i in range(7) if (nb == i).any()}
    out["by_next_token_class"] = {names[i]: {"gain": float(gain[cls[nxt] == i].mean()), "n": int((cls[nxt] == i).sum())} for i in range(6) if (cls[nxt] == i).any()}
    out["by_current_token_class"] = {names[i]: {"gain": float(gain[cls[tok] == i].mean()), "n": int((cls[tok] == i).sum())} for i in range(6) if (cls[tok] == i).any()}
    q = np.quantile(sink, [0, .2, .4, .6, .8, 1])
    sb_ = np.clip(np.digitize(sink, q[1:-1]), 0, 4)
    out["by_sink_quintile"] = {f"q{i + 1} ({q[i]:.2f}-{q[i + 1]:.2f})": float(gain[sb_ == i].mean()) for i in range(5)}
    out["corr_gain_vs_sink"] = float(np.corrcoef(gain, sink)[0, 1])
    pb = np.digitize(pos, [1, 2, 8, 64, 512])
    out["by_position"] = {lab: float(gain[pb == i].mean()) for i, lab in enumerate(["0", "1", "2-7", "8-63", "64-511", "512+"]) if (pb == i).any()}
    dp = Path(a.interp) / "pertoken_d270.npz"
    if dp.exists():
        ced = np.load(dp)["ce"].astype(np.float64)
        diff = ced - ce  # >0: SB better than the dense model on this token
        out["sb_minus_dense"] = {"mean_dense_minus_sb": float(diff.mean()), "ci": boot_mean(diff, doc),
                                 "by_current_token_freq": {fl[i]: float(diff[fb == i].mean()) for i in range(7) if (fb == i).any()},
                                 "by_next_token_class": {names[i]: float(diff[cls[nxt] == i].mean()) for i in range(6) if (cls[nxt] == i).any()},
                                 "corr_with_memory_gain": float(np.corrcoef(diff, gain)[0, 1]),
                                 "note": "D270 is the 2B-token checkpoint; positive = SB better"}
    print("[benefit] " + json.dumps(out)[:1500], flush=True)
    return out


def draftacc(a) -> dict:
    out = {}
    tc = None
    for m in ("sdc", "sdb"):
        p = Path(a.interp) / f"draft_{m}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        dt, dce = z["draft_top1"], z["draft_ce"].astype(np.float64)
        _, d = load_dump(Path(a.dumps) / m, ["tokens", "next", "top1", "topk_logp", "sink_L1", "doc", "logp_next"])
        tok, nxt, top1, doc, sink = (np.asarray(d[k]) for k in ("tokens", "next", "top1", "doc", "sink_L1"))
        pmax = np.exp(np.asarray(d["topk_logp"])[:, 0].astype(np.float64))
        model_ce = -np.asarray(d["logp_next"]).astype(np.float64)
        agree = (dt == top1).astype(np.float64)
        tc = tc if tc is not None else train_counts(a)
        cls, names, _ = token_classes()
        res = {"agree": float(agree.mean()), "agree_ci": boot_mean(agree, doc),
               "draft_ce": float(dce.mean()), "model_ce": float(model_ce.mean()),
               "draft_acc_true": float((dt == nxt).mean()), "model_acc_true": float((top1 == nxt).mean())}
        edges = [0, .2, .4, .6, .8, .9, 1.01]
        b = np.digitize(pmax, edges[1:-1])
        res["agree_by_model_confidence"] = {f"{edges[i]:.1f}-{min(edges[i + 1], 1):.1f}": {"agree": float(agree[b == i].mean()), "frac": float((b == i).mean())} for i in range(6) if (b == i).any()}
        fb, fl = freq_buckets(tok, tc)
        res["agree_by_current_token_freq"] = {fl[i]: float(agree[fb == i].mean()) for i in range(7) if (fb == i).any()}
        res["agree_by_next_class"] = {names[i]: float(agree[cls[top1] == i].mean()) for i in range(6) if (cls[top1] == i).any()}
        q = np.quantile(sink, [0, .2, .4, .6, .8, 1])
        sq = np.clip(np.digitize(sink, q[1:-1]), 0, 4)
        res["agree_by_sink_quintile"] = {f"q{i + 1}": float(agree[sq == i].mean()) for i in range(5)}
        # how much of the model's high-confidence predictions does the drafter capture
        hc = pmax > 0.6
        res["frac_positions_model_conf_gt_0.6"] = float(hc.mean())
        res["agree_when_model_conf_gt_0.6"] = float(agree[hc].mean())
        if "draft_pmax" in z.files:
            # the drafter's own confidence as a gate (usable live: needs nothing from the target)
            dp = z["draft_pmax"].astype(np.float64)
            res["agree_by_draft_confidence"] = {
                f"{lo:.1f}-{hi:.1f}": {"agree": float(agree[(dp >= lo) & (dp < hi)].mean()), "frac": float(((dp >= lo) & (dp < hi)).mean())}
                for lo, hi in [(0, .1), (.1, .2), (.2, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)] if ((dp >= lo) & (dp < hi)).any()}
            res["gate"] = {f"{tau:.2f}": {"coverage": float((dp >= tau).mean()), "agree": float(agree[dp >= tau].mean())}
                           for tau in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)}
            r_ = np.corrcoef(dp, agree)[0, 1]
            res["corr_draft_conf_vs_agree"] = float(r_)
        out[MODELS[m]] = res
        print(f"[draftacc] {MODELS[m]}: " + json.dumps(res)[:1200], flush=True)
    return out


def accept(a) -> dict:
    out = {}
    for m in ("sdc", "sdb", "sb"):
        p = Path(a.interp) / f"rows_{m}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        res = {}
        for t in ("open", "long"):
            pld = z[f"{t}/pld"].astype(np.float64)
            matched = z[f"{t}/pld_matched"]
            prompt = z[f"{t}/prompt"]
            row = z[f"{t}/row"]
            cand = [k.split("/", 1)[1] for k in z.files if k.startswith(f"{t}/") and k.split("/", 1)[1] not in ("pld", "row", "prompt", "pld_matched")]
            r = {"pld": float(pld.mean()), "frac_rows_pld_match": float(matched.mean()),
                 "pld_when_matched": float(pld[matched].mean()) if matched.any() else None}
            for c in cand:
                v = z[f"{t}/{c}"].astype(np.float64)
                hyb = np.where(matched, pld, v)  # deployable: PLD if it finds a match, else the other drafter
                orc = np.maximum(pld, v)          # oracle per row (not deployable)
                d = hyb - pld
                r[c] = {"alone": float(v.mean()), "alone_ci": boot_mean(v, prompt),
                        "when_pld_unmatched": float(v[~matched].mean()) if (~matched).any() else None,
                        "hybrid_pld_else": float(hyb.mean()), "hybrid_ci": boot_mean(hyb, prompt),
                        "hybrid_minus_pld": float(d.mean()), "hybrid_minus_pld_ci": boot_mean(d, prompt),
                        "oracle_max": float(orc.mean()),
                        "frac_rows_beats_pld": float((v > pld).mean()), "frac_rows_pld_beats": float((pld > v).mean())}
            # acceptance along the continuation (rows 0..124 at gamma=4)
            bins = [0, 16, 32, 64, 96, 125]
            b = np.digitize(row, bins[1:-1])
            r["by_continuation_position"] = {f"{bins[i]}-{bins[i + 1] - 1}": {k: float(z[f"{t}/{k}"].astype(float)[b == i].mean()) for k in ["pld"] + cand} for i in range(5)}
            res[t] = r
        out[MODELS[m]] = res
        print(f"[accept] {MODELS[m]}: " + json.dumps(res)[:1500], flush=True)
    return out


def replicate(a) -> dict:
    if not (Path(a.dumps) / "sbr").exists():
        return {}
    _, d1 = load_dump(Path(a.dumps) / "sb", ["codes_L1"])
    _, d2 = load_dump(Path(a.dumps) / "sbr", ["codes_L1"])
    c1, c2 = np.asarray(d1["codes_L1"]), np.asarray(d2["codes_L1"])
    same = (c1 == c2).mean(0)
    nm = np.array([nmi(c1[:300_000, r].astype(np.int64), c2[:300_000, r].astype(np.int64)) for r in range(c1.shape[1])])
    # per-bit agreement
    bits = np.stack([((c1 >> b) & 1) == ((c2 >> b) & 1) for b in range(4)]).mean((1, 2))
    out = {"code_agreement_mean": float(same.mean()), "code_agreement_min": float(same.min()),
           "code_agreement_max": float(same.max()), "bit_agreement": bits.tolist(),
           "nmi_same_route_mean": float(nm.mean()), "chance_agreement": 1 / 16,
           "all_routes_identical_frac_positions": float((c1 == c2).all(1).mean())}
    print("[replicate] " + json.dumps(out), flush=True)
    return out


STUDIES = {"utilization": utilization, "redundancy": redundancy, "semantics": semantics, "benefit": benefit,
           "draftacc": draftacc, "accept": accept, "replicate": replicate}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("studies", nargs="+", help=f"'all' or any of {list(STUDIES)}")
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--interp", default="results/interp")
    ap.add_argument("--train_shard", required=True)
    ap.add_argument("--out", default="results/interp")
    ap.add_argument("--acc_cache", default="~/morbius_data/acc_cache")
    a = ap.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    names = list(STUDIES) if a.studies == ["all"] else a.studies
    for n in names:
        res = STUDIES[n](a)
        if res:
            (Path(a.out) / f"{n}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")



def repetition(a) -> dict:
    """How much of each greedy continuation copies earlier text (the regime prompt lookup exploits)."""
    out = {}
    for m, cache in (("sb", "w3_sb_1b"), ("sdb", "w3_sdb_1b"), ("sdc", "w3_sdc_1b")):
        res = {}
        for t in ("open", "long"):
            p = Path(a.acc_cache) / cache / f"records_{t}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            toks = z["tokens"].astype(np.int64)
            P = 256 if t == "open" else 1536
            n = 4
            copied = np.zeros((len(toks), toks.shape[1] - P), bool)
            from_prompt = np.zeros_like(copied)
            for i, s in enumerate(toks):
                seen, seen_prompt = set(), set()
                for j in range(n, len(s)):
                    g = tuple(s[j - n:j + 1])  # (n-gram context, next token)
                    if j >= P:
                        copied[i, j - P] = g in seen
                        from_prompt[i, j - P] = g in seen_prompt
                    seen.add(g)
                    if j < P:
                        seen_prompt.add(g)
            bins = [0, 16, 32, 64, 96, 128]
            res[t] = {"frac_tokens_copied_5gram": float(copied.mean()),
                      "frac_copied_from_prompt": float(from_prompt.mean()),
                      "frac_copied_from_own_continuation": float((copied & ~from_prompt).mean()),
                      "by_position": {f"{bins[k]}-{bins[k + 1] - 1}": float(copied[:, bins[k]:bins[k + 1]].mean()) for k in range(5)}}
        out[MODELS[m]] = res
        print(f"[repetition] {MODELS[m]}: " + json.dumps(res), flush=True)
    return out


STUDIES["repetition"] = repetition


if __name__ == "__main__":
    main()
