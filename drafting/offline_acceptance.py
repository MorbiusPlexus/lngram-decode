"""M3.T4: offline acceptance harness.

    python -m drafting.offline_acceptance --target runs/w3_sb_1b --data_dir $DATA \
        --drafters pld ngram=ngram@ngram/climbmix_300m.npz sketch=sketch@sketches/w3_sb_1b.npz@runs/ph_sb/drafter.pt \
                   posthoc=posthoc@runs/ph_sb/drafter.pt posthoc_ssm=posthoc@runs/ph_sb_ssm/drafter.pt \
        --timings results/timings_B.json
    python -m drafting.offline_acceptance --target runs/w3_sdb_1b --data_dir $DATA \
        --drafters pld ngram=ngram@ngram/climbmix_300m.npz sdhead sdhead_online=sdhead+online

1. Prompts from val text (documents split at EOS, in val order): ``open`` = first 256 tokens
   of a document with >= 256 tokens; ``long`` = first 1,536 tokens of a document with >= 2,048
   tokens; ``code`` only with ``--code_doc_ids`` (document indices in val order) -- the shards
   carry no subset label (docs/VERIFIED.md section 8), so by default it is skipped and the
   reason is written to the JSON.
2. Greedy continuation of 128 tokens per prompt (``greedy_generate``, batched, KV cache,
   fp16 autocast on CUDA) and the target's memory codes at every position -> ``SeqRecord``s,
   cached in ``--cache_dir`` (reused while target / lengths / counts match), so drafters can
   be added later without regenerating.
3. Each drafter drafts gamma tokens at every continuation position (``Drafter.drafts``;
   prefix-consistent drafters are run once at max gamma and sliced).  Accepted length per
   row (``accepted_lengths``, rows with a full reference only), averaged per prompt.
4. Per text type and gamma in {2, 4, 6}: mean accepted length, implied speedup
   ``E[accepted + 1] / (1 + gamma * c_draft)``, bootstrap 95% CIs over prompts (1,000
   resamples), and the per-prompt means (the gates' paired bootstrap uses them).

Drafter specs: ``[label=]kind[@arg[@arg2]][+online][+own]`` with kinds ``pld``,
``ngram@<npz>``, ``sketch@<npz>[@<posthoc.pt> (token router for drafted positions)]``,
``posthoc@<drafter.pt>``, ``sdhead``, ``oracle``.  ``+online``: online sketch counts from
verified tokens; ``+own``: the drafter's own codes at verified positions.
``c_draft`` (drafter cost / target step): ``--c_draft label=value`` or ``--timings`` JSON
(``{"c_draft": {label: v}}`` or ``{label: v}`` or ``{label: {"c_draft": v}}``); without it the
speedup is null.  Output ``results/acceptance_<target>.json`` (merged with an existing file).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from drafting.common import Drafter, SeqRecord, accepted_lengths, full_rows

GAMMAS = (2, 4, 6)


# ------------------------------------------------------------------ prompts
def split_docs(tokens: np.ndarray, eos: int) -> list[tuple[int, int]]:
    """(start, end) of each document (EOS excluded)."""
    ends = np.flatnonzero(np.asarray(tokens) == eos)
    starts = np.concatenate(([0], ends[:-1] + 1)) if len(ends) else np.array([0])
    if not len(ends):
        ends = np.array([len(tokens)])
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]


def select_prompts(tokens: np.ndarray, docs: list, n: int, prompt_len: int, min_doc_len: int,
                   doc_ids=None) -> tuple[np.ndarray, np.ndarray]:
    """First ``prompt_len`` tokens of the first ``n`` documents with >= ``min_doc_len`` tokens."""
    pool = range(len(docs)) if doc_ids is None else [int(i) for i in doc_ids if 0 <= int(i) < len(docs)]
    out, ids = [], []
    for i in pool:
        s, e = docs[i]
        if e - s >= max(min_doc_len, prompt_len):
            out.append(np.asarray(tokens[s: s + prompt_len], dtype=np.int64))
            ids.append(i)
            if len(out) == n:
                break
    if not out:
        return np.zeros((0, prompt_len), np.int64), np.zeros(0, np.int64)
    return np.stack(out), np.asarray(ids, np.int64)


# ------------------------------------------------------------------ generation
@torch.no_grad()
def generate_records(model, prompts: np.ndarray, n_new: int, text_type: str, doc_ids: np.ndarray,
                     batch: int = 16, amp_dtype: torch.dtype | None = None) -> list[SeqRecord]:
    """Greedy continuations + the target's codes at every position (prompt and continuation)."""
    from ffnmem.infer import decode_step, greedy_generate

    dev = model.tok_emb.weight.device
    recs = []
    for b0 in range(0, len(prompts), batch):
        p = torch.as_tensor(prompts[b0: b0 + batch], device=dev)
        gen, state = greedy_generate(model, p, n_new, amp_dtype=amp_dtype)
        ctx = torch.autocast(dev.type, dtype=amp_dtype) if amp_dtype else torch.autocast(dev.type, enabled=False)
        with ctx:
            decode_step(model, state, gen[:, -1:])  # codes of the last position
        toks = torch.cat((p, gen), 1).cpu().numpy()
        length = toks.shape[1]
        codes = {pos: c[:, :length].cpu().numpy() for pos, c in state.codes.items()}
        for i in range(len(toks)):
            recs.append(SeqRecord(toks[i], p.shape[1], text_type, {pos: c[i] for pos, c in codes.items()},
                                  int(doc_ids[b0 + i])))
    return recs


def save_records(path: str | Path, recs: list[SeqRecord], meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"tokens": np.stack([r.tokens for r in recs]).astype(np.int32),
              "doc_ids": np.asarray([r.doc_id for r in recs], np.int64)}
    for pos in recs[0].codes:
        arrays[f"codes_{pos}"] = np.stack([r.codes[pos] for r in recs]).astype(np.uint8)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, meta=json.dumps(dict(meta, prompt_len=recs[0].prompt_len, text_type=recs[0].text_type)),
             **arrays)
    tmp.replace(path)


def load_records(path: str | Path) -> tuple[list[SeqRecord], dict]:
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    toks = z["tokens"].astype(np.int64)
    code_keys = [k for k in z.files if k.startswith("codes_")]
    codes = {int(k[6:]): z[k] for k in code_keys}
    recs = [SeqRecord(toks[i], int(meta["prompt_len"]), meta["text_type"], {p: c[i] for p, c in codes.items()},
                      int(z["doc_ids"][i])) for i in range(len(toks))]
    return recs, meta


# ------------------------------------------------------------------ drafters
def parse_spec(spec: str) -> dict:
    label, _, body = spec.partition("=") if "=" in spec else ("", "", spec)
    parts = body.split("+")
    opts = set()
    while len(parts) > 1 and parts[-1] in ("online", "own"):
        opts.add(parts.pop())
    kind, *args = "+".join(parts).split("@")
    return {"label": label or spec, "kind": kind, "args": args, "online": "online" in opts,
            "verified": "own" if "own" in opts else "exact", "spec": spec}


def make_drafter(spec: dict, target, amp_dtype: torch.dtype | None = None) -> Drafter:
    from drafting.baselines import PromptLookupDrafter, TokenNgramDrafter
    from drafting.drafters import OracleDrafter, PosthocNeuralDrafter, SDHeadDrafter, SketchDrafter
    from drafting.posthoc_model import PosthocDrafter
    from drafting.sketches import Sketch

    kind, args = spec["kind"], spec["args"]
    common = dict(verified=spec["verified"], online=spec["online"])
    if kind == "pld":
        return PromptLookupDrafter()
    if kind == "ngram":
        return TokenNgramDrafter(args[0])
    if kind == "sketch":
        router = PosthocDrafter.load(args[1], target).eval() if len(args) > 1 else None
        return SketchDrafter(Sketch.load(args[0]), target, router=router, **common)
    if kind == "posthoc":
        return PosthocNeuralDrafter(PosthocDrafter.load(args[0], target).eval(), amp_dtype=amp_dtype, **common)
    if kind == "sdhead":
        return SDHeadDrafter(target, amp_dtype=amp_dtype, **common)
    if kind == "oracle":
        return OracleDrafter(target, amp_dtype=amp_dtype)
    raise ValueError(f"unknown drafter kind {kind!r}")


def per_prompt_accepted(drafter: Drafter, recs: list[SeqRecord], gammas=GAMMAS) -> dict:
    """gamma -> float [n_prompts]: mean accepted length over each prompt's full rows."""
    out = {g: np.zeros(len(recs)) for g in gammas}
    gmax = max(gammas)
    for i, rec in enumerate(recs):
        base = drafter.drafts(rec, gmax) if getattr(drafter, "prefix_consistent", False) else None
        for g in gammas:
            d = base[:, :g] if base is not None else drafter.drafts(rec, g)
            acc = accepted_lengths(rec, d)[full_rows(rec, g)]
            out[g][i] = acc.mean() if len(acc) else np.nan
    return out


# ------------------------------------------------------------------ statistics
def bootstrap_means(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n_boot, len(v)))
    return v[idx].mean(1)


def ci95(samples: np.ndarray) -> list[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def summarize(per_prompt: np.ndarray, gamma: int, c_draft: float | None, n_boot: int = 1000) -> dict:
    v = per_prompt[np.isfinite(per_prompt)]
    if not len(v):
        return {"n_prompts": 0}
    boots = bootstrap_means(v, n_boot)
    out = {"mean_accepted": float(v.mean()), "ci95": ci95(boots), "n_prompts": int(len(v)),
           "per_prompt": [float(x) for x in per_prompt]}
    if c_draft is not None:
        f = lambda a: (a + 1.0) / (1.0 + gamma * c_draft)  # noqa: E731
        out.update(speedup=float(f(v.mean())), speedup_ci95=ci95(f(boots)), c_draft=float(c_draft))
    else:
        out.update(speedup=None, speedup_ci95=None, c_draft=None)
    return out


def read_c_draft(pairs: list[str], timings: str | None) -> dict:
    out: dict = {}
    if timings:
        blob = json.loads(Path(timings).read_text(encoding="utf-8"))
        blob = blob.get("c_draft", blob)
        for k, v in blob.items():
            if isinstance(v, dict) and "c_draft" in v:
                v = v["c_draft"]
            if isinstance(v, (int, float)):
                out[k] = float(v)
    for p in pairs:
        k, v = p.split("=")
        out[k] = float(v)
    return out


# ------------------------------------------------------------------ CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, help="checkpoint file or run dir")
    p.add_argument("--name", help="target name in the output file (default: the run's config name)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--drafters", nargs="*", default=["pld"])
    p.add_argument("--types", nargs="+", default=["open", "long", "code"])
    p.add_argument("--n_prompts", type=int, default=1000)
    p.add_argument("--n_new", type=int, default=128)
    p.add_argument("--open_len", type=int, default=256)
    p.add_argument("--long_len", type=int, default=1536)
    p.add_argument("--long_min_doc", type=int, default=2048)
    p.add_argument("--code_doc_ids", help=".npy / .txt of val document indices that are code")
    p.add_argument("--val_tokens", type=float, default=0, help="use only the first N val tokens (0 = all)")
    p.add_argument("--gammas", type=int, nargs="+", default=list(GAMMAS))
    p.add_argument("--gen_batch", type=int, default=16)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--c_draft", action="append", default=[], metavar="LABEL=VALUE")
    p.add_argument("--timings", help="JSON with c_draft per drafter label (M5)")
    p.add_argument("--cache_dir", help="record cache (default results/acceptance_cache/<name>)")
    p.add_argument("--out", help="default results/acceptance_<name>.json")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--precision", default="auto", choices=["auto", "fp16", "fp32"])
    return p.parse_args(argv)


def get_records(a, model, cfg: dict, name: str, amp_dtype) -> tuple[dict, dict]:
    from ffnmem.data.loader import ShardSet

    shards = ShardSet(a.data_dir)
    val = shards.memmap(shards.meta["val_file"])
    if a.val_tokens:
        val = val[: int(a.val_tokens)]
    val = np.asarray(val)
    docs = split_docs(val, int(shards.meta["eos_id"]))
    cache = Path(a.cache_dir or Path("results") / "acceptance_cache" / name)
    recs_by_type, info = {}, {}
    for t in a.types:
        if t == "code" and not a.code_doc_ids:
            info[t] = {"skipped": "no code subset identifiable in the val shards (pass --code_doc_ids)"}
            continue
        plen = a.long_len if t == "long" else a.open_len
        min_doc = a.long_min_doc if t == "long" else plen
        doc_ids = None
        if t == "code":
            f = Path(a.code_doc_ids)
            doc_ids = np.load(f) if f.suffix == ".npy" else np.loadtxt(f, dtype=np.int64, ndmin=1)
        meta = {"checkpoint": str(cfg.get("_checkpoint")), "n_new": a.n_new, "n_prompts": a.n_prompts,
                "min_doc": min_doc, "val_tokens": a.val_tokens, "precision": str(amp_dtype)}
        path = cache / f"records_{t}.npz"
        recs = None
        if path.exists():
            recs, old = load_records(path)
            if any(old.get(k) != v for k, v in meta.items()) or old.get("prompt_len") != plen:
                recs = None
        if recs is None:
            prompts, ids = select_prompts(val, docs, a.n_prompts, plen, min_doc, doc_ids)
            if not len(prompts):
                info[t] = {"skipped": f"no val document with >= {min_doc} tokens"}
                continue
            t0 = time.time()
            recs = generate_records(model, prompts, a.n_new, t, ids, a.gen_batch, amp_dtype)
            save_records(path, recs, meta)
            print(f"[accept] generated {len(recs)} {t} records in {time.time() - t0:.1f}s", flush=True)
        recs_by_type[t] = recs
        info[t] = {"n_prompts": len(recs), "prompt_len": plen, "min_doc_len": min_doc, "n_new": a.n_new,
                   "cache": str(path)}
    return recs_by_type, info


def main(argv=None) -> dict:
    from ffnmem.infer import load_model

    a = parse_args(argv)
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) or a.device == "cuda" else "cpu")
    amp = torch.float16 if (a.precision == "fp16" or (a.precision == "auto" and dev.type == "cuda")) else None
    max_len = max(a.open_len, a.long_len) + a.n_new + max(a.gammas) + 1
    model, cfg = load_model(a.target, device=dev, seq_len=max_len)
    name = a.name or str(cfg.get("name") or Path(a.target).name)
    recs_by_type, type_info = get_records(a, model, cfg, name, amp)
    c_draft = read_c_draft(a.c_draft, a.timings)
    out_path = Path(a.out or Path("results") / f"acceptance_{name}.json")
    result = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    result.update(target=name, checkpoint=cfg.get("_checkpoint"), gammas=list(a.gammas), n_boot=a.n_boot)
    result.setdefault("text_types", {}).update(type_info)
    result.setdefault("drafters", {})
    for s in a.drafters:
        spec = parse_spec(s)
        drafter = make_drafter(spec, model, amp)
        t0 = time.time()
        entry = {"kind": spec["kind"], "spec": spec["spec"], "online": spec["online"],
                 "verified": spec["verified"], "results": {}, "doc_ids": {}}
        for t, recs in recs_by_type.items():
            pp = per_prompt_accepted(drafter, recs, a.gammas)
            entry["results"][t] = {str(g): summarize(pp[g], g, c_draft.get(spec["label"]), a.n_boot)
                                   for g in a.gammas}
            entry["doc_ids"][t] = [int(r.doc_id) for r in recs]
        entry["offline_seconds"] = time.time() - t0
        result["drafters"][spec["label"]] = entry
        line = "  ".join(f"{t}/g{g}={entry['results'][t][str(g)].get('mean_accepted', float('nan')):.3f}"
                         for t in recs_by_type for g in a.gammas)
        print(f"[accept] {spec['label']}: {line}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[accept] wrote {out_path}", flush=True)
    return result


if __name__ == "__main__":
    main()
