"""lm-evaluation-harness adapter for our checkpoints (M6.T1).

    python -m eval.lm_eval_adapter --checkpoint runs/w3_sb_1b --name SB \\
        --tasks hellaswag,mmlu_continuation,sciq,winogrande,piqa,arc_easy --device cuda --batch_size 16
    # -> results/lmeval/SB/results.json + samples_<task>.jsonl (per-sample acc / acc_norm / log-likelihoods)

    # the stock lm-eval CLI, with the ``morbius`` model registered first:
    python -m eval.lm_eval_adapter lm_eval --model morbius \\
        --model_args checkpoint=runs/w3_sb_1b,device=cuda --tasks piqa --log_samples --output_path out/

Scoring (``MorbiusScorer``) does not depend on lm-eval: requests are token-id pairs
(context, continuation), sorted by length (longest first), right-padded into
batches and scored with one causal forward each; the window is ``(ctx + cont)``
truncated from the left to ``seq_len + 1`` tokens.  Log-softmax is taken in fp32
over the continuation positions only; fp16 autocast on CUDA.  Empty contexts
are conditioned on EOS (id 2, the document separator in training).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ffnmem.infer import DecodeState, decode_step, load_model  # noqa: E402

DEFAULT_TOKENIZER = ROOT / "data" / "tokenizer" / "tokenizer.json"
EOS_ID = 2
DEFAULT_TASKS = "hellaswag,mmlu_continuation,sciq,winogrande,piqa,arc_easy"


def amp_dtype_for(device: torch.device, precision: str) -> torch.dtype | None:
    if precision == "fp16" and device.type == "cuda":
        return torch.float16
    if precision not in ("fp16", "fp32"):
        raise ValueError(f"unsupported precision {precision!r} (fp16 | fp32)")
    return None


class MorbiusScorer:
    """Token-level scoring and greedy generation for one model (no lm-eval dependency)."""

    def __init__(self, model: torch.nn.Module, seq_len: int, batch_size: int = 16, precision: str = "fp16"):
        self.model = model.eval()
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.device = next(model.parameters()).device
        self.amp = amp_dtype_for(self.device, precision)

    def _autocast(self):
        if self.amp is None:
            return torch.autocast(self.device.type, enabled=False)
        return torch.autocast(self.device.type, dtype=self.amp)

    def window(self, context: list[int], continuation: list[int]) -> tuple[list[int], list[int]]:
        """(input ids, target ids of the scored continuation) after left truncation."""
        if not continuation:
            raise ValueError("empty continuation")
        if not context:
            context = [EOS_ID]
        whole = (list(context) + list(continuation))[-(self.seq_len + 1):]
        n_cont = min(len(continuation), len(whole) - 1)  # continuation > seq_len: score its last seq_len tokens
        return whole[:-1], whole[-n_cont:]

    @torch.no_grad()
    def loglikelihood_tokens(self, requests: list[tuple[list[int], list[int]]]) -> list[tuple[float, bool]]:
        """[(context ids, continuation ids)] -> [(sum log p(continuation), continuation is greedy)]."""
        windows = [self.window(c, t) for c, t in requests]
        order = sorted(range(len(windows)), key=lambda i: (-len(windows[i][0]), i))
        out: list[tuple[float, bool] | None] = [None] * len(windows)
        for s in range(0, len(order), self.batch_size):
            chunk = order[s:s + self.batch_size]
            width = max(len(windows[i][0]) for i in chunk)
            x = torch.zeros(len(chunk), width, dtype=torch.long)
            for r, i in enumerate(chunk):
                inp = windows[i][0]
                x[r, :len(inp)] = torch.tensor(inp)
            with self._autocast():
                logits = self.model(x.to(self.device))
            for r, i in enumerate(chunk):
                inp, tgt = windows[i]
                n = len(tgt)
                lp = F.log_softmax(logits[r, len(inp) - n:len(inp)].float(), dim=-1)
                t = torch.tensor(tgt, device=lp.device)
                ll = float(lp.gather(-1, t[:, None]).sum())
                greedy = bool((lp.argmax(-1) == t).all())
                out[i] = (ll, greedy)
        return out  # type: ignore[return-value]

    def rolling_requests(self, tokens: list[int]) -> list[tuple[list[int], list[int]]]:
        """Disjoint chunks of ``tokens`` (each predicted once, conditioned on EOS + maximal left context)."""
        seq = [EOS_ID] + list(tokens)
        return [(seq[:a], seq[a:a + self.seq_len]) for a in range(1, len(seq), self.seq_len)]

    def loglikelihood_rolling_tokens(self, docs: list[list[int]]) -> list[float]:
        reqs, owner = [], []
        for d, toks in enumerate(docs):
            for r in self.rolling_requests(toks):
                reqs.append(r)
                owner.append(d)
        totals = [0.0] * len(docs)
        for d, (ll, _) in zip(owner, self.loglikelihood_tokens(reqs)):
            totals[d] += ll
        return totals

    @torch.no_grad()
    def generate_tokens(self, context: list[int], max_new: int, stop=None) -> list[int]:
        """Greedy continuation (cached decoding); ``stop(new_ids) -> bool`` ends early."""
        max_new = max(1, min(int(max_new), self.seq_len - 1))
        ctx = (list(context) or [EOS_ID])[-(self.seq_len - max_new):]
        state = DecodeState.empty(self.model, 1, len(ctx) + max_new)
        new: list[int] = []
        with self._autocast():
            logits = decode_step(self.model, state, torch.tensor([ctx], device=self.device))
            while True:
                nxt = int(logits[0, -1].argmax())
                new.append(nxt)
                if nxt == EOS_ID or len(new) >= max_new or (stop is not None and stop(new)):
                    break
                logits = decode_step(self.model, state, torch.tensor([[nxt]], device=self.device))
        return new


def load_tokenizer(path: str | Path = DEFAULT_TOKENIZER):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


# ------------------------------------------------------------------ lm-eval wrapper
try:
    from lm_eval.api.model import TemplateLM
    from lm_eval.api.registry import register_model
except ImportError:  # scoring stays usable without lm-eval
    TemplateLM = None  # type: ignore[assignment,misc]

if TemplateLM is not None:
    try:
        import lm_eval.models  # noqa: F401  (populate the stock registry before adding ours)
    except Exception:  # pragma: no cover
        pass

    @register_model("morbius")
    class MorbiusLM(TemplateLM):
        """``lm_eval --model morbius --model_args checkpoint=PATH[,device=cuda,tokenizer=...,precision=fp16]``."""

        def __init__(self, checkpoint: str | None = None, device: str = "cuda", batch_size: int | str = 16,
                     tokenizer: str | None = None, precision: str = "fp16", max_length: int | None = None,
                     model=None, tok=None, **kwargs):
            super().__init__()
            if model is None:
                if checkpoint is None:
                    raise ValueError("morbius: checkpoint=... is required")
                model, cfg = load_model(checkpoint, device=device)
                self.cfg = cfg
            else:
                self.cfg = {}
            seq_len = int(max_length) if max_length else int(model.cfg["seq_len"])
            if batch_size in (None, "auto") or str(batch_size).startswith("auto"):
                batch_size = 16
            self.scorer = MorbiusScorer(model, seq_len, int(batch_size), precision)
            self._device = self.scorer.device
            self.tok = tok if tok is not None else load_tokenizer(tokenizer or DEFAULT_TOKENIZER)

        @classmethod
        def create_from_arg_obj(cls, arg_dict: dict, additional_config: dict | None = None):
            extra = {k: v for k, v in (additional_config or {}).items() if v is not None and k not in arg_dict}
            return cls(**arg_dict, **extra)  # model_args win over the CLI's --device / --batch_size

        @classmethod
        def create_from_arg_string(cls, arg_string: str, additional_config: dict | None = None):
            from lm_eval.utils import simple_parse_args_string

            return cls.create_from_arg_obj(simple_parse_args_string(arg_string), additional_config)

        @property
        def eot_token_id(self) -> int:
            return EOS_ID

        @property
        def max_length(self) -> int:
            return self.scorer.seq_len

        @property
        def tokenizer_name(self) -> str:
            return "morbius-bpe32k"

        def tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs) -> list[int]:
            return list(self.tok.encode(string, add_special_tokens=False).ids)

        def tok_decode(self, tokens: list[int], **kwargs) -> str:
            return self.tok.decode(list(tokens), skip_special_tokens=False)

        def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, **kwargs):
            res = self.scorer.loglikelihood_tokens([(c, t) for _, c, t in requests])
            for (key, _, _), r in zip(requests, res):
                self.cache_hook.add_partial("loglikelihood", key, r)
            return res

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
            strings = [req.args[0] for req in requests]
            res = self.scorer.loglikelihood_rolling_tokens([self.tok_encode(s) for s in strings])
            for s, r in zip(strings, res):
                self.cache_hook.add_partial("loglikelihood_rolling", (s,), r)
            return res

        def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
            out = []
            for req in requests:
                context, gen_kwargs = req.args
                gen_kwargs = dict(gen_kwargs or {})
                until = gen_kwargs.get("until") or []
                until = [until] if isinstance(until, str) else list(until)
                max_new = int(gen_kwargs.get("max_gen_toks", 256))

                def stop(ids: list[int]) -> bool:
                    text = self.tok_decode(ids)
                    return any(u in text for u in until)

                ids = self.scorer.generate_tokens(self.tok_encode(context), max_new, stop if until else None)
                if ids and ids[-1] == EOS_ID:
                    ids = ids[:-1]
                text = self.tok_decode(ids)
                for u in until:
                    text = text.split(u)[0]
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text)
                out.append(text)
            return out
else:
    MorbiusLM = None  # type: ignore[assignment,misc]


# ------------------------------------------------------------------ outputs
def compact_sample(task: str, s: dict) -> dict:
    """Per-sample record for bootstrapping: metric values + per-choice log-likelihoods."""
    rec = {"task": task, "doc_id": s.get("doc_id"), "target": s.get("target")}
    for k in s.get("metrics", []):
        if k in s:
            rec[k] = s[k]
    try:
        rec["lls"] = [float(r[0]) if isinstance(r, (list, tuple)) else float(r) for r in s.get("filtered_resps", [])]
    except (TypeError, ValueError):
        pass
    return rec


def write_outputs(res: dict, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for task, samples in (res.get("samples") or {}).items():
        with open(out_dir / f"samples_{task}.jsonl", "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(compact_sample(task, s), default=str) + "\n")
    summary = {"meta": meta, "results": res.get("results"), "groups": res.get("groups"),
               "versions": res.get("versions"), "n-samples": res.get("n-samples")}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "lm_eval":  # stock CLI with ``morbius`` registered
        from lm_eval.__main__ import cli_evaluate

        sys.argv = ["lm_eval"] + argv[1:]
        cli_evaluate()
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="checkpoint file or run dir (latest checkpoint)")
    ap.add_argument("--name", default=None, help="label; default: checkpoint's run name")
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--limit", type=float, default=None, help="docs per task (debugging)")
    ap.add_argument("--out_dir", default=None, help="default: results/lmeval/<name>")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)
    if MorbiusLM is None:
        raise SystemExit("lm-eval is not installed (uv pip install lm-eval)")
    import lm_eval

    lm = MorbiusLM(checkpoint=args.checkpoint, device=args.device, batch_size=args.batch_size,
                   tokenizer=args.tokenizer, precision=args.precision)
    name = args.name or lm.cfg.get("name") or Path(args.checkpoint).name
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "lmeval" / name
    limit = None if args.limit is None else (int(args.limit) if args.limit >= 1 else args.limit)
    res = lm_eval.simple_evaluate(model=lm, tasks=args.tasks.split(","), num_fewshot=args.num_fewshot,
                                  limit=limit, log_samples=True, bootstrap_iters=0, random_seed=args.seed,
                                  numpy_random_seed=args.seed, torch_random_seed=args.seed,
                                  fewshot_random_seed=args.seed)
    meta = {"name": name, "checkpoint": lm.cfg.get("_checkpoint", args.checkpoint), "step": lm.cfg.get("_step"),
            "config": lm.cfg.get("name"), "tasks": args.tasks, "num_fewshot": args.num_fewshot,
            "precision": args.precision, "seq_len": lm.max_length, "limit": args.limit,
            "lm_eval_version": getattr(lm_eval, "__version__", None)}
    write_outputs(dict(res), out_dir, meta)
    for task, r in sorted((res.get("results") or {}).items()):
        vals = " ".join(f"{k}={v:.4f}" for k, v in r.items() if isinstance(v, float))
        print(f"[{name}] {task}: {vals}", flush=True)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
