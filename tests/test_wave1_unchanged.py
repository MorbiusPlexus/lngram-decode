"""M1.T0: Wave 3 code must reproduce Wave 1 models exactly.

For each Wave 1 / pivot config: parameter counts equal the recorded values, and a
forward pass (train and eval mode, seq 64) gives bitwise-identical logits to the
frozen pre-Wave-3 model in ``tests/wave1_ref``.  Stored reference slices in
``tests/data/wave1_reference.npz`` are checked too (exact on the machine/torch
version that wrote them, close elsewhere, since CPU kernels differ across hosts).

    python tests/test_wave1_unchanged.py --write   # (re)write the stored slices
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ffnmem.config import load_config  # noqa: E402
from ffnmem.model import Transformer, count_params  # noqa: E402
from tests.wave1_ref.model import Transformer as RefTransformer  # noqa: E402

CFG = ROOT / "configs"
REF_FILE = ROOT / "tests" / "data" / "wave1_reference.npz"
SEQ = 64
RECORDED = {
    "w1_mem_full": 369_767_936,
    "w1_mem_cut_near": 369_767_936,
    "w1_dense_full": 370_459_648,
    "w2_mem_super1": 369_777_920,
}


def _model_cfg(name: str) -> dict:
    return load_config(CFG / f"{name}.yaml", [f"model.seq_len={SEQ}"])["model"]


def _build(cls, name: str):
    torch.manual_seed(0)
    model = cls(_model_cfg(name))
    # Memory output projections are zero at init; perturb so the memory path matters.
    torch.manual_seed(1)
    for b in model.blocks:
        if b.memory is not None:
            torch.nn.init.normal_(b.memory.module.readout_o_proj.weight, std=0.02)
    return model


def _batch() -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return torch.randint(0, 32768, (1, SEQ), generator=g)


def _logits(model, x: torch.Tensor, train: bool) -> torch.Tensor:
    model.train(train)
    with torch.no_grad():
        return model(x)


@pytest.mark.parametrize("name", sorted(RECORDED))
def test_param_counts(name):
    with torch.device("meta"):
        model = Transformer(_model_cfg(name))
    assert count_params(model)["non_embedding"] == RECORDED[name]


@pytest.mark.parametrize("name", sorted(RECORDED))
def test_logits_bitwise_vs_frozen(name):
    new, ref = _build(Transformer, name), _build(RefTransformer, name)
    assert new.state_dict().keys() == ref.state_dict().keys()
    for k, v in ref.state_dict().items():
        assert torch.equal(new.state_dict()[k], v), k
    x = _batch()
    for train in (False, True):
        assert torch.equal(_logits(new, x, train), _logits(ref, x, train)), (name, train)


def _env_tag() -> str:
    return f"{torch.__version__}|{platform.machine()}|{platform.system()}"


@pytest.mark.skipif(not REF_FILE.exists(), reason="no stored reference (run with --write)")
@pytest.mark.parametrize("name", sorted(RECORDED))
def test_logits_vs_stored(name):
    data = np.load(REF_FILE)
    got = _logits(_build(Transformer, name), _batch(), False)[..., :256].numpy()
    want = data[name]
    if str(data["env"]) == _env_tag():
        assert np.array_equal(got, want)
    else:
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def write_reference() -> None:
    REF_FILE.parent.mkdir(parents=True, exist_ok=True)
    out = {"env": np.array(_env_tag())}
    for name in sorted(RECORDED):
        out[name] = _logits(_build(RefTransformer, name), _batch(), False)[..., :256].numpy()
    np.savez_compressed(REF_FILE, **out)
    print(f"wrote {REF_FILE} ({_env_tag()})")


if __name__ == "__main__":
    if "--write" in sys.argv:
        write_reference()
