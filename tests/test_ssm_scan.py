from __future__ import annotations

import torch

from ffnmem.memory.ssm import DiagonalSSM, chunked_scan, sequential_scan


def test_chunked_scan_matches_loop_t2048():
    torch.manual_seed(0)
    ssm = DiagonalSSM(64, state_dim=256, chunk=128)
    a = ssm.decay()
    assert float(a.min()) >= 0.5 - 1e-6 and float(a.max()) <= 0.999 + 1e-6
    u = torch.randn(2, 2048, 256)
    torch.testing.assert_close(chunked_scan(a, u, 128), sequential_scan(a, u), rtol=0, atol=1e-5)


def test_chunked_scan_ragged_length_and_random_decays():
    torch.manual_seed(1)
    a = torch.rand(16) * 0.999
    u = torch.randn(3, 301, 16)
    torch.testing.assert_close(chunked_scan(a, u, 64), sequential_scan(a, u), rtol=0, atol=1e-5)


def test_step_recurrence_matches_forward():
    torch.manual_seed(0)
    ssm = DiagonalSSM(32, state_dim=8, chunk=4)
    emb = torch.randn(2, 13, 32)
    full = ssm(emb)
    state = None
    for t in range(13):
        state = ssm.step(state, emb[:, t])
        torch.testing.assert_close(state, full[:, t], rtol=0, atol=1e-5)
