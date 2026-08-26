"""The MD-ET interface, checked without any MD-ET weights.

The parts that can go wrong silently are the replacements for MD-ET's fused
kernels: they have to compute what MD-ET computes (checked against MD-ET's own
reference fallbacks) and, unlike the fused kernels, be twice differentiable
(checked directly).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("md_et")

from md_et.nn.triplet_attention import sparse_triplet_attention_fallback
from md_et.nn.triplet_attention_sigmoid import sparse_triplet_attention_sigmoid_fallback

from schnetpack.interfaces.mdet import (
    _edge_geometry,
    triplet_attention_sigmoid,
    triplet_attention_softmax,
)


@pytest.fixture
def csr():
    """A small triplet CSR: 6 edges, rows of varying length, one empty row."""
    torch.manual_seed(0)
    counts = [3, 2, 0, 4, 1]
    offsets = torch.tensor([0] + list(torch.tensor(counts).cumsum(0)), dtype=torch.int64)
    n_triplets = int(offsets[-1])
    n_edges = 6
    src = torch.randint(0, n_edges, (n_triplets,), dtype=torch.int32)
    # dst is constant within a row, as both attention variants assume
    dst = torch.cat(
        [torch.full((c,), i % n_edges, dtype=torch.int32) for i, c in enumerate(counts)]
    )
    return src, dst, offsets, n_edges, n_triplets


@pytest.fixture
def qkvv():
    torch.manual_seed(1)
    return torch.randn(6, 4 * 8, dtype=torch.float64, requires_grad=True)


def test_softmax_matches_mdet_fallback(csr, qkvv):
    # in the mediator-keyed CSR the softmax path reads, the destination edge
    # varies within a row -- unlike the dst-keyed layout the sigmoid path needs
    src, dst, offsets, n_edges, n_triplets = csr
    dst = torch.randint(0, n_edges, (n_triplets,), dtype=torch.int32)
    env_gate = torch.rand(n_triplets, dtype=torch.float64)

    ours = triplet_attention_softmax(qkvv, src, dst, 2, offsets, env_gate=env_gate)
    theirs = sparse_triplet_attention_fallback(qkvv, src, dst, 2, offsets, env_gate=env_gate)
    assert torch.allclose(ours, theirs, atol=1e-10)


def test_sigmoid_matches_mdet_fallback(csr, qkvv):
    src, dst, offsets, n_edges, n_triplets = csr
    env_pair = torch.rand(n_triplets, dtype=torch.float64)
    inv_sqrt_K = torch.rand(n_edges, dtype=torch.float64) + 0.5
    bias = torch.randn(2, dtype=torch.float64)

    ours = triplet_attention_sigmoid(
        qkvv, src, dst, 2, offsets, env_pair, inv_sqrt_K, bias
    )
    theirs = sparse_triplet_attention_sigmoid_fallback(
        qkvv, src, dst, 2, offsets, env_pair, inv_sqrt_K, bias
    )
    assert torch.allclose(ours, theirs, atol=1e-10)


def test_softmax_is_twice_differentiable(csr, qkvv):
    src, dst, offsets, n_edges, n_triplets = csr
    dst = torch.randint(0, n_edges, (n_triplets,), dtype=torch.int32)
    env_gate = (torch.rand(n_triplets, dtype=torch.float64)).requires_grad_(True)

    out = triplet_attention_softmax(qkvv, src, dst, 2, offsets, env_gate=env_gate)
    (grad_qkvv,) = torch.autograd.grad(out.sum(), qkvv, create_graph=True)
    # the second derivative has to reach both the weights and the envelope
    second = torch.autograd.grad(grad_qkvv.sum(), [qkvv, env_gate], allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() for g in second)


def test_sigmoid_is_twice_differentiable(csr, qkvv):
    src, dst, offsets, n_edges, n_triplets = csr
    env_pair = torch.rand(n_triplets, dtype=torch.float64, requires_grad=True)
    inv_sqrt_K = torch.rand(n_edges, dtype=torch.float64) + 0.5
    bias = torch.randn(2, dtype=torch.float64, requires_grad=True)

    out = triplet_attention_sigmoid(qkvv, src, dst, 2, offsets, env_pair, inv_sqrt_K, bias)
    (grad_qkvv,) = torch.autograd.grad(out.sum(), qkvv, create_graph=True)
    second = torch.autograd.grad(grad_qkvv.sum(), [qkvv, env_pair, bias], allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() for g in second)


def test_edge_geometry_matches_the_plain_norm_away_from_self_loops():
    torch.manual_seed(2)
    positions = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    i_idx = torch.tensor([0, 1, 2, 3])
    j_idx = torch.tensor([1, 2, 3, 4])

    diff, d = _edge_geometry(positions, i_idx, j_idx)
    reference = (positions[j_idx] - positions[i_idx]).norm(dim=-1)
    assert torch.allclose(d, reference)
    assert torch.autograd.gradcheck(
        lambda p: _edge_geometry(p, i_idx, j_idx)[1], (positions,)
    )


def test_self_loops_are_constant_and_do_not_move_the_gradient():
    """A self-loop edge is position-independent, so it must contribute nothing.

    Left to autograd it contributes ``+g`` and ``-g`` to the same atom, which
    cancels -- but only after both have been added to the real gradient, which
    in float32 is where the precision goes.
    """
    torch.manual_seed(3)
    positions = torch.randn(4, 3, requires_grad=True)
    i_idx = torch.tensor([0, 1, 2, 3, 0, 1])
    j_idx = torch.tensor([0, 1, 2, 3, 1, 2])

    diff, d = _edge_geometry(positions, i_idx, j_idx)
    assert torch.equal(d[:4], torch.zeros(4))
    assert torch.isfinite(d).all()

    # only the two real edges may carry a derivative
    grad = torch.autograd.grad((d[:4] * 1e6).sum(), positions, allow_unused=True)[0]
    assert grad is None or torch.equal(grad, torch.zeros_like(positions))

    # and the norm is differentiable a second time despite the zeros
    diff, d = _edge_geometry(positions, i_idx, j_idx)
    (g,) = torch.autograd.grad(d.sum(), positions, create_graph=True)
    (h,) = torch.autograd.grad(g.sum(), positions)
    assert torch.isfinite(h).all()
