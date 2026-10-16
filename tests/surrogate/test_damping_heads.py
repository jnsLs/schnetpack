"""The soft-maximum damping heads: invariances, positivity, and the surrogate Hessian's structure."""

import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

import schnetpack as spk
from schnetpack import properties

from .conftest import CUTOFF, N_ATOM_BASIS, _representation

HEADS = {
    "softmax": lambda: spk.atomistic.SoftMaxDampingFactor(n_in=N_ATOM_BASIS),
    "pair": lambda: spk.atomistic.PairDampingFactor(n_in=N_ATOM_BASIS, cutoff=CUTOFF),
    "hessian": lambda: spk.atomistic.SurrogateHessianDamping(
        n_in=N_ATOM_BASIS, n_vector_channels=3, cutoff=CUTOFF
    ),
}


def _model(head):
    torch.manual_seed(0)
    return spk.model.NeuralNetworkPotential(
        representation=_representation(),
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[head],
    )


def _run(model, batch):
    return model({k: v.clone() for k, v in batch.items()})


@pytest.fixture(params=list(HEADS))
def head_model(request):
    return _model(HEADS[request.param]())


def test_one_non_negative_damping_per_molecule(head_model, newton_batch):
    lam = _run(head_model, newton_batch)[properties.damping_factor]
    assert lam.shape == (3,)
    assert (lam >= 0).all() and torch.isfinite(lam).all()


def test_damping_is_invariant_to_rotation_and_translation(head_model, newton_batch):
    lam = _run(head_model, newton_batch)[properties.damping_factor]
    rotation = torch.tensor(
        Rotation.random(random_state=3).as_matrix(), dtype=torch.float32
    )
    moved = dict(newton_batch)
    moved[properties.R] = newton_batch[properties.R] @ rotation.T + torch.tensor(
        [1.0, -2.0, 0.5]
    )
    lam_moved = _run(head_model, moved)[properties.damping_factor]
    torch.testing.assert_close(lam_moved, lam, rtol=1e-4, atol=1e-4)


def test_damping_is_invariant_to_atom_order_within_a_molecule(head_model, newton_batch):
    """Swap the two H of water (atoms 1, 2), relabelling the neighbor list accordingly."""
    lam = _run(head_model, newton_batch)[properties.damping_factor]
    perm = torch.arange(newton_batch[properties.Z].shape[0])
    perm[1], perm[2] = 2, 1
    inverse = torch.argsort(perm)
    swapped = dict(newton_batch)
    for key in (properties.Z, properties.R):
        swapped[key] = newton_batch[key][perm]
    for key in (properties.idx_i, properties.idx_j):
        swapped[key] = inverse[newton_batch[key]]
    lam_swapped = _run(head_model, swapped)[properties.damping_factor]
    torch.testing.assert_close(lam_swapped, lam, rtol=1e-5, atol=1e-5)


def test_gradients_reach_every_head_parameter(head_model, newton_batch):
    _run(head_model, newton_batch)[properties.damping_factor].sum().backward()
    head = head_model.output_modules[0]
    missing = [
        n
        for n, p in head.named_parameters()
        if p.grad is None or p.grad.abs().sum() == 0
    ]
    assert not missing, f"no gradient reached: {missing}"


def test_segment_soft_max_is_bounded_by_the_hard_max():
    torch.manual_seed(0)
    x = torch.randn(10)
    segment = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    tau = torch.tensor(0.3)
    soft = spk.atomistic.segment_soft_max(x, segment, 3, tau)
    hard = torch.stack([x[segment == s].max() for s in range(3)])
    counts = torch.bincount(segment).float()
    assert (soft >= hard - 1e-6).all()
    assert (soft <= hard + tau * torch.log(counts) + 1e-6).all()


def test_segment_soft_max_with_zero_weights_ignores_those_entries():
    x = torch.tensor([5.0, 1.0, 2.0])
    segment = torch.zeros(3, dtype=torch.long)
    tau = torch.tensor(0.1)
    soft = spk.atomistic.segment_soft_max(
        x, segment, 1, tau, weights=torch.tensor([0.0, 1.0, 1.0])
    )
    expected = tau * torch.logsumexp(torch.tensor([1.0, 2.0]) / tau, 0)
    torch.testing.assert_close(soft, expected.reshape(1))


@pytest.fixture
def hessian_model():
    return _model(HEADS["hessian"]())


def test_surrogate_hessian_is_symmetric_with_translation_zero_modes(
    hessian_model, newton_batch
):
    out = _run(hessian_model, newton_batch)
    H = out["surrogate_hessian"]
    assert H.shape == (3, 12, 12)  # NH3 has 4 atoms, the largest
    torch.testing.assert_close(H, H.transpose(1, 2))
    for m, n in enumerate(newton_batch[properties.n_atoms].tolist()):
        for c in range(3):
            t = torch.zeros(12)
            t[c : 3 * n : 3] = 1.0
            torch.testing.assert_close(H[m] @ t, torch.zeros(12), atol=1e-4, rtol=0)


def test_padding_does_not_change_the_damping(hessian_model, newton_batch):
    """Each molecule alone (no padding) must give the same damping as inside the padded batch."""
    head = hessian_model.output_modules[0]
    lam = _run(hessian_model, newton_batch)[properties.damping_factor]
    H = _run(hessian_model, newton_batch)["surrogate_hessian"]
    for m, n in enumerate(newton_batch[properties.n_atoms].tolist()):
        e = torch.linalg.eigvalsh(H[m, : 3 * n, : 3 * n])
        expected = torch.nn.functional.softplus(
            head.offset + head.tau * torch.logsumexp(-e / head.tau, 0)
        )
        torch.testing.assert_close(lam[m], expected, rtol=1e-5, atol=1e-5)


def test_dense_assembly_matches_a_loop_over_pairs():
    torch.manual_seed(0)
    n_atoms = torch.tensor([2, 3])
    idx_m = torch.tensor([0, 0, 1, 1, 1])
    idx_i = torch.tensor([0, 1, 2, 2, 3, 3, 4, 4])
    idx_j = torch.tensor([1, 0, 3, 4, 2, 4, 2, 3])
    blocks = torch.randn(8, 3, 3)
    dense = spk.atomistic.dense_from_pair_blocks(blocks, idx_i, idx_j, idx_m, n_atoms)
    expected = torch.zeros(2, 9, 9)
    first = [0, 0, 2]
    local = [0, 1, 0, 1, 2]
    for p in range(8):
        i, j = int(idx_i[p]), int(idx_j[p])
        m = int(idx_m[i])
        li, lj = local[i], local[j]
        expected[m, 3 * li : 3 * li + 3, 3 * lj : 3 * lj + 3] -= blocks[p]
        expected[m, 3 * li : 3 * li + 3, 3 * li : 3 * li + 3] += blocks[p]
    torch.testing.assert_close(dense, expected)
