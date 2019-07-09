import pytest

import torch

import numpy as np
from numpy.testing import assert_allclose


from ttmodule import tensor


@pytest.fixture
def random_state():
    return np.random.RandomState(None)  # (1249563438)


def test_tensor_to_tt_and_back(random_state):
    a = random_state.randn(10, 3, 10, 3, 10, 3, 10)

    x = torch.from_numpy(a)
    z = tensor.tt_to_tensor(*tensor.tensor_to_tt(x))

    assert torch.allclose(z, x)

    cores = list(tensor.tensor_to_tt(x))
    z = tensor.tt_to_tensor(*cores, squeeze=False)

    a, *body, z = z.shape
    assert a == 1 and z == 1

    assert all(core.shape[1] == b for b, core in zip(body, cores))


def test_tr_circularity(random_state):
    ranks = [3, 2, 3, 5]
    shape = [2, 3, 7, 5], [3, 7, 5, 2]

    cores = []
    for r0, n, m, r1 in zip(ranks[-1:] + ranks[:-1], *shape, ranks):
        cores.append(torch.randn(r0, n, m, r1, dtype=torch.double))

    base = tensor.tr_to_tensor(*cores, k=0)
    for k in range(1, len(cores)):
        res = tensor.tr_to_tensor(*cores, k=k)
        assert torch.allclose(res, base)
