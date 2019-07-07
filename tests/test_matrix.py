import pytest

import torch

import numpy as np
from numpy.testing import assert_allclose


from ttmodule import matrix


@pytest.fixture
def random_state():
    return np.random.RandomState(None)  # (1249563438)


def test_shuffle():
    for d in range(100):
        s, i = matrix.get_shuffle(d)

        assert all(n == s[j] for n, j in enumerate(i))
        assert all(n == i[j] for n, j in enumerate(s))

        assert i == matrix.invert(*s)


def test_matrix_to_tt_and_back(random_state):
    a = random_state.randn(1000, 1000)

    x = torch.from_numpy(a)
    shape = [10, 5, 2, 10], [10, 10, 5, 2]

    z = matrix.tt_to_matrix(shape, *matrix.matrix_to_tt(x, shape))
    assert torch.allclose(z, x)
