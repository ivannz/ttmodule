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
