import pytest

import torch
import torch.nn.functional as F

import numpy as np


from torch.nn import Linear
from ttmodule.layers import TTLinear
from ttmodule.matrix import matrix_to_tt


def test_ttlayer():
    in_shape, out_shape = [11, 7, 3], [2, 3, 7]

    lin = Linear(np.prod(in_shape), np.prod(out_shape), bias=True).double()

    # get a tt decomposition and recover the maximal ranks
    shapes = in_shape, out_shape
    cores = list(matrix_to_tt(lin.weight.t(), shapes, rank=None))
    ranks = [1] + [core.shape[-1] for core in cores]
    tt_lin = TTLinear(*shapes, rank=ranks, bias=True,
                      reassemble=False).double()

    # forge a state_dict
    state_dict = dict(bias=lin.bias, **{
        f"cores.{i}": core for i, core in enumerate(cores)
    })

    tt_lin.load_state_dict(state_dict)

    assert torch.allclose(tt_lin.weight, lin.weight)
    assert torch.allclose(tt_lin.bias, lin.bias)


def test_reassemble():
    shape = [11, 7, 3], [2, 3, 7]

    tt_lin_slow = TTLinear(*shape, rank=5, bias=True, reassemble=False)

    x = torch.randn(11, 3, np.prod(shape[0]))

    p_tt = tt_lin_slow(x)
    p_mm = F.linear(x, tt_lin_slow.weight, tt_lin_slow.bias)

    assert torch.allclose(p_tt, p_mm)
    assert torch.allclose(p_mm, p_tt)

    tt_lin_fast = TTLinear(*shape, rank=5, bias=True, reassemble=False)

    status = tt_lin_fast.load_state_dict(tt_lin_slow.state_dict(), strict=False)

    assert not status.missing_keys
    assert not status.unexpected_keys

    p_slow, p_fast = tt_lin_slow(x), tt_lin_fast(x)

    assert torch.allclose(p_tt, p_mm)
    assert torch.allclose(p_mm, p_tt)
