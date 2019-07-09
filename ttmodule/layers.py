import math

import torch
import torch.nn.functional as F

from numpy import prod
from torch.nn import init

from .matrix import tt_to_matrix


class TTLinear(torch.nn.Module):
    def __init__(self, in_shape, out_shape, *, rank, bias=True,
                 reassemble=True):
        assert len(in_shape) == len(out_shape)
        if not isinstance(rank, (list, tuple)):
            rank = [1] + (len(in_shape) - 1) * [rank] + [1]

        if not all(isinstance(r, int) for r in rank):
            raise TypeError("`rank` must be an int or a list of ints.")

        assert len(rank) == len(out_shape) + 1
        assert rank[0] == rank[-1]

        super().__init__()

        self.reassemble, self.rank = reassemble, rank
        self.shape = in_shape, out_shape

        self.cores = torch.nn.ParameterList([
            torch.nn.Parameter(torch.Tensor(r0, n, m, r1))
            for r0, n, m, r1 in zip(rank[:-1], in_shape, out_shape, rank[1:])
        ])

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(*out_shape).flatten())
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        for core in self.cores:
            # WIP See the NIPS 2015 paper on the proper initialization
            init.normal_(core, std=0.02)

        if self.bias is not None:
            bound = 1. / math.sqrt(prod(self.shape[0]))
            init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self):
        return tt_to_matrix(self.shape, *self.cores).t()

    def forward(self, input, assemble=False):
        if self.reassemble:
            return F.linear(input, self.weight, self.bias)

        *head, tail = input.shape
        data = input.view(-1, *self.shape[0], 1)
        for core in self.cores:
            data = torch.tensordot(data, core, dims=[[1, -1], [1, 0]])

        # the first dim of `data` is squeezed `head`. The last is
        #  exactly `self.shape[1]`.
        output = data.reshape(*head, -1)
        if self.bias is not None:
            output += self.bias

        return output

    def extra_repr(self):
        shapes = ["x".join(map(str, nm)) for nm in zip(*self.shape)]
        return "[" + "]x[".join(shapes) + f"] ({repr(self.rank)[1:-1]})"
