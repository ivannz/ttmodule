from numpy import prod
from itertools import chain

from .tensor import tt_to_tensor, tensor_to_tt
from .tensor import tr_to_tensor


def get_shuffle(d):
    shuffle = list(chain(*enumerate(range(d, 2*d))))
    inverse = list(chain(range(0, 2 * d, 2), range(1, 2 * d, 2)))

    return shuffle, inverse


def invert(*permutation):
    key = permutation.__getitem__
    return sorted(range(len(permutation)), key=key)


def matrix_to_tt(matrix, shape, rank=None):
    assert len(shape[0]) == len(shape[1])

    # 1. factor the dimensions of the matrix and reshape into a 2d-dim tensor
    tensor = matrix.reshape(*shape[0], *shape[1])

    # 2. hierarchically shuffle the dimensions and squeeze
    shuffle, _ = get_shuffle(len(shape[0]))
    tensor = tensor.permute(shuffle).reshape([
        i * j for i, j in zip(*shape)
    ])

    # 3. get the TT-cores of the tensor and unsqueeze them
    cores = tensor_to_tt(tensor, rank=rank)
    for core, n, m in zip(cores, *shape):
        rm1k, *mid, rk = core.shape
        yield core.reshape(rm1k, n, m, rk)


def tt_to_matrix(shape, *cores):
    assert len(shape[0]) == len(shape[1])

    _, inverse = get_shuffle(len(shape[1]))
    # 1. assemble the 2d-dim tensor and undo the dimshuffle
    tensor = tt_to_tensor(*cores).permute(*inverse).contiguous()

    # 2. rehsape into a matrix, knowing that shape factor the dimensions
    return tensor.reshape(prod(shape[0]), prod(shape[1]))


def tr_to_matrix(shape, *cores, k=0):
    assert len(shape[0]) == len(shape[1])

    _, inverse = get_shuffle(len(shape[1]))
    tensor = tr_to_tensor(*cores, k=k).permute(*inverse).contiguous()

    return tensor.reshape(prod(shape[0]), prod(shape[1]))
