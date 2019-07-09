import torch


def tensor_to_tt(tensor, rank=None):
    """TT-SVD from [1]_ algorithm 1 (p. 2301).
    Originally implmented in October 2015

    Arguments
    ---------
    tensor : torch.Tensor
        The tensor, for which to compute the TT representation.

    rank : int, list of ints, or None
        The maximal rank of each computed tensor train core. Uses the
        maximal computed rank by the TT-svd alogirthm if set to None.

    Returns
    -------
    cores : generator
        The TT core generator.

    References
    ----------
    .. [1] Oseledets, I. V. (2011). 'Tensor-Train decomposition.' SIAM Journal
           of Scientific Computing, vol. 33, no. 5, pp. 2295-2317.
    """
    if not isinstance(rank, (list, tuple)):
        rank = (tensor.dim() - 1) * [rank]

    r_km1, shape = 1, tensor.shape
    for k, n_k in enumerate(shape[:-1]):
        # L4: reshape into a rectangular matrix
        tensor = tensor.reshape(r_km1 * n_k, -1)

        # L5: the thin SVD of the matrix
        u, s, v = torch.svd(tensor, some=True)
        # u, v have min(n, m) = len(s_k) columns
        r_k = min(len(s), rank[k]) if rank[k] is not None else len(s)

        # L6: reshape and yield the next core
        yield u[:, :r_k].reshape(r_km1, n_k, r_k)

        # L7: Update the temporary tensor
        tensor = (v[:, :r_k] * s[:r_k]).t()
        r_km1 = r_k

    yield tensor.reshape(r_km1, shape[-1], 1)


def tt_to_tensor(*cores, squeeze=True):
    """Assemble the tensor from TT representation with eq. (1.3) from [1]_."""
    tensor = cores[0]
    for core in cores[1:]:
        tensor = torch.tensordot(tensor, core, dims=[[-1], [0]])

    return tensor.squeeze(0).squeeze(-1) if squeeze else tensor


def tr_to_tensor(*cores, k=0):
    """Assemble the tensor from a TR representation with eq. (1) from [2]_."""
    k = (len(cores) + k) if k < 0 else k
    assert 0 <= k < len(cores)

    # chip off the specified core and contract the rest of the cycle
    rest = tt_to_tensor(*cores[k+1:], *cores[:k], squeeze=False)

    # contract with tensordot (reshape einsum("i...j, j...i->...") was slower)
    output = torch.tensordot(cores[k], rest, dims=[[0, -1], [-1, 0]])

    # take the inner dims of the last k cores and roll back the dimensions
    tail = sum(core.dim() - 2 for core in cores[:k])
    if tail <= 0:
        return output

    dims = list(range(output.dim()))
    return output.permute(dims[-tail:] + dims[:-tail])
