from typing import Iterable, Union

import numpy as np


_prod_accepted = (
    int, bool,
    np.bool_, np.signedinteger, np.unsignedinteger
)

ProdAccepted = Union[
    int, bool,
    np.bool_, np.signedinteger, np.unsignedinteger
]


def prod(iterable: Iterable[ProdAccepted]):
    '''
    Safe product for large integer size calculations.

    :meth:`numpy.prod` uses 32 bit for default :code:`int` on Windows 64 bit. This
    function uses infinite width integers to calculate the product and
    throws a ValueError if it encounters types other than the supported ones.
    '''
    result = 1

    for item in iterable:
        if isinstance(item, _prod_accepted):
            result *= int(item)
        else:
            raise ValueError()
    return result


def count_nonzero(array):
    try:
        return np.count_nonzero(array)
    except TypeError:
        return array.nnz


def flat_nonzero(array):
    return array.flatten().nonzero()[0]


def _sparse_ndenumerate(array):
    flat_array = array.flatten()
    nonzero = flat_nonzero(flat_array)
    for idx in nonzero:
        coords = np.unravel_index(idx, array.shape)
        yield coords, flat_array[idx]


def ndenumerate(array):
    try:
        yield from np.ndenumerate(array)
    except RuntimeError:
        yield from _sparse_ndenumerate(array)
