import cupy.cuda.thrust as thrust
from libcpp.vector cimport vector
import compyle.array as carr
import numpy as np


cpdef argsort(array, keys=None):
    idx_array = carr.empty(array.length, np.intp, backend='cuda')

    cdef vector[int] shape
    shape.push_back(<int> array.length)

    cdef size_t keys_ptr
    if keys:
        keys_ptr = <size_t> keys.dev.ptr
    else:
        keys_ptr = 0

    thrust.argsort(array.dtype, idx_array.dev.ptr, array.dev.ptr, keys_ptr, shape)

    return idx_array
