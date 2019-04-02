from cupy.cuda.thrust import argsort as _argsort
from libcpp.vector cimport vector
import compyle.array as carr
import numpy as np

cpdef argsort(array):
    idx_array = carr.empty(array.length, np.intp,
                           backend='cuda')

    cdef vector[int] shape
    shape.push_back(<int> array.length)

    _argsort(array.dtype, idx_array.dev.ptr, array.dev.ptr, 0, shape)

    return idx_array
