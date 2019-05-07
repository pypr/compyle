import unittest
import pytest

import numpy as np


class TestStructMapping(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("SetupClass")
        pytest.importorskip("pycuda")
        from compyle.cuda import set_context
        set_context()

    def test_cuda_struct_mapping(self):
        from compyle.cuda import match_dtype_to_c_struct
        from pycuda import gpuarray
        # Given
        dtype = np.dtype([('l', np.int64),
                          ('i', np.uint8),
                          ('x', np.float32)])
        a = np.empty(1, dtype)
        a['l'] = 1.0
        a['i'] = 2
        a['x'] = 1.23

        # When
        gs1, code1 = match_dtype_to_c_struct(None, "junk", a.dtype)
        a_ga = a.astype(gs1)
        ga = gpuarray.to_gpu(a_ga)

        # Then
        result = ga.get()
        np.testing.assert_almost_equal(result.tolist(), a.tolist())
        self.assertFalse(a.dtype.fields == gs1.fields)
