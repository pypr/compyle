import unittest
from unittest import TestCase
from ..c_backend import CBackend, CCompile
from ..types import annotate
import numpy as np


class TestCBackend(TestCase):
    def test_get_func_signature(self):
        cbackend = CBackend()

        @annotate(x='int', y='intp', z='int', w='double')
        def test_fn(x, y, z=2, w=3.0):
            return x+y+z+w
        temp = cbackend.get_func_signature(test_fn)
        (pyb11_args, pyb11_call), (c_args, c_call) = temp
        exp_pyb11_args = ['int x', 'int[:] y', 'int z', 'double w']
        exp_pyb11_call = ['x', '&y[0]', 'z', 'w']
        exp_c_args = ['int x', 'int* y', 'int z', 'double w']
        exp_c_call = ['x', 'y', 'z', 'w']

        self.assertListEqual(pyb11_args, exp_pyb11_args)
        self.assertListEqual(pyb11_call, exp_pyb11_call)
        self.assertListEqual(c_args, exp_c_args)
        self.assertListEqual(c_call, exp_c_call)

class TestCCompile(TestCase):
    def test_compile(self):
        @annotate(int='n, p', intp='x, y')
        def get_pow(n, p, x, y):
            for i in range(n):
                y[i] = x[i]**p
        c_get_pow = CCompile(get_pow)
        n = 5
        p = 5
        x = np.ones(n, dtype=np.int32) * 2
        y = np.zeros(n, dtype=np.int32)
        y_exp = np.ones(n, dtype=np.int32) * 32
        c_get_pow(n, p, x, y)
        assert(np.all(y == y_exp))
        
if __name__ == '__main__':
    unittest.main()
