import unittest

import numpy as np
from pytest import importorskip

from ..config import use_config
from ..array import wrap
from ..types import annotate
from ..parallel import elementwise, Reduction, Scan


class TestChangeBackend(unittest.TestCase):
    def test_elementwise_late_binding(self):
        # Given/When
        @elementwise
        @annotate
        def axpb(i, y, x, a, b):
            y[i] = a*x[i] + b

        # Then
        self.assertIsNone(axpb.elementwise)

    def test_reduction_late_binding(self):
        # Given/When
        r = Reduction('a+b')

        # Then
        self.assertIsNone(r.reduction)

    def test_scan_late_binding(self):
        # Given/When
        @annotate
        def output_f(i, last_item, item, ary):
            ary[i] = item + last_item

        scan = Scan(output=output_f, scan_expr='a+b',
                    dtype=np.int32)

        # Then
        self.assertIsNone(scan.scan)

    def test_elementwise_supports_changing_backend(self):
        importorskip("pyopencl")

        # Given/When
        @elementwise
        @annotate
        def axpb(i, y, x, a, b):
            y[i] = a*x[i] + b

        # When
        a, b = 2.0, 1.5
        x = np.linspace(0, 2*np.pi, 100)
        y = np.zeros_like(x)
        y0 = a*x + b
        with use_config(use_opencl=True):
            x, y = wrap(x, y)
            axpb(y, x, a, b)
            y.pull()

        # Then
        np.testing.assert_array_almost_equal(y.data, y0)
        self.assertEqual(axpb.elementwise.backend, 'opencl')

        # When
        x, y = wrap(x.data, y.data)
        axpb.set_backend('cython')
        axpb(y, x, a, b)
        # Then
        np.testing.assert_array_almost_equal(y.data, y0)
        self.assertEqual(axpb.elementwise.backend, 'cython')

    def test_reduction_supports_changing_backend(self):
        importorskip("pyopencl")

        # Given
        r = Reduction('a+b')

        # When
        x = np.linspace(0, 1, 1000) / 1000
        x_orig = x.copy()
        expect = 0.5

        with use_config(use_opencl=True):
            x = wrap(x)
            result = r(x)

        # Then
        self.assertAlmostEqual(result, expect, 6)

        # When
        x = wrap(x_orig)
        r.set_backend('cython')
        result = r(x)

        # Then
        self.assertAlmostEqual(result, expect, 6)

    def test_scan_supports_changing_backend(self):
        importorskip("pyopencl")

        # Given/When
        @annotate
        def input_f(i, ary):
            return ary[i]

        @annotate
        def output_f(i, item, ary):
            ary[i] = item

        scan = Scan(input_f, output_f, 'a+b', dtype=np.int32)

        # When
        a = np.arange(10000, dtype=np.int32)
        data = a.copy()
        expect = np.cumsum(a)

        with use_config(use_opencl=True):
            a = wrap(a)
            scan(input=a, ary=a)
            a.pull()

        # Then
        np.testing.assert_array_almost_equal(a.data, expect)

        # When
        a = wrap(data)
        scan.set_backend('cython')
        scan(input=a, ary=a)
        a.pull()

        # Then
        np.testing.assert_array_almost_equal(a.data, expect)

    def test_wrap_is_identity_on_arrays_with_same_backend(self):
        # Given
        x = np.linspace(0, 1, 100)

        # When
        xw = wrap(x)

        res = wrap(xw)

        # Then
        self.assertIs(res, xw)

    def test_wrap_can_wrap_array_to_different_backend(self):
        importorskip("pyopencl")
        # Given
        x = np.linspace(0, 1, 100)

        # When
        xc = wrap(x)
        with use_config(use_opencl=True):
            xocl = wrap(xc)

        # Then
        self.assertEqual(xc.backend, 'cython')
        self.assertEqual(xocl.backend, 'opencl')
        np.testing.assert_array_almost_equal(xocl.data, xc.data)
