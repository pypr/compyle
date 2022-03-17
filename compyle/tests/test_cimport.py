from genericpath import exists
from ntpath import join
import tempfile
import unittest
from unittest import TestCase
import numpy as np
from os.path import exists, expanduser, isdir, join
import sys
import os
from mako.template import Template


from compyle.cimport import Cmodule
from compyle.types import annotate
from compyle.ext_module import get_platform_dir, get_md5, get_ext_extension

dummy_module = '''
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

void f(int n, int* x, int* y)
{
    for(int i = 0; i < n; i++){
        y[i] = (2 * x[i]);
    }
}
'''
pybind = """

PYBIND11_MODULE(${name}, m) {

    m.def("${name}", [](py::array_t<int> x, py::array_t<int> y){
        return f(x.request().size, (int*)x.request().ptr,
        (int*)y.request().ptr);
    });
}
"""


class TestCmodule(TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def test_build(self):
        hash_fn = get_md5(dummy_module)
        name = f'm_{hash_fn}'
        pyb_template = Template(pybind)
        src_pybind = pyb_template.render(name=name)

        all_src = dummy_module + src_pybind
        mod = Cmodule(all_src, hash_fn=hash_fn, root=self.root)
        checksum = get_md5(dummy_module)
        self.assertTrue(mod.is_build_needed())

        mod.load()
        self.assertTrue(exists(join(self.root, 'build')))
        self.assertTrue(exists(join(self.root, 'm_' + checksum + '.cpp')))
        self.assertTrue(
            exists(join(self.root, f'{name}' + get_ext_extension())))
        self.assertFalse(mod.is_build_needed())


if __name__ == '__main__':
    unittest.main()
