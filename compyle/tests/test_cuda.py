import pytest

pytest.importorskip('pycuda')

from compyle.array import wrap
from compyle.thrust.sort import argsort
import numpy as np


def test_sort():
    length = 100
    a = np.array(np.random.rand(length), dtype=np.float32)
    b = wrap(a, backend='cuda')
    res_gpu = argsort(b).get()
    res_cpu = np.argsort(a)
    assert np.all(res_gpu == res_cpu)
