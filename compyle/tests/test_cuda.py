import pytest

pytest.importorskip('pycuda')

import numpy as np
from compyle.thrust.sort import argsort
from compyle.array import wrap

def test_sort():
    length = 100
    a = np.array(np.random.rand(length), dtype=np.float32)
    b = wrap(a, backend='cuda')
    res_gpu = argsort(b).get()
    res_cpu = np.argsort(a)
    assert np.all(res_gpu == res_cpu)

