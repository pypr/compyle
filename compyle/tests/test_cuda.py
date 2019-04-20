import unittest
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
#from compyle.cuda import make_sort_module
from compyle.thrust.sort import argsort
from compyle.array import wrap

def test_sort():
    length = 100
    a = np.array(np.random.rand(length), dtype=np.float32)
    print("---------------------- Unsorted -----------------------")
    print(a)
    b = wrap(a, backend='cuda')
    # Call Thrust!!
    print("----------------------- Sorted ------------------------")
    print(argsort(b).get())
    print("-------------------------------------------------------")


