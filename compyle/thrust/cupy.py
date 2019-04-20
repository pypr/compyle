from cupy import ndarray
from cupy.cuda.memory import BaseMemory, MemoryPointer


def argsort(arr):
    basemem = BaseMemory()
    basemem.ptr = arr.dev.ptr
    basemem.size = arr.dev.nbytes

    memptr = MemoryPointer(basemem, 0)

