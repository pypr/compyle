import numpy as np
from pytools import memoize_method

from .config import get_config
from .types import annotate, dtype_to_knowntype, knowntype_to_ctype
from .template import Template
from .jit import get_ctype_from_arg


try:
    import pycuda
    #if pycuda.VERSION >= (2014, 1):
    if False:
        def cu_bufint(arr, nbytes, offset):
            return arr.gpudata.as_buffer(nbytes, offset)
    else:
        import cffi
        ffi = cffi.FFI()
        def cu_bufint(arr, nbytes, offset):
            return ffi.buffer(
                    ffi.cast('void *', arr.ptr + arr.itemsize * offset),
                    nbytes
                    )
except ImportError as e:
    pass


def get_backend(backend=None):
    if not backend:
        cfg = get_config()
        if cfg.use_opencl:
            return 'opencl'
        elif cfg.use_cuda:
            return 'cuda'
        else:
            return 'cython'
    else:
        return backend


def wrap_array(arr, backend):
    wrapped_array = Array(arr.dtype, allocate=False, backend=backend)
    if isinstance(arr, np.ndarray):
        wrapped_array.data = arr
        if backend == 'opencl' or backend == 'cuda':
            use_double = get_config().use_double
            _dtype = np.float64 if use_double else np.float32
            if np.issubdtype(arr.dtype, np.floating):
                wrapped_array.dtype = _dtype
                wrapped_array.data = arr.astype(_dtype)
            q = None
            if backend == 'opencl':
                from .opencl import get_queue
                from pyopencl.array import to_device
                q = get_queue()
                if arr is not None:
                    dev_ary = to_device(q, wrapped_array.data)
                    wrapped_array.set_data(dev_ary)
            elif backend == 'cuda':
                from .cuda import set_context
                set_context()
                from pycuda.gpuarray import to_gpu
                if arr is not None:
                    dev_ary = to_gpu(wrapped_array.data)
                    wrapped_array.set_data(dev_ary)
        else:
            wrapped_array.set_data(wrapped_array.data)
    elif backend == 'opencl':
        import pyopencl.array as gpuarray
        if isinstance(arr, gpuarray.Array):
            wrapped_array.set_data(arr)
    elif backend == 'cuda':
        import pycuda.gpuarray as gpuarray
        if isinstance(arr, gpuarray.GPUArray):
            wrapped_array.set_data(arr)
    return wrapped_array


def wrap(*args, **kw):
    '''
    Parameters
    ----------

    *args: any numpy arrays to be wrapped.

    **kw: only one keyword arg called `backend` is supported.

    backend: str: use appropriate backend for arrays.
    '''
    backend = get_backend(kw.get('backend'))
    if len(args) == 1:
        return wrap_array(args[0], backend)
    else:
        return [wrap_array(x, backend) for x in args]


def to_device(array, backend='cython'):
    if backend == 'cython':
        out = array
    elif backend == 'opencl':
        import pyopencl.array as gpuarray
        from .opencl import get_queue
        out = gpuarray.to_device(get_queue(), array)
    elif backend == 'cuda':
        import pycuda.gpuarray as gpuarray
        out = gpuarray.to_gpu(array)
    return wrap_array(out, backend)


def ones_like(array, backend=None):
    if backend is None:
        backend = array.backend
    if backend == 'opencl':
        import pyopencl.array as gpuarray
        out = 1 + gpuarray.zeros_like(array.dev)
    elif backend == 'cuda':
        import pycuda.gpuarray as gpuarray
        out = gpuarray.ones_like(array.dev)
    else:
        out = np.ones_like(array.dev)
    return wrap_array(out, backend)


def ones(n, dtype, backend='cython'):
    if backend == 'opencl':
        import pyopencl.array as gpuarray
        from .opencl import get_queue
        out = 1 + gpuarray.zeros(get_queue(), n, dtype)
    elif backend == 'cuda':
        import pycuda.gpuarray as gpuarray
        out = np.array(1, dtype=dtype) + gpuarray.zeros(n, dtype)
    else:
        out = np.ones(n, dtype=dtype)
    return wrap_array(out, backend)


def empty(n, dtype, backend='cython'):
    if backend == 'opencl':
        import pyopencl.array as gpuarray
        from .opencl import get_queue
        out = gpuarray.empty(get_queue(), n, dtype)
    elif backend == 'cuda':
        import pycuda.gpuarray as gpuarray
        out = gpuarray.empty(n, dtype)
    else:
        out = np.empty(n, dtype=dtype)
    return wrap_array(out, backend)


def zeros(n, dtype, backend='cython'):
    if backend == 'opencl':
        import pyopencl.array as gpuarray
        from .opencl import get_queue
        out = gpuarray.zeros(get_queue(), n, dtype)
    elif backend == 'cuda':
        import pycuda.gpuarray as gpuarray
        out = gpuarray.zeros(n, dtype)
    else:
        out = np.zeros(n, dtype=dtype)
    return wrap_array(out, backend)


def zeros_like(array, backend=None):
    if backend is None:
        backend = array.backend
    if backend == 'opencl':
        import pyopencl.array as gpuarray
        out = gpuarray.zeros_like(array.dev)
    elif backend == 'cuda':
        import pycuda.gpuarray as gpuarray
        out = gpuarray.zeros_like(array.dev)
    else:
        out = np.zeros_like(array.dev)
    return wrap_array(out, backend)


def arange(start, stop, step, dtype=None, backend='cython'):
    if backend == 'opencl':
        import pyopencl.array as gpuarray
        from .opencl import get_queue
        out = gpuarray.arange(get_queue(), start, stop,
                              step, dtype=dtype)
    elif backend == 'cuda':
        import pycuda.gpuarray as gpuarray
        out = gpuarray.arange(start, stop, step, dtype=dtype)
    else:
        out = np.arange(start, stop, step, dtype=dtype)
    return wrap_array(out, backend)


def minimum(ary, backend=None):
    if backend is None:
        backend = ary.backend
    if backend == 'cython':
        return ary.dev.min()
    elif backend == 'opencl':
        import pyopencl.array as gpuarray
        return gpuarray.min(ary.dev).get()
    elif backend == 'cuda':
        import pycuda.gpuarray as gpuarray
        return gpuarray.min(ary.dev).get()


def maximum(ary, backend=None):
    if backend is None:
        backend = ary.backend
    if backend == 'cython':
        return ary.dev.max()
    elif backend == 'opencl':
        import pyopencl.array as gpuarray
        return gpuarray.max(ary.dev).get()
    elif backend == 'cuda':
        import pycuda.gpuarray as gpuarray
        return gpuarray.max(ary.dev).get()


def sum(ary, backend=None):
    if backend is None:
        backend = ary.backend
    if backend == 'cython':
        return np.sum(ary.dev)
    if backend == 'opencl':
        import pyopencl.array as gpuarray
        return gpuarray.sum(ary.dev).get()
    if backend == 'cuda':
        import pycuda.gpuarray as gpuarray
        return gpuarray.sum(ary.dev).get()


def dot(a, b, backend=None):
    if backend is None:
        backend = a.backend
    if backend == 'cython':
        return np.dot(a.dev, b.dev)
    if backend == 'opencl':
        import pyopencl.array as gpuarray
        return gpuarray.dot(a.dev, b.dev).get()
    if backend == 'cuda':
        import pycuda.gpuarray as gpuarray
        return gpuarray.dot(a.dev, b.dev).get()


######### Sorting #################

def sort_by_keys(ary_list, out_list=None, key_bits=None,
                 backend=None):
    # first arg of ary_list is the key
    if backend is None:
        backend = ary.backend
    if backend == 'opencl':
        import pyopencl as cl
        import pyopencl.algorithm
        from pyopencl.scan import GenericScanKernel
        from compyle.opencl import get_context, get_queue

        arg_types = [get_ctype_from_arg(arg) for arg in ary_list]

        arg_names = ["ary_%s" % i for i in range(len(ary_list))]

        sort_args = ["%s %s" % (knowntype_to_ctype(ktype), name) \
                for ktype, name in zip(arg_types, arg_names)]

        sort_args = [arg.replace('GLOBAL_MEM', '__global') \
                for arg in sort_args]

        sort_knl = cl.algorithm.RadixSort(
            get_context(),
            sort_args,
            scan_kernel=GenericScanKernel, key_expr="ary_0[i]",
            sort_arg_names=arg_names
        )

        allocator = cl.tools.MemoryPool(
                cl.tools.ImmediateAllocator(get_queue())
                )

        arg_list = [ary.dev for ary in ary_list]

        out_list, event = sort_knl(*arg_list, key_bits=key_bits,
                                   allocator=allocator)
        return out_list
    else:
        order = argsort(ary_list[0], backend=backend)
        out_list = align(ary_list, order, out_list=out_list,
                         backend=backend)
        return out_list


def argsort(ary, backend=None):
    if backend is None:
        backend = ary.backend
    if backend == 'cython':
        result = np.argsort(ary.dev)
        return wrap_array(result, backend=backend)
    elif backend == 'cuda':
        from compyle.cuda import argsort
        return argsort(ary)
    else:
        raise ValueError("Only cython and cuda backends supported")


@annotate
def take_elwise(i, indices, ary, out_ary):
    out_ary[i] = ary[indices[i]]


def take(ary, indices, backend=None, out=None):
    import compyle.parallel as parallel
    if backend is None:
        backend = ary.backend
    if out is None:
        out = empty(indices.length, ary.dtype, backend=backend)
    if backend == 'opencl' or backend == 'cuda':
        take_knl = parallel.Elementwise(take_elwise, backend=backend)
        take_knl(indices, ary, out)
    elif backend == 'cython':
        np.take(ary.dev, indices.dev, out=out.dev)
    return out


@annotate
def inp_cumsum(i, ary):
    return ary[i]


@annotate
def out_cumsum(i, ary, out, item):
    out[i] = item


def cumsum(ary, backend=None, out=None):
    if backend is None:
        backend = ary.backend
    if backend == 'opencl' or backend == 'cuda':
        if out is None:
            out = empty(ary.length, ary.dtype, backend=backend)
        cumsum_scan = Scan(inp_cumsum, out_cumsum, 'a+b',
                           dtype=ary.dtype, backend=backend)
        cumsum_scan(ary=ary, out=out)
        return out
    elif backend == 'cython':
        output = np.cumsum(ary, out=out)
        return wrap_array(output, backend)


class AlignMultiple(Template):
    def __init__(self, name, num_arys):
        super(AlignMultiple, self).__init__(name=name)
        self.num_arys = num_arys

    def extra_args(self):
        args = ['inp_%s' % num for num in range(self.num_arys)]
        args += ['out_%s' % num for num in range(self.num_arys)]
        return args, {}

    def template(self, i, order):
        '''
        % for num in range(obj.num_arys):
        out_${num}[i] = inp_${num}[order[i]]
        % endfor
        '''


def align(ary_list, order, out_list=None, backend=None):
    import compyle.parallel as parallel
    if backend is None:
        backend = order.backend
    if not out_list:
        out_list = []
        for ary in ary_list:
            out_list.append(empty(order.length, ary.dtype,
                            backend=ary.backend))

    args_list = [order] + ary_list + out_list

    align_multiple_knl = AlignMultiple('align_multiple_knl',
                                       len(ary_list))
    align_multiple_elwise = parallel.Elementwise(align_multiple_knl.function,
                                        backend=backend)
    align_multiple_elwise(*args_list)

    return out_list


class Array(object):
    def __init__(self, dtype, n=0, allocate=True, backend=None):
        self.backend = get_backend(backend)
        if backend == 'cuda':
            from .cuda import set_context
            set_context()
        self.dtype = dtype
        self.gptr_type = dtype_to_knowntype(dtype, address='global')
        self.minimum = 0
        self.maximum = 0
        self.data = None
        self._data = None
        self.dev = None
        if allocate:
            length = n
            if n == 0:
                n = 16
            data = empty(n, dtype, backend=self.backend)
            self.set_data(data)
            self.length = length
            self._update_array_ref()

    def __len__(self):
        return len(self.dev)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return wrap_array(self.dev[key], self.backend)
        elif isinstance(key, Array):
            return self.align(key)
        # NOTE: Not sure about this, done for PyCUDA compatibility
        if self.backend is not 'cython':
            return self.dev[key].get()
        else:
            return self.dev[key]

    def __setitem__(self, key, value):
        if self.backend == 'cuda':
            if isinstance(key, slice):
                if isinstance(value, np.ndarray):
                    self.dev[key] = np.asarray(value, dtype=self.dtype)
                else:
                    self.dev[key].fill(value)
            else:
                self.dev[key] = np.asarray(value, dtype=self.dtype)
        else:
            self.dev[key] = value

    def __add__(self, other):
        if isinstance(other, Array):
            other = other.dev
        ans = self.dev + other
        return wrap_array(ans, self.backend)

    def __sub__(self, other):
        if isinstance(other, Array):
            other = other.dev
        ans = self.dev - other
        return wrap_array(ans, self.backend)

    def __radd__(self, other):
        if isinstance(other, Array):
            other = other.dev
        ans = other + self.dev
        return wrap_array(ans, self.backend)

    def __rsub__(self, other):
        if isinstance(other, Array):
            other = other.dev
        ans = other - self.dev
        return wrap_array(ans, self.backend)

    def __str__(self):
        return self.dev.__str__()

    def _update_array_ref(self):
        # For PyCUDA compatibility
        if self.length == 0 and len(self._data) == 0:
            self.dev = self._data
        else:
            self.dev = self._data[:self.length]

    def _get_np_data(self):
        return self.data

    def get_buff(self, offset=0, length=0):
        if not length:
            nbytes = int(self.dev.nbytes - offset * self.dev.itemsize)
        else:
            nbytes = length * self.dev.itemsize
        if self.backend == 'cython':
            length = nbytes // self.dev.itemsize
            return self.dev[offset:offset + length]
        elif self.backend == 'cuda':
            return cu_bufint(self._data, nbytes, int(offset))

    def get(self):
        if self.backend == 'cython':
            return self.dev
        elif self.backend == 'opencl' or self.backend == 'cuda':
            return self.dev.get()

    def get_view(self, offset=0, length=None):
        if length is None:
            length = self.length - offset
        view_arr = Array(self.dtype, allocate=False, backend=self.backend)
        view_arr.set_data(self.dev[offset:offset + length])
        return view_arr

    def set(self, nparr):
        if self.backend == 'cython':
            self.set_data(nparr)
        else:
            self.set_data(to_device(nparr, backend=self.backend))

    def pull(self):
        if self.data is None:
            self.data = np.empty(len(self.dev), dtype=self.dtype)
        self.data[:] = self.get()

    def push(self):
        if self.backend == 'opencl' or self.backend == 'cuda':
            self._data.set(self._get_np_data())
            self.set_data(self._data)

    def resize(self, size):
        self.reserve(size)
        self.length = size
        self._update_array_ref()

    def reserve(self, size):
        if size > self.alloc:
            new_data = empty(size, self.dtype, backend=self.backend)
            # For PyCUDA compatibility
            if self.length > 0:
                new_data.dev[:self.length] = self.dev
            self._data = new_data.dev
            self.alloc = size
            self._update_array_ref()

    def set_data(self, data):
        # data can be an Array instance or
        # a numpy/cl array/cuda array
        if isinstance(data, Array):
            data = data.dev
        self._data = data
        self.length = data.size
        self.alloc = data.size
        self.dtype = data.dtype
        self._update_array_ref()

    def get_array(self):
        return self[:self.length]

    def get_data(self):
        return self._data

    def copy(self):
        arr_copy = Array(self.dtype, backend=self.backend, allocate=False)
        arr_copy.set_data(self.dev.copy())
        return arr_copy

    def update_min_max(self):
        self.minimum = minimum(self, backend=self.backend)
        self.maximum = maximum(self, backend=self.backend)
        self.minimum = self.minimum.astype(self.dtype)
        self.maximum = self.maximum.astype(self.dtype)

    def fill(self, value):
        self.dev.fill(value)

    def append(self, value):
        if self.length >= self.alloc:
            self.reserve(2 * self.length)
        self._data[self.length] = np.asarray(value, dtype=self.dtype)
        self.length += 1
        self._update_array_ref()

    def extend(self, ary):
        if self.length + len(ary.dev) > self.alloc:
            self.reserve(self.length + len(ary.dev))
        self._data[-len(ary.dev):] = ary.dev
        self.length += len(ary.dev)
        self._update_array_ref()

    @memoize_method
    def _get_remove_kernels(self):
        import compyle.parallel as parallel

        @annotate(i='int', gintp='indices, if_remove')
        def fill_if_remove(i, indices, if_remove):
            if_remove[indices[i]] = 1

        fill_if_remove_knl = parallel.Elementwise(
            fill_if_remove, backend=self.backend)

        @annotate(i='int', if_remove='gintp', return_='int')
        def remove_input_expr(i, if_remove):
            return if_remove[i]

        types = {'i': 'int', 'item': 'int',
                 'if_remove': 'gintp',
                 'new_array': self.gptr_type,
                 'old_array': self.gptr_type}

        @annotate(**types)
        def remove_output_expr(i, item, if_remove, new_array, old_array):
            if not if_remove[i]:
                new_array[i - item] = old_array[i]

        remove_knl = parallel.Scan(remove_input_expr, remove_output_expr,
                                   'a+b', dtype=np.int32,
                                   backend=self.backend)

        return fill_if_remove_knl, remove_knl

    def remove(self, indices, input_sorted=False):
        if len(indices) > self.length:
            msg = 'Number of indices to be removed is greater than'
            msg += 'number of indices in array'
            raise ValueError(msg)

        if_remove = Array(np.int32, n=self.length, backend=self.backend)
        if_remove.fill(0)
        new_array = self.copy()

        fill_if_remove_knl, remove_knl = self._get_remove_kernels()

        fill_if_remove_knl(indices, if_remove)

        remove_knl(if_remove=if_remove, old_array=self, new_array=new_array)

        self.set_data(new_array.dev[:-len(indices.dev)])

    def align(self, indices, out=None):
        return take(self, indices, backend=self.backend, out=out)

    def squeeze(self):
        self.set_data(self._data[:self.length])

    def copy_values(self, indices, dest):
        # indices and dest need to be Array instances
        if not isinstance(indices, Array) or \
                not isinstance(dest, Array):
            raise TypeError('indices and dest need to be \
                    Array instances')
        dest.dev[:len(indices.dev)] = take(
            self, indices, backend=self.backend
        ).dev
