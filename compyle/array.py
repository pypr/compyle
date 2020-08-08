import numpy as np
import mako.template as mkt
import time
from pytools import memoize, memoize_method

from .config import get_config
from .types import (annotate, dtype_to_ctype,
                    dtype_to_knowntype, knowntype_to_ctype)
from .template import Template
from .sort import radix_sort
from .profile import profile


try:
    import pycuda
    from .cuda import set_context
    set_context()
    # if pycuda.VERSION >= (2014, 1):
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


minmax_tpl = """

    WITHIN_KERNEL ${dtype} mmc_neutral()
    {
        ${dtype} result;

        % for prop in prop_names:
        % if not only_max:
        result.cur_min_${prop} = ${inf};
        % endif
        % if not only_min:
        result.cur_max_${prop} = -${inf};
        % endif
        % endfor

        return result;
    }

    WITHIN_KERNEL ${dtype} mmc_from_scalar(${args})
    {
        ${dtype} result;

        % for prop in prop_names:
        % if not only_max:
        result.cur_min_${prop} = ${prop};
        % endif
        % if not only_min:
        result.cur_max_${prop} = ${prop};
        % endif
        % endfor

        return result;
    }

    WITHIN_KERNEL ${dtype} agg_mmc(${dtype} a, ${dtype} b)
    {
        ${dtype} result = a;

        % for prop in prop_names:
        % if not only_max:
        if (b.cur_min_${prop} < result.cur_min_${prop})
            result.cur_min_${prop} = b.cur_min_${prop};
        % endif
        % if not only_min:
        if (b.cur_max_${prop} > result.cur_max_${prop})
            result.cur_max_${prop} = b.cur_max_${prop};
        % endif
        % endfor

        return result;
    }

    """


minmax_operator_tpl = """

    __device__ ${dtype} volatile &operator=(
        ${dtype} const &src) volatile
    {
        % for prop in prop_names:
        % if not only_max:
        this->cur_min_${prop} = src.cur_min_${prop};
        % endif
        % if not only_min:
        this->cur_max_${prop} = src.cur_max_${prop};
        % endif
        % endfor
        return *this;
    }
"""


def minmax_collector_key(device, dtype, props, name, *args):
    return (device, dtype, tuple(props), name)


@memoize(key=minmax_collector_key)
def make_collector_dtype(device, dtype, props, name,
                         only_min, only_max, backend):
    fields = [("pad", np.int32)]

    for prop in props:
        if not only_min:
            fields.append(("cur_max_%s" % prop, dtype))
        if not only_max:
            fields.append(("cur_min_%s" % prop, dtype))

    custom_dtype = np.dtype(fields)

    if backend == 'opencl':
        from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct
    elif backend == 'cuda':
        from compyle.cuda import match_dtype_to_c_struct
        from pycuda.tools import get_or_register_dtype

    custom_dtype, c_decl = match_dtype_to_c_struct(device, name, custom_dtype)
    custom_dtype = get_or_register_dtype(name, custom_dtype)

    return custom_dtype, c_decl


@memoize(key=lambda *args: (args[-3], args[-2], args[-1]))
def get_minmax_kernel(ctx, dtype, inf, mmc_dtype, prop_names,
                      only_min, only_max, name, mmc_c_decl, backend):
    tpl_args = ", ".join(
        ["%(dtype)s %(prop)s" % {'dtype': dtype, 'prop': prop}
            for prop in prop_names]
    )

    if backend == 'cuda':
        # overload assignment operator in struct
        mmc_overload = mkt.Template(text=minmax_operator_tpl).render(
            prop_names=prop_names, dtype=name,
            only_min=only_min, only_max=only_max
        )
        mmc_c_decl_lines = mmc_c_decl.splitlines()
        mmc_c_decl_lines = mmc_c_decl_lines[:-2] + \
            mmc_overload.splitlines() + mmc_c_decl_lines[-2:]
        mmc_c_decl = '\n'.join(mmc_c_decl_lines)

    mmc_preamble = mmc_c_decl + minmax_tpl
    preamble = mkt.Template(text=mmc_preamble).render(
        args=tpl_args, prop_names=prop_names, dtype=name,
        only_min=only_min, only_max=only_max, inf=inf
    )

    map_args = ", ".join(
        ["%(prop)s[i]" % {'dtype': dtype, 'prop': prop}
            for prop in prop_names]
    )

    if backend == 'opencl':
        knl_args = ", ".join(
            ["__global %(dtype)s* %(prop)s" % {'dtype': dtype, 'prop': prop}
                for prop in prop_names]
        )

        from pyopencl._cluda import CLUDA_PREAMBLE
        from pyopencl.reduction import ReductionKernel

        cluda_preamble = mkt.Template(text=CLUDA_PREAMBLE).render(
            double_support=True
        )

        knl = ReductionKernel(
            ctx, mmc_dtype, neutral="mmc_neutral()",
            reduce_expr="agg_mmc(a, b)",
            map_expr="mmc_from_scalar(%s)" % map_args,
            arguments=knl_args,
            preamble='\n'.join([cluda_preamble, preamble])
        )

    elif backend == 'cuda':
        knl_args = ", ".join(
            ["%(dtype)s* %(prop)s" % {'dtype': dtype, 'prop': prop}
                for prop in prop_names]
        )

        from pycuda._cluda import CLUDA_PREAMBLE
        from pycuda.reduction import ReductionKernel

        cluda_preamble = mkt.Template(text=CLUDA_PREAMBLE).render(
            double_support=True
        )

        knl = ReductionKernel(
            mmc_dtype, neutral="mmc_neutral()",
            reduce_expr="agg_mmc(a, b)",
            map_expr="mmc_from_scalar(%s)" % map_args,
            arguments=knl_args,
            preamble='\n'.join([cluda_preamble, preamble])
        )

    return knl


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


def arange(start, stop, step, dtype=np.int32, backend='cython'):
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


@memoize(key=lambda *args: tuple(args[0]))
def get_cl_sort_kernel(arg_types, ary_list):
    import pyopencl as cl
    from pyopencl.scan import GenericScanKernel
    import pyopencl.algorithm
    from compyle.opencl import get_context, get_queue
    arg_names = ["ary_%s" % i for i in range(len(ary_list))]

    sort_args = ["%s %s" % (knowntype_to_ctype(ktype), name)
                 for ktype, name in zip(arg_types, arg_names)]

    sort_args = [arg.replace('GLOBAL_MEM', '__global')
                 for arg in sort_args]

    sort_knl = cl.algorithm.RadixSort(
        get_context(),
        sort_args,
        scan_kernel=GenericScanKernel, key_expr="ary_0[i]",
        sort_arg_names=arg_names
    )

    return sort_knl


@memoize(key=lambda q: q)
def get_allocator(queue):
    import pyopencl as cl

    allocator = cl.tools.MemoryPool(
        cl.tools.ImmediateAllocator(queue)
    )

    return allocator


@profile
def sort_by_keys(ary_list, out_list=None, key_bits=None,
                 backend=None, use_radix_sort=False):
    # FIXME: Need to use returned values, cuda backend uses
    # thrust that will internally allocate a new array for storing
    # the sorted data so out_list will not have the sorted arrays
    # first arg of ary_list is the key
    if backend is None:
        backend = ary_list[0].backend
    if backend == 'opencl':
        from .jit import get_ctype_from_arg
        from compyle.opencl import get_queue

        if not out_list:
            out_list = [
                Array(ary.dtype, allocate=False, backend=backend)
                for ary in ary_list
            ]

        arg_types = [get_ctype_from_arg(arg) for arg in ary_list]

        sort_knl = get_cl_sort_kernel(arg_types, ary_list)
        allocator = get_allocator(get_queue())

        arg_list = [ary.dev for ary in ary_list]

        out_arrays, event = sort_knl(*arg_list, key_bits=key_bits,
                                     allocator=allocator)
        for i, out in enumerate(out_list):
            out.set_data(out_arrays[i])
        return out_list
    elif backend == 'cython' and use_radix_sort:
        out_list, order = radix_sort(ary_list, out_list=out_list,
                                     max_key_bits=key_bits, backend=backend)
        return out_list
    elif backend == 'cython':
        order = wrap(np.argsort(ary_list[0].dev), backend=backend)
        out_list = align(ary_list, order, out_list=out_list,
                         backend=backend)
        return out_list
    else:
        order = argsort(ary_list[0], backend=backend)
        modified_out_list = None
        if out_list:
            modified_out_list = out_list[1:]
        out_list = align(ary_list[1:], order, out_list=modified_out_list,
                         backend=backend)
        return [ary_list[0]] + out_list


def argsort(ary, backend=None):
    # FIXME: Implement an OpenCL backend and add tests
    # NOTE: argsort also sorts the array
    if backend is None:
        backend = ary.backend
    if backend == 'cython':
        result = np.argsort(ary.dev)
        ary.dev = np.take(ary.dev, result)
        return wrap_array(result, backend=backend)
    elif backend == 'cuda':
        from compyle.cuda import argsort
        return argsort(ary)
    else:
        raise ValueError("Only cython and cuda backends supported")


def update_minmax_gpu(ary_list, only_min=False, only_max=False,
                      backend=None):
    if not backend:
        backend = ary_list[0].backend

    if only_min and only_max:
        raise ValueError("Only one of only_min and only_max can be True")

    props = ['ary_%s' % i for i in range(len(ary_list))]

    dtype = ary_list[0].dtype
    ctype = dtype_to_ctype(dtype)

    op = 'min' if not only_max else ''
    op += 'max' if not only_min else ''
    name = "%s_collector_%s" % (op, ''.join([ctype] + props))

    if backend == 'opencl':
        from compyle.opencl import get_context
        ctx = get_context()
        device = ctx.devices[0]
    elif backend == 'cuda':
        ctx = None
        device = None

    mmc_dtype, mmc_c_decl = make_collector_dtype(device,
                                                 dtype, props, name,
                                                 only_min, only_max,
                                                 backend)

    if np.issubdtype(dtype, np.floating):
        inf = np.finfo(dtype).max
    else:
        inf = np.iinfo(dtype).max

    knl = get_minmax_kernel(ctx, ctype, inf, mmc_dtype, props,
                            only_min, only_max, name, mmc_c_decl,
                            backend)

    args = [ary.dev for ary in ary_list]

    result = knl(*args).get()

    for ary, prop in zip(ary_list, props):
        if not only_max:
            ary.minimum = result["cur_min_%s" % prop]
        if not only_min:
            ary.maximum = result["cur_max_%s" % prop]


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


@profile
def cumsum(ary, backend=None, out=None):
    if backend is None:
        backend = ary.backend
    if backend == 'opencl' or backend == 'cuda':
        import compyle.parallel as parallel
        if out is None:
            out = empty(ary.length, ary.dtype, backend=backend)
        cumsum_scan = parallel.Scan(
            inp_cumsum, out_cumsum, 'a+b', dtype=ary.dtype, backend=backend
        )
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


def key_align_kernel(ary_list, order, backend=None):
    from .jit import get_ctype_from_arg
    key = [get_ctype_from_arg(ary) for ary in ary_list]
    key.append(backend)
    key.append(get_config().use_openmp)
    return tuple(key)


@memoize(key=key_align_kernel)
def get_align_kernel(ary_list, order, backend=None):
    import compyle.parallel as parallel
    align_multiple_knl = AlignMultiple('align_multiple_knl',
                                       len(ary_list))
    align_multiple_elwise = parallel.Elementwise(align_multiple_knl.function,
                                                 backend=backend)
    return align_multiple_elwise


def align(ary_list, order, out_list=None, backend=None):
    if not ary_list:
        return []

    if backend is None:
        backend = order.backend
    if not out_list:
        out_list = []
        for ary in ary_list:
            out_list.append(empty(order.length, ary.dtype,
                                  backend=ary.backend))

    args_list = [order] + ary_list + out_list

    align_multiple_elwise = get_align_kernel(ary_list, order, backend=backend)

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
        if self.backend != 'cython':
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

    @profile
    def update_min_max(self, only_min=False, only_max=False):
        if self.backend == 'cython':
            self.minimum = minimum(self, backend=self.backend)
            self.maximum = maximum(self, backend=self.backend)
            self.minimum = self.minimum.astype(self.dtype)
            self.maximum = self.maximum.astype(self.dtype)
        else:
            update_minmax_gpu([self])

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

    @profile
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
