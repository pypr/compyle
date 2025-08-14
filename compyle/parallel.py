"""A set of parallel algorithms that allow users to solve a variety of
problems. These functions are heavily inspired by the same functionality
provided in pyopencl. However, we also provide Cython implementations for these
and unify the syntax in a transparent way which allows users to write the code
once and have it run on different execution backends.

"""

from compyle import c_backend
from functools import wraps
from textwrap import wrap

from mako.template import Template
import numpy as np
import pybind11

from .cimport import Cmodule
from .config import get_config
from .profile import profile
from .cython_generator import get_parallel_range, CythonGenerator
from .transpiler import Transpiler, convert_to_float_if_needed
from .types import TYPES, annotate, dtype_to_ctype
from .ext_module import get_md5

from . import array


elementwise_cy_template = '''
from cython.parallel import parallel, prange

cdef c_${name}(${c_arg_sig}):
    cdef int i
%if openmp:
    with nogil, parallel():
        for i in ${get_parallel_range("SIZE")}:
            ${name}(${c_args})
%else:
    for i in range(SIZE):
        ${name}(${c_args})
%endif

cpdef py_${name}(${py_arg_sig}):
    c_${name}(${py_args})
'''

reduction_cy_template = '''
from cython.parallel import parallel, prange, threadid
from libc.stdlib cimport abort, malloc, free
from libc.math cimport INFINITY
cimport openmp

cdef double INFTY = float('inf')

cpdef int get_number_of_threads():
% if openmp:
    cdef int i, n
    with nogil, parallel():
        for i in prange(1):
            n = openmp.omp_get_num_threads()
    return n
% else:
    return 1
% endif

cdef int gcd(int a, int b):
    while b != 0:
        a, b = b, a%b
    return a

cdef int get_stride(int sz, int itemsize):
    return sz//gcd(sz, itemsize)


cdef ${type} c_${name}(${c_arg_sig}):
    cdef int i, n_thread, tid, scan_stride, sz
    cdef ${type} a, b
    n_thread = get_number_of_threads()
    sz = sizeof(${type})

    # This striding is to do 64 bit alignment to prevent false sharing.
    scan_stride = get_stride(64, sz)
    cdef ${type}* buffer
    buffer = <${type}*>malloc(n_thread*scan_stride*sz)
    if buffer == NULL:
        raise MemoryError("Unable to allocate memory for reduction")

%if openmp:
    with nogil, parallel():
% else:
    if 1:
% endif
        tid = threadid()
        buffer[tid*scan_stride] = ${neutral}
%if openmp:
        for i in ${get_parallel_range("SIZE")}:
%else:
        for i in range(SIZE):
%endif
            a = buffer[tid*scan_stride]
            b = ${map_expr}
            buffer[tid*scan_stride] = ${reduce_expr}

    a = ${neutral}
    for i in range(n_thread):
        b = buffer[i*scan_stride]
        a = ${reduce_expr}

    free(buffer)
    return a


cpdef py_${name}(${py_arg_sig}):
    return c_${name}(${py_args})
'''

scan_cy_template = '''
from cython.parallel import parallel, prange, threadid
from libc.stdlib cimport abort, malloc, free
cimport openmp
cimport numpy as np
cpdef int get_number_of_threads():
    % if openmp:
        cdef int i, n
        with nogil, parallel():
            for i in prange(1):
                n = openmp.omp_get_num_threads()
        return n
    % else:
        return 1
    % endif

cdef int gcd(int a, int b):
    while b != 0:
        a, b = b, a % b
    return a

cdef int get_stride(int sz, int itemsize):
    return sz // gcd(sz, itemsize)


cdef void c_${name}(${c_arg_sig}):
    cdef int i, n_thread, tid, scan_stride, sz, N

    N = SIZE
    n_thread = get_number_of_threads()
    sz = sizeof(${type})

    # This striding is to do 64 bit alignment to prevent false sharing.
    scan_stride = get_stride(64, sz)

    cdef ${type}* buffer
    buffer = <${type}*> malloc(n_thread * scan_stride * sz)

    if buffer == NULL:
        raise MemoryError("Unable to allocate memory for scan.")

    % if use_segment:
    cdef int* scan_seg_flags
    cdef int* chunk_new_segment
    scan_seg_flags = <int*> malloc(SIZE * sizeof(int))
    chunk_new_segment = <int*> malloc(n_thread * scan_stride * sizeof(int))

    if scan_seg_flags == NULL or chunk_new_segment == NULL:
        raise MemoryError("Unable to allocate memory for segmented scan")
    % endif

    % if complex_map:
    cdef ${type}* map_output
    map_output = <${type}*> malloc(SIZE * sz)

    if map_output == NULL:
        raise MemoryError("Unable to allocate memory for scan. (Recommended:
        Set complex_map=False.)")
    % endif


    cdef int buffer_idx, start, end, has_segment
    cdef ${type} a, b, temp
    # This chunksize would divide input data equally
    # between threads
    % if not calc_last_item:
    # A chunk of 1 MB per thread
    cdef int chunksize = 1048576 // sz
    % else:
    # Process all data together. Only then can we get
    # the last item immediately
    cdef int chunksize = (SIZE + n_thread - 1) // n_thread
    % endif

    cdef int offset = 0
    cdef ${type} global_carry = ${neutral}
    cdef ${type} last_item
    cdef ${type} carry, item, prev_item

    while offset < SIZE:
        # Pass 1
        with nogil, parallel():
            tid = threadid()
            buffer_idx = tid * scan_stride

            start = offset + tid * chunksize
            end = min(offset + (tid + 1) * chunksize, SIZE)
            has_segment = 0

            temp = ${neutral}
            for i in range(start, end):

                % if use_segment:
                # Generate segment flags
                scan_seg_flags[i] = ${is_segment_start_expr}
                if (scan_seg_flags[i]):
                    has_segment = 1
                % endif

                # Carry
                % if use_segment:
                if (scan_seg_flags[i]):
                    a = ${neutral}
                else:
                    a = temp
                % else:
                a = temp
                % endif

                # Map
                b = ${input_expr}
                % if complex_map:
                map_output[i] = b
                % endif

                # Scan
                temp = ${scan_expr}

            buffer[buffer_idx] = temp
            % if use_segment:
            chunk_new_segment[buffer_idx] = has_segment
            % endif

        # Pass 2: Aggregate chunks
        # Add previous carry to buffer[0]
        % if use_segment:
        if chunk_new_segment[0]:
            a = ${neutral}
        else:
            a = global_carry
        b = buffer[0]
        % else:
        a = global_carry
        b = buffer[0]
        % endif
        buffer[0] = ${scan_expr}

        for i in range(n_thread - 1):
            % if use_segment:

            # With segmented scan
            if chunk_new_segment[(i + 1) * scan_stride]:
                a = ${neutral}
            else:
                a = buffer[i * scan_stride]
            b = buffer[(i + 1) * scan_stride]
            buffer[(i + 1) * scan_stride] = ${scan_expr}

            % else:

            # Without segmented scan
            a = buffer[i * scan_stride]
            b = buffer[(i + 1) * scan_stride]
            buffer[(i + 1) * scan_stride] = ${scan_expr}

            % endif

        last_item = buffer[(n_thread - 1) * scan_stride]

        # Shift buffer to right by 1 unit
        for i in range(n_thread - 1, 0, -1):
            buffer[i * scan_stride] = buffer[(i - 1) * scan_stride]

        buffer[0] = global_carry
        global_carry = last_item

        # Pass 3: Output
        with nogil, parallel():
            tid = threadid()
            buffer_idx = tid * scan_stride
            carry = buffer[buffer_idx]

            start = offset + tid * chunksize
            end = min(offset + (tid + 1) * chunksize, SIZE)

            for i in range(start, end):
                # Output
                % if use_segment:
                if scan_seg_flags[i]:
                    a = ${neutral}
                else:
                    a = carry
                % else:
                a = carry
                % endif

                % if complex_map:
                b = map_output[i]
                % else:
                b = ${input_expr}
                % endif

                % if calc_prev_item:
                prev_item = carry
                % endif

                carry = ${scan_expr}
                item = carry

                ${output_expr}
        offset += chunksize * n_thread

    # Clean up
    free(buffer)

    % if use_segment:
    free(scan_seg_flags)
    free(chunk_new_segment)
    % endif

    % if complex_map:
    free(map_output)
    % endif

cpdef py_${name}(${py_arg_sig}):
    return c_${name}(${py_args})
'''

scan_cy_single_thread_template = '''
from cython.parallel import parallel, prange, threadid
from libc.stdlib cimport abort, malloc, free
cimport openmp
cimport numpy as np

cdef void c_${name}(${c_arg_sig}):
    cdef int i, N, across_seg_boundary
    cdef ${type} a, b, item
    N = SIZE

    % if calc_last_item:
    a = ${neutral}
    for i in range(N):
        b = ${input_expr}
        a = ${scan_expr}
    last_item = a
    % endif

    a = ${neutral}

    for i in range(N):
        # Segment operation
        % if use_segment:
        across_seg_boundary = ${is_segment_start_expr}
        if across_seg_boundary:
            a = ${neutral}
        % endif

        # Map
        b = ${input_expr}

        % if calc_prev_item:
        prev_item = a
        % endif

        # Scan
        a = ${scan_expr}
        item = a

        # Output
        ${output_expr}

cpdef py_${name}(${py_arg_sig}):
    return c_${name}(${py_args})
'''


def drop_duplicates(arr):
    result = []
    for x in arr:
        if x not in result:
            result.extend([x])
    return result


def serial(func=None, **kw):
    """Decorator to specify serial execution of a cython
    function
    """
    def wrapper(func):
        func.is_serial = True
        return func

    if func is None:
        return wrapper
    else:
        return wrapper(func)


def get_common_cache_key(obj):
    return obj.backend, obj._config.use_openmp, obj._config.use_double


class ElementwiseBase(object):
    def __init__(self, func, backend=None):
        backend = array.get_backend(backend)
        self.tp = Transpiler(backend=backend)
        self.backend = backend
        self.name = 'elwise_%s' % func.__name__
        self.func = func
        self._config = get_config()
        self.cython_gen = CythonGenerator()
        self.queue = None
        # This is the source generated for the user code.
        self.source = '# Source not yet generated.'
        # This is all the source code used for the elementwise.
        self.all_source = '# Source not yet generated.'
        self.c_func = self._generate()

    def _generate(self, declarations=None):
        self.tp.add(self.func, declarations=declarations)
        if self.backend == 'cython':
            # FIXME: Handle the name of the kernel correctly
            py_data, c_data = self.cython_gen.get_func_signature(self.func)
            py_defn = ['long SIZE'] + py_data[0][1:]
            c_defn = ['long SIZE'] + c_data[0][1:]
            py_args = ['SIZE'] + py_data[1][1:]
            template = Template(text=elementwise_cy_template)
            src = template.render(
                name=self.name[7:],
                c_arg_sig=', '.join(c_defn),
                c_args=', '.join(c_data[1]),
                py_arg_sig=', '.join(py_defn),
                py_args=', '.join(py_args),
                openmp=self._config.use_openmp and not getattr(
                    self.func, 'is_serial', False),
                get_parallel_range=get_parallel_range
            )
            # This is the user code source.
            self.source = self.tp.get_code()
            self.tp.add_code(src)
            self.tp.compile()
            # All the source code for the elementwise
            self.all_source = self.tp.source
            return getattr(self.tp.mod, 'py_' + self.name[7:])
        elif self.backend == 'opencl':
            py_data, c_data = self.cython_gen.get_func_signature(self.func)
            self._correct_opencl_address_space(c_data)

            from .opencl import get_context, get_queue
            from pyopencl.elementwise import ElementwiseKernel
            from pyopencl._cluda import CLUDA_PREAMBLE
            ctx = get_context()
            self.queue = get_queue()
            name = self.func.__name__
            expr = '{func}({args})'.format(
                func=name,
                args=', '.join(c_data[1])
            )
            arguments = convert_to_float_if_needed(', '.join(c_data[0][1:]))
            preamble = convert_to_float_if_needed(self.tp.get_code())
            cluda_preamble = Template(text=CLUDA_PREAMBLE).render(
                double_support=True
            )
            knl = ElementwiseKernel(
                ctx,
                name=self.name,
                arguments=arguments,
                operation=expr,
                preamble="\n".join([cluda_preamble, preamble])
            )
            # only code we generate is saved here.
            self.source = "\n".join([cluda_preamble, preamble])
            all_source = knl.get_kernel(False)[0].program.source
            self.all_source = all_source or self.source
            return knl
        elif self.backend == 'cuda':
            py_data, c_data = self.cython_gen.get_func_signature(self.func)
            self._correct_opencl_address_space(c_data)

            from .cuda import set_context
            set_context()
            from pycuda.elementwise import ElementwiseKernel
            from pycuda._cluda import CLUDA_PREAMBLE
            name = self.func.__name__
            expr = '{func}({args})'.format(
                func=name,
                args=', '.join(c_data[1])
            )
            arguments = convert_to_float_if_needed(', '.join(c_data[0][1:]))
            preamble = convert_to_float_if_needed(self.tp.get_code())
            cluda_preamble = Template(text=CLUDA_PREAMBLE).render(
                double_support=True
            )
            knl = ElementwiseKernel(
                name=self.name,
                arguments=arguments,
                operation=expr,
                preamble="\n".join([cluda_preamble, preamble])
            )
            # only code we generate is saved here.
            self.source = cluda_preamble + preamble
            # FIXME: it is difficult to get the sources from pycuda.
            self.all_source = self.source
            return knl
        elif self.backend == 'c':
            self.pyb11_backend = c_backend.CBackend()
            py_data, c_data = self.pyb11_backend.get_func_signature_pyb11(
                self.func)
            pyb11_args = ', '.join(py_data[0][1:])
            size = '{arg}.request().size'.format(arg=c_data[1][1])
            pyb11_call = ', '.join([size] + py_data[1][1:])
            c_defn = ['size_t SIZE'] + c_data[0][1:]
            arguments = ', '.join(c_defn)
            name = self.func.__name__
            expr = '{func}({args})'.format(
                func=name,
                args=', '.join(c_data[1])
            )

            openmp = self._config.use_openmp

            templete_elwise = Template(c_backend.elwise_c_template)
            src_elwise = templete_elwise.render(
                name=self.name,
                arguments=arguments,
                openmp=openmp,
                operations=expr
            )

            self.source = self.tp.get_code()
            if openmp:
                self.source = '#include <omp.h>\n' + self.source
            self.all_source = self.source + '\n' + src_elwise
            hash_fn = get_md5(self.all_source)
            modname = f'm_{hash_fn}'

            template = Template(c_backend.elwise_c_pybind)
            src_bind = template.render(
                name=self.name,
                modname=modname,
                pyb11_args=pyb11_args,
                pyb11_call=pyb11_call
            )

            self.all_source += src_bind

            mod = Cmodule(self.all_source, hash_fn, openmp=openmp,
                          extra_inc_dir=[pybind11.get_include()])
            module = mod.load()
            return getattr(module, modname)

    def _correct_opencl_address_space(self, c_data):
        code = self.tp.blocks[-1].code.splitlines()
        header_idx = 1
        for line in code:
            if line.rstrip().endswith(')'):
                break
            header_idx += 1

        def _add_address_space(arg):
            if '*' in arg and 'GLOBAL_MEM' not in arg:
                return 'GLOBAL_MEM ' + arg
            else:
                return arg

        args = [_add_address_space(arg) for arg in c_data[0]]
        code[:header_idx] = wrap(
            'WITHIN_KERNEL void {func}({args})'.format(
                func=self.func.__name__,
                args=', '.join(args)
            ),
            width=78, subsequent_indent=' ' * 4, break_long_words=False
        )
        self.tp.blocks[-1].code = '\n'.join(code)

    def _massage_arg(self, x):
        if isinstance(x, array.Array):
            return x.dev
        elif self.backend != 'cuda' or isinstance(x, np.ndarray):
            return x
        else:
            return np.asarray(x)

    @profile
    def __call__(self, *args, **kw):
        c_args = [self._massage_arg(x) for x in args]
        if self.backend == 'cython':
            size = len(c_args[0])
            c_args.insert(0, size)
            self.c_func(*c_args, **kw)
        elif self.backend == 'opencl':
            self.c_func(*c_args, **kw)
            self.queue.finish()
        elif self.backend == 'cuda':
            import pycuda.driver as drv
            event = drv.Event()
            self.c_func(*c_args, **kw)
            event.record()
            event.synchronize()
        elif self.backend == 'c':
            self.c_func(*c_args)


class Elementwise(object):
    def __init__(self, func, backend=None):
        self._func = func
        self._backend = backend
        self.elementwise = None

    def _setup(self):
        func = self._func
        backend = array.get_backend(self._backend)
        if getattr(func, '__annotations__',
                   None) and not hasattr(func, 'is_jit'):
            self.elementwise = ElementwiseBase(func, backend=backend)
        else:
            from .jit import ElementwiseJIT
            self.elementwise = ElementwiseJIT(func, backend=backend)

    def __getattr__(self, name):
        return getattr(self.elementwise, name)

    def __dir__(self):
        return sorted(dir(self.elementwise) + ['elementwise'])

    def __call__(self, *args, **kwargs):
        if self.elementwise is None:
            self._setup()
        self.elementwise(*args, **kwargs)

    def set_backend(self, backend=None):
        self._backend = backend
        self.elementwise = None


def elementwise(func=None, backend=None):
    def _wrapper(function):
        return wraps(function)(Elementwise(function, backend=backend))

    if func is None:
        return _wrapper
    else:
        return _wrapper(func)


class ReductionBase(object):
    def __init__(self, reduce_expr, map_func=None, dtype_out=np.float64,
                 neutral='0', backend=None):
        backend = array.get_backend(backend)
        self.tp = Transpiler(backend=backend)
        self.backend = backend
        self.func = map_func
        if map_func is not None:
            self.name = 'reduce_' + map_func.__name__
        else:
            self.name = 'reduce'
        self.reduce_expr = reduce_expr
        self.dtype_out = dtype_out
        self.type = dtype_to_ctype(dtype_out, backend=backend)
        if backend == 'cython':
            # On Windows, INFINITY is not defined so we use INFTY which we
            # internally define.
            self.neutral = neutral.replace('INFINITY', 'INFTY')
        else:
            self.neutral = neutral
        self._config = get_config()
        self.cython_gen = CythonGenerator()
        self.queue = None
        # This is the source generated for the user code.
        self.source = '# Source not yet generated.'
        # This is all the source code used.
        self.all_source = '# Source not yet generated.'
        self.c_func = self._generate()

    def _generate(self, declarations=None):
        if self.backend == 'cython':
            if self.func is not None:
                self.tp.add(self.func, declarations=declarations)
                py_data, c_data = self.cython_gen.get_func_signature(self.func)
                self._correct_return_type(c_data)
                name = self.func.__name__
                cargs = ', '.join(c_data[1])
                map_expr = '{name}({cargs})'.format(name=name, cargs=cargs)
            else:
                py_data = (['int i', '{type}[:] inp'.format(type=self.type)],
                           ['i', '&inp[0]'])
                c_data = (['int i', '{type}* inp'.format(type=self.type)],
                          ['i', 'inp'])
                map_expr = 'inp[i]'
            py_defn = ['long SIZE'] + py_data[0][1:]
            c_defn = ['long SIZE'] + c_data[0][1:]
            py_args = ['SIZE'] + py_data[1][1:]
            template = Template(text=reduction_cy_template)
            src = template.render(
                name=self.name,
                type=self.type,
                map_expr=map_expr,
                reduce_expr=self.reduce_expr,
                neutral=self.neutral,
                c_arg_sig=', '.join(c_defn),
                py_arg_sig=', '.join(py_defn),
                py_args=', '.join(py_args),
                openmp=self._config.use_openmp,
                get_parallel_range=get_parallel_range
            )
            # This is the user code source.
            self.source = self.tp.get_code()
            self.tp.add_code(src)
            self.tp.compile()
            self.all_source = self.tp.source
            return getattr(self.tp.mod, 'py_' + self.name)
        elif self.backend == 'c':
            self.pyb11_backend = c_backend.CBackend()
            if self.func is not None:
                self.func.__annotations__['return'] = TYPES[self.type]
                self.tp.add(self.func, declarations=declarations)
                pyb_data, c_data = self.pyb11_backend.get_func_signature_pyb11(
                    self.func)
                c_call = c_data[1]

                c_call_default = ['N', 'neutral']
                predefined_vars = ['i'] + c_call_default
                c_args_extra = [[], []]
                pyb_args_extra = [[], []]
                for i, var in enumerate(c_call[1:]):
                    if var not in predefined_vars:
                        c_args_extra[0].append(c_data[0][i + 1])
                        c_args_extra[1].append(var)
                        pyb_args_extra[0].append(pyb_data[0][i + 1])
                        pyb_args_extra[1].append(pyb_data[1][i + 1])
                c_args_extra_str = ", " + ', '.join(c_args_extra[0])
                c_call_extra_str = ", " + ', '.join(c_args_extra[1])
                pyb_args_extra_str = ", " + ', '.join(pyb_args_extra[0])
                pyb_call_extra_str = ", " + ', '.join(pyb_args_extra[1])
                map_expr = f"{self.func.__name__}({', '.join(c_call)})"
            else:
                c_args_extra_str = f", {self.type + '*'} in"
                c_call_extra_str = ", in"
                arg_typ = self.pyb11_backend.ctype_to_pyb11(self.type + '*')
                pyb_args_extra_str = f", {arg_typ} in"
                pyb_call_extra_str = f", ({self.type}*) in.request().ptr"
                map_expr = "in[i]"
            self.source = self.tp.get_code()
            openmp = self._config.use_openmp
            if openmp:
                self.source = '#include <omp.h>\n' + self.source

            template_red = Template(c_backend.reduction_c_template)
            src_red = template_red.render(
                args_extra=c_args_extra_str,
                call_extra=c_call_extra_str,
                map_expr=map_expr,
                red_expr=self.reduce_expr,
                name=self.name,
                type=self.type,
                openmp=openmp
            )
            self.all_source = self.source + src_red
            hash_fn = get_md5(self.all_source)
            modname = f'm_{hash_fn}'

            template_pybind = Template(c_backend.reduction_c_pybind)
            src_pybind = template_pybind.render(
                name=modname,
                type=self.type,
                pyb_args=pyb_args_extra_str,
                pyb_call=pyb_call_extra_str,
                neutral=self.neutral,
            )
            self.all_source += src_pybind

            mod = Cmodule(self.all_source, hash_fn, openmp=openmp,
                          extra_inc_dir=[pybind11.get_include()])
            module = mod.load()
            return getattr(module, modname)

        elif self.backend == 'opencl':
            if self.func is not None:
                self.tp.add(self.func, declarations=declarations)
                py_data, c_data = self.cython_gen.get_func_signature(self.func)
                self._correct_opencl_address_space(c_data)
                name = self.func.__name__
                expr = '{func}({args})'.format(
                    func=name,
                    args=', '.join(c_data[1])
                )
                arguments = convert_to_float_if_needed(
                    ', '.join(c_data[0][1:])
                )
                preamble = convert_to_float_if_needed(self.tp.get_code())
            else:
                arguments = '{type} *in'.format(type=self.type)
                expr = None
                preamble = ''

            from .opencl import get_context, get_queue
            from pyopencl.reduction import ReductionKernel
            from pyopencl._cluda import CLUDA_PREAMBLE
            cluda_preamble = Template(text=CLUDA_PREAMBLE).render(
                double_support=True
            )

            ctx = get_context()
            self.queue = get_queue()
            knl = ReductionKernel(
                ctx,
                dtype_out=self.dtype_out,
                neutral=self.neutral,
                reduce_expr=self.reduce_expr,
                map_expr=expr,
                arguments=arguments,
                preamble="\n".join([cluda_preamble, preamble])
            )
            # only code we generate is saved here.
            self.source = "\n".join([cluda_preamble, preamble])
            if knl.stage_1_inf.source:
                self.all_source = "\n".join([
                    "// ------ stage 1 -----",
                    knl.stage_1_inf.source,
                    "// ------ stage 2 -----",
                    knl.stage_2_inf.source,
                ])
            else:
                self.all_source = self.source
            return knl
        elif self.backend == 'cuda':
            if self.func is not None:
                self.tp.add(self.func, declarations=declarations)
                py_data, c_data = self.cython_gen.get_func_signature(self.func)
                self._correct_opencl_address_space(c_data)
                name = self.func.__name__
                expr = '{func}({args})'.format(
                    func=name,
                    args=', '.join(c_data[1])
                )
                arguments = convert_to_float_if_needed(
                    ', '.join(c_data[0][1:])
                )
                preamble = convert_to_float_if_needed(self.tp.get_code())
            else:
                arguments = '{type} *in'.format(type=self.type)
                expr = None
                preamble = ''

            from .cuda import set_context
            set_context()
            from pycuda.reduction import ReductionKernel
            from pycuda._cluda import CLUDA_PREAMBLE
            cluda_preamble = Template(text=CLUDA_PREAMBLE).render(
                double_support=True
            )

            knl = ReductionKernel(
                dtype_out=self.dtype_out,
                neutral=self.neutral,
                reduce_expr=self.reduce_expr,
                map_expr=expr,
                arguments=arguments,
                preamble="\n".join([cluda_preamble, preamble])
            )
            # only code we generate is saved here.
            self.source = cluda_preamble + preamble
            # FIXME: it is difficult to get the sources from pycuda.
            self.all_source = self.source
            return knl

    def _correct_return_type(self, c_data):
        code = self.tp.blocks[-1].code.splitlines()
        if self._config.use_openmp:
            gil = " noexcept nogil"
        else:
            gil = ""
        code[0] = "cdef inline {type} {name}({args}){gil}:".format(
            type=self.type, name=self.func.__name__, args=', '.join(c_data[0]),
            gil=gil
        )
        self.tp.blocks[-1].code = '\n'.join(code)

    def _add_address_space(self, arg):
        if '*' in arg and 'GLOBAL_MEM' not in arg:
            return 'GLOBAL_MEM ' + arg
        else:
            return arg

    def _correct_opencl_address_space(self, c_data):
        code = self.tp.blocks[-1].code.splitlines()
        header_idx = 1
        for line in code:
            if line.rstrip().endswith(')'):
                break
            header_idx += 1

        args = [self._add_address_space(arg) for arg in c_data[0]]
        code[:header_idx] = wrap(
            'WITHIN_KERNEL {type} {func}({args})'.format(
                type=self.type,
                func=self.func.__name__,
                args=', '.join(args)
            ),
            width=78, subsequent_indent=' ' * 4, break_long_words=False
        )
        self.tp.blocks[-1].code = '\n'.join(code)

    def _massage_arg(self, x):
        if isinstance(x, array.Array):
            return x.dev
        elif self.backend != 'cuda' or isinstance(x, np.ndarray):
            return x
        else:
            return np.asarray(x)

    @profile
    def __call__(self, *args):
        c_args = [self._massage_arg(x) for x in args]
        if self.backend == 'cython':
            size = len(c_args[0])
            c_args.insert(0, size)
            return self.c_func(*c_args)
        elif self.backend == 'opencl':
            result = self.c_func(*c_args)
            self.queue.finish()
            return result.get()
        elif self.backend == 'cuda':
            import pycuda.driver as drv
            event = drv.Event()
            result = self.c_func(*c_args)
            event.record()
            event.synchronize()
            return result.get()
        elif self.backend == 'c':
            size = len(c_args[0])
            c_args.insert(0, size)
            return self.c_func(*c_args)


class Reduction(object):
    def __init__(self, reduce_expr, map_func=None, dtype_out=np.float64,
                 neutral='0', backend=None):
        self._reduce_expr = reduce_expr
        self._map_func = map_func
        self._dtype_out = dtype_out
        self._neutral = neutral
        self._backend = backend
        self.reduction = None

    def _setup(self):
        map_func = self._map_func
        reduce_expr = self._reduce_expr
        dtype_out, neutral = self._dtype_out, self._neutral
        backend = array.get_backend(self._backend)
        cfg = get_config()
        if (backend != 'cython' and dtype_out == np.float64 and
           not cfg.use_double):
            dtype_out = np.float32
        if map_func is None or getattr(map_func, '__annotations__', None) and \
                not hasattr(map_func, 'is_jit'):
            self.reduction = ReductionBase(reduce_expr, map_func=map_func,
                                           dtype_out=dtype_out,
                                           neutral=neutral,
                                           backend=backend)
        else:
            from .jit import ReductionJIT
            self.reduction = ReductionJIT(reduce_expr, map_func=map_func,
                                          dtype_out=dtype_out,
                                          neutral=neutral,
                                          backend=backend)

    def __dir__(self):
        return sorted(dir(self.reduction) + ['reduction'])

    def __getattr__(self, name):
        return getattr(self.reduction, name)

    def __call__(self, *args, **kwargs):
        if self.reduction is None:
            self._setup()
        return self.reduction(*args, **kwargs)

    def set_backend(self, backend=None):
        self._backend = backend
        self.reduction = None


class ScanBase(object):
    def __init__(self, input=None, output=None, scan_expr="a+b",
                 is_segment=None, dtype=np.float64, neutral='0',
                 complex_map=False, backend=None):
        backend = array.get_backend(backend)
        self.tp = Transpiler(backend=backend, incl_cluda=False)
        self.backend = backend
        self.input_func = input
        self.output_func = output
        self.is_segment_func = is_segment
        self.complex_map = complex_map
        if input is not None:
            self.name = 'scan_' + input.__name__
        else:
            self.name = 'scan'
        self.scan_expr = scan_expr
        self.dtype = dtype
        self.type = dtype_to_ctype(dtype, backend=backend)
        if backend == 'cython':
            # On Windows, INFINITY is not defined so we use INFTY which we
            # internally define.
            self.neutral = neutral.replace('INFINITY', 'INFTY')
        else:
            self.neutral = neutral
        self._config = get_config()
        # This is the source generated for the user code.
        self.source = '# Source not yet generated.'
        # This is all the source code used for the elementwise.
        self.all_source = '# Source not yet generated.'
        self.cython_gen = CythonGenerator()
        self.queue = None
        self.c_func = self._generate()

    def _get_backend_key(self):
        return get_common_cache_key(self)

    def _correct_return_type(self, c_data, modifier):
        code = self.tp.blocks[-1].code.splitlines()
        if self._config.use_openmp:
            gil = " noexcept nogil"
        else:
            gil = ""
        code[0] = "cdef inline {type} {name}_{modifier}({args}){gil}:".format(
            type=self.type, name=self.name, modifier=modifier,
            args=', '.join(c_data[0]), gil=gil
        )
        self.tp.blocks[-1].code = '\n'.join(code)

    def _include_prev_item(self):
        if 'prev_item' in self.tp.blocks[-1].code:
            return True
        else:
            return False

    def _include_last_item(self):
        if 'last_item' in self.tp.blocks[-1].code:
            return True
        else:
            return False

    def _not_ignored(self, args):
        ignore = ['item', 'prev_item', 'last_item', 'i', 'N']
        return [i for (i, x) in enumerate(args) if x not in ignore]

    def _filter_ignored(self, args, indices):
        return [args[x] for x in indices]

    def _generate(self, declarations=None):
        if self.backend == 'opencl':
            return self._generate_opencl_kernel(declarations=declarations)
        elif self.backend == 'cuda':
            return self._generate_cuda_kernel(declarations=declarations)
        elif self.backend == 'cython':
            return self._generate_cython_code(declarations=declarations)
        elif self.backend == 'c':
            return self._generate_c_code(declarations=declarations)

    def _default_cython_input_function(self):
        py_data = (['int i', '{type}[:] input'.format(type=self.type)],
                   ['i', '&input[0]'])
        c_data = (['int i', '{type}* input'.format(type=self.type)],
                  ['i', 'input'])
        input_expr = 'input[i]'
        return py_data, c_data, input_expr

    def _wrap_cython_code(self, func, func_type=None,
                          declarations=None):
        name = self.name
        if func is not None:
            self.tp.add(func, declarations=declarations)
            py_data, c_data = self.cython_gen.get_func_signature(func)
            self._correct_return_type(c_data, func_type)

            cargs = ', '.join(c_data[1])
            expr = '{name}_{modifier}({cargs})'.format(name=name, cargs=cargs,
                                                       modifier=func_type)
        else:
            if func_type == 'input':
                py_data, c_data, expr = self._default_cython_input_function()
            else:
                py_data, c_data, expr = [], [], None

        return py_data, c_data, expr

    def _append_cython_arg_data(self, all_py_data, all_c_data, py_data,
                                c_data):
        if len(c_data) > 0:
            select = self._not_ignored(c_data[1])
            all_py_data[0].extend(self._filter_ignored(py_data[0], select))
            all_py_data[1].extend(self._filter_ignored(py_data[1], select))
            all_c_data[0].extend(self._filter_ignored(c_data[0], select))
            all_c_data[1].extend(self._filter_ignored(c_data[1], select))

    def _generate_cython_code(self, declarations=None):
        all_py_data = [[], []]
        all_c_data = [[], []]

        # Process input function
        py_data, c_data, input_expr = self._wrap_cython_code(
            self.input_func,
            func_type='input',
            declarations=declarations
        )
        self._append_cython_arg_data(all_py_data, all_c_data,
                                     py_data, c_data)

        # Process segment function
        use_segment = True if self.is_segment_func is not None else False
        py_data, c_data, segment_expr = self._wrap_cython_code(
            self.is_segment_func, func_type='segment',
            declarations=declarations
        )
        self._append_cython_arg_data(all_py_data, all_c_data, py_data, c_data)

        # Process output expression
        calc_last_item = False
        calc_prev_item = False

        py_data, c_data, output_expr = self._wrap_cython_code(
            self.output_func, func_type='output',
            declarations=declarations)
        if self.output_func is not None:
            calc_last_item = self._include_last_item()
            calc_prev_item = self._include_prev_item()
        self._append_cython_arg_data(all_py_data, all_c_data, py_data, c_data)

        # Add size argument
        py_defn = ['long SIZE'] + all_py_data[0]
        c_defn = ['long SIZE'] + all_c_data[0]
        py_args = ['SIZE'] + all_py_data[1]
        c_args = ['SIZE'] + all_c_data[1]

        # Only use unique arguments
        py_defn = drop_duplicates(py_defn)
        c_defn = drop_duplicates(c_defn)
        py_args = drop_duplicates(py_args)
        c_args = drop_duplicates(c_args)

        if not hasattr(self.output_func, 'arg_keys'):
            self.output_func.arg_keys = {}
        self.output_func.arg_keys[self._get_backend_key()] = c_args

        if self._config.use_openmp:
            template = Template(text=scan_cy_template)
        else:
            template = Template(text=scan_cy_single_thread_template)
        src = template.render(
            name=self.name,
            type=self.type,
            input_expr=input_expr,
            scan_expr=self.scan_expr,
            output_expr=output_expr,
            neutral=self.neutral,
            c_arg_sig=', '.join(c_defn),
            py_arg_sig=', '.join(py_defn),
            py_args=', '.join(py_args),
            openmp=self._config.use_openmp,
            calc_last_item=calc_last_item,
            calc_prev_item=calc_prev_item,
            use_segment=use_segment,
            is_segment_start_expr=segment_expr,
            complex_map=self.complex_map
        )
        self.source = self.tp.get_code()
        self.tp.add_code(src)
        self.tp.compile()
        self.all_source = self.tp.source
        return getattr(self.tp.mod, 'py_' + self.name)

    def _generate_c_code(self, declarations=None):
        self.pyb11_backend = c_backend.CBackend()
        if not self.input_func:
            @annotate(i='int', ary=f'{self.type}p', return_=f'{self.type}')
            def input_expr(i, ary):
                return ary[i]
            self.input_func = input_expr

        self.tp.add(self.input_func, declarations=declarations)
        pyb_data_in, c_data_in = self.pyb11_backend.get_func_signature_pyb11(
            self.input_func)
        self.tp.add(self.output_func, declarations=declarations)
        self.source = self.tp.get_code()
        openmp = self._config.use_openmp
        if openmp:
            self.source = '#include <omp.h>\n' + self.source
        c_call_in = c_data_in[1]
        pyb_data_out, c_data_out = self.pyb11_backend.get_func_signature_pyb11(
            self.output_func)
        c_call_out = c_data_out[1]
        c_call_default = ['ary', 'N']
        c_internal_var = ['item', 'prev_item', 'last_item']
        predefined_vars = c_call_default + c_internal_var

        c_args_in_extra = [[], []]
        pyb_args_in_extra = [[], []]
        for i, var in enumerate(c_call_in[1:]):
            if var not in predefined_vars:
                c_args_in_extra[0].append(c_data_in[0][i + 1])
                c_args_in_extra[1].append(var)
                pyb_args_in_extra[0].append(pyb_data_in[0][i + 1])
                pyb_args_in_extra[1].append(pyb_data_in[1][i + 1])
        c_args_out_extra = [[], []]
        pyb_args_extra = [[], []]
        for i, var in enumerate(c_call_out[1:]):
            if var not in predefined_vars:
                c_args_out_extra[0].append(c_data_out[0][i + 1])
                c_args_out_extra[1].append(var)
                pyb_args_extra[0].append(pyb_data_out[0][i + 1])
                pyb_args_extra[1].append(pyb_data_out[1][i + 1])

        c0 = c_args_in_extra[0]
        c1 = c_args_in_extra[1]
        c_args_in_extra_str = f", {','.join(c0)}" if c1 else ""
        c_call_in_extra_str = f", {','.join(c1)}" if c1 else ""

        c_args_extra = c_args_out_extra.copy()
        for i, var in enumerate(c_args_in_extra[1]):
            if var not in c_args_extra[1]:
                c_args_extra[0].append(c_args_in_extra[0][i])
                c_args_extra[1].append(var)
                pyb_args_extra[0].append(pyb_args_in_extra[0][i])
                pyb_args_extra[1].append(pyb_args_in_extra[1][i])

        if not hasattr(self.output_func, 'arg_keys'):
            self.output_func.arg_keys = {}
        self.output_func.arg_keys[self._get_backend_key(
        )] = c_call_default + c_args_extra[1]

        c0 = c_args_extra[0]
        c1 = c_args_extra[1]
        p0 = pyb_args_extra[0]
        p1 = pyb_args_extra[1]
        c_args_extra_str = f", {', '.join(c0)}" if c1 else ""
        c_call_extra_str = f", {', '.join(c1)}" if c1 else ""
        pyb_args_extra_str = f", {', '.join(p0)}" if p1 else ""
        pyb_call_extra_str = f", {', '.join(p1)}" if p1 else ""

        ip_fname = self.input_func.__name__
        op_fname = self.output_func.__name__
        c_call_in_str = f"{ip_fname}({', '.join(c_call_in)})"
        c_call_out_str = f"{op_fname}({', '.join(c_call_out)})"

        template_scan = Template(c_backend.scan_c_template)
        src_scan = template_scan.render(
            scan_expr=self.scan_expr,
            scan_input_expr_call=c_call_in_str,
            scan_output_expr_call=c_call_out_str,
            args_extra=c_args_extra_str,
            args_in_extra=c_args_in_extra_str,
            call_extra=c_call_extra_str,
            call_in_extra=c_call_in_extra_str,
            name=self.name,
            type=self.type,
            pyb_args=pyb_args_extra_str,
            pyb_call=pyb_call_extra_str,
            openmp=openmp
        )
        self.all_source = self.source + src_scan
        hash_fn = get_md5(self.all_source)
        modname = f'm_{hash_fn}'

        pybind_template = Template(c_backend.scan_c_pybind)
        src_pybind = pybind_template.render(
            name=modname,
            type=self.type,
            pyb_args=pyb_args_extra_str,
            pyb_call=pyb_call_extra_str,
            neutral=self.neutral
        )

        self.all_source += src_pybind

        mod = Cmodule(self.all_source, hash_fn, openmp=openmp,
                      extra_inc_dir=[pybind11.get_include()])
        module = mod.load()
        return getattr(module, modname)

    def _wrap_ocl_function(self, func, func_type=None, declarations=None):
        if func is not None:
            self.tp.add(func, declarations=declarations)
            py_data, c_data = self.cython_gen.get_func_signature(func)
            self._correct_opencl_address_space(c_data, func, func_type)
            name = func.__name__
            expr = '{func}({args})'.format(
                func=name,
                args=', '.join(c_data[1])
            )

            select = self._not_ignored(c_data[1])
            arguments = self._filter_ignored(c_data[0], select)
            c_args = self._filter_ignored(c_data[1], select)
        else:
            if func_type == 'input':
                if self.backend == 'opencl':
                    arguments = ['__global %(type)s *input' %
                                 {'type': self.type}]
                elif self.backend == 'cuda':
                    arguments = ['%(type)s *input' % {'type': self.type}]
                expr = 'input[i]'
                c_args = ['input']
            else:
                arguments = []
                expr = None
                c_args = []
        return expr, arguments, c_args

    def _get_scan_expr_opencl_cuda(self):
        if self.is_segment_func is not None:
            return '(across_seg_boundary ? b : (%s))' % self.scan_expr
        else:
            return self.scan_expr

    def _get_opencl_cuda_code(self, declarations=None):
        input_expr, input_args, input_c_args = \
            self._wrap_ocl_function(self.input_func, func_type='input',
                                    declarations=declarations)

        output_expr, output_args, output_c_args = \
            self._wrap_ocl_function(self.output_func, func_type='output',
                                    declarations=declarations)

        segment_expr, segment_args, segment_c_args = \
            self._wrap_ocl_function(self.is_segment_func,
                                    declarations=declarations)

        scan_expr = self._get_scan_expr_opencl_cuda()

        preamble = convert_to_float_if_needed(self.tp.get_code())

        args = input_args + segment_args + output_args
        args = drop_duplicates(args)
        arg_defn = convert_to_float_if_needed(','.join(args))

        c_args = input_c_args + segment_c_args + output_c_args
        c_args = drop_duplicates(c_args)
        if not hasattr(self.output_func, 'arg_keys'):
            self.output_func.arg_keys = {}
        self.output_func.arg_keys[self._get_backend_key()] = c_args

        return scan_expr, arg_defn, input_expr, output_expr, \
            segment_expr, preamble

    def _generate_opencl_kernel(self, declarations=None):
        scan_expr, arg_defn, input_expr, output_expr, \
            segment_expr, preamble = self._get_opencl_cuda_code(
                declarations=declarations
            )

        from .opencl import get_context, get_queue
        from pyopencl.scan import GenericScanKernel
        ctx = get_context()
        self.queue = get_queue()
        knl = GenericScanKernel(
            ctx,
            dtype=self.dtype,
            arguments=arg_defn,
            input_expr=input_expr,
            scan_expr=scan_expr,
            neutral=self.neutral,
            output_statement=output_expr,
            is_segment_start_expr=segment_expr,
            preamble=preamble
        )
        self.source = preamble
        if knl.first_level_scan_info.kernel.program.source:
            self.all_source = '\n'.join([
                '// ----- Level 1 ------',
                knl.first_level_scan_info.kernel.program.source,
                '// ----- Level 2 ------',
                knl.second_level_scan_info.kernel.program.source,
                '// ----- Final output ------',
                knl.final_update_info.kernel.program.source,
            ])
        else:
            self.all_source = self.source
        return knl

    def _generate_cuda_kernel(self, declarations=None):
        scan_expr, arg_defn, input_expr, output_expr, \
            segment_expr, preamble = self._get_opencl_cuda_code(
                declarations=declarations
            )

        from .cuda import set_context, GenericScanKernel
        set_context()
        knl = GenericScanKernel(
            dtype=self.dtype,
            arguments=arg_defn,
            input_expr=input_expr,
            scan_expr=scan_expr,
            neutral=self.neutral,
            output_statement=output_expr,
            is_segment_start_expr=segment_expr,
            preamble=preamble
        )
        self.source = preamble
        # FIXME: Difficult to get the pycuda sources
        self.all_source = self.source
        return knl

    def _add_address_space(self, arg):
        if '*' in arg and 'GLOBAL_MEM' not in arg:
            return 'GLOBAL_MEM ' + arg
        else:
            return arg

    def _correct_opencl_address_space(self, c_data, func, func_type):
        return_type = 'void' if func_type == 'output' else self.type
        code = self.tp.blocks[-1].code.splitlines()
        header_idx = 1
        for line in code:
            if line.rstrip().endswith(')'):
                break
            header_idx += 1

        args = [self._add_address_space(arg) for arg in c_data[0]]
        code[:header_idx] = wrap(
            'WITHIN_KERNEL {type} {func}({args})'.format(
                type=return_type,
                func=func.__name__,
                args=', '.join(args)
            ),
            width=78, subsequent_indent=' ' * 4, break_long_words=False
        )
        self.tp.blocks[-1].code = '\n'.join(code)

    def _massage_arg(self, x):
        if isinstance(x, array.Array):
            return x.dev
        elif self.backend != 'cuda' or isinstance(x, np.ndarray):
            return x
        else:
            return np.asarray(x)

    @profile
    def __call__(self, **kwargs):
        c_args_dict = {k: self._massage_arg(x) for k, x in kwargs.items()}
        if self._get_backend_key() in self.output_func.arg_keys:
            output_arg_keys = self.output_func.arg_keys[
                self._get_backend_key()
            ]
        else:
            raise ValueError("No kernel arguments found for backend = %s, "
                             "use_openmp = %s, use_double = %s" %
                             self._get_backend_key())

        if self.backend == 'cython':
            size = len(c_args_dict[output_arg_keys[1]])
            c_args_dict['SIZE'] = size
            self.c_func(*[c_args_dict[k] for k in output_arg_keys])
        elif self.backend == 'opencl':
            self.c_func(*[c_args_dict[k] for k in output_arg_keys])
            self.queue.finish()
        elif self.backend == 'cuda':
            import pycuda.driver as drv
            event = drv.Event()
            self.c_func(*[c_args_dict[k] for k in output_arg_keys])
            event.record()
            event.synchronize()
        elif self.backend == 'c':
            size = len(c_args_dict[output_arg_keys[0]])
            c_args_dict['N'] = size
            self.c_func(*[c_args_dict[k] for k in output_arg_keys])


class Scan(object):
    def __init__(self, input=None, output=None, scan_expr="a+b",
                 is_segment=None, dtype=np.float64, neutral='0',
                 complex_map=False, backend=None):
        self._input = input
        self._output = output
        self._scan_expr = scan_expr
        self._is_segment = is_segment
        self._dtype = dtype
        self._neutral = neutral
        self._complex_map = complex_map
        self._backend = backend
        self.scan = None

    def _setup(self):
        input, output, scan_expr = self._input, self._output, self._scan_expr
        is_segment, dtype = self._is_segment, self._dtype
        neutral, complex_map = self._neutral, self._complex_map
        backend = array.get_backend(self._backend)
        # FIXME: Revisit these conditions
        input_base = input is None or \
            getattr(input, '__annotations__', None) and \
            not hasattr(input, 'is_jit')
        output_base = output is None or \
            getattr(output, '__annotations__', None) and \
            not hasattr(input, 'is_jit')
        is_segment_base = is_segment is None or \
            getattr(is_segment, '__annotations__', None) and \
            not hasattr(input, 'is_jit')

        if input_base and output_base and is_segment_base:
            self.scan = ScanBase(input=input, output=output,
                                 scan_expr=scan_expr,
                                 is_segment=is_segment,
                                 dtype=dtype, neutral=neutral,
                                 complex_map=complex_map,
                                 backend=backend)
        else:
            from .jit import ScanJIT
            self.scan = ScanJIT(input=input, output=output,
                                scan_expr=scan_expr,
                                is_segment=is_segment,
                                dtype=dtype, neutral=neutral,
                                complex_map=complex_map,
                                backend=backend)

    def __dir__(self):
        return sorted(dir(self.scan) + ['scan'])

    def __getattr__(self, name):
        return getattr(self.scan, name)

    def __call__(self, **kwargs):
        if self.scan is None:
            self._setup()
        self.scan(**kwargs)

    def set_backend(self, backend=None):
        self._backend = backend
        self.scan = None
