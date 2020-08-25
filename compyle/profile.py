""" Utils for profiling kernels
"""

import time
from contextlib import contextmanager
from collections import defaultdict
from .config import get_config


_profile_info = defaultdict(lambda: {'calls': 0, 'time': 0})


def _record_profile(name, time):
    global _profile_info
    _profile_info[name]['time'] += time
    _profile_info[name]['calls'] += 1


@contextmanager
def profile_ctx(name):
    """ Context manager for profiling

    For profiling a function f, it can be used as follows::

    with profile_ctx('f'):
        f()
    """
    start = time.time()
    yield start
    end = time.time()
    _record_profile(name, end - start)


def profile(method):
    """Decorator for profiling a function. Can be used as follows::

    @profile
    def f():
        pass

    If used on a class method, it will use self.name as the
    name for recording the profile. If 'name' attribute is not available, it
    will use the method name
    """
    def wrapper(*args, **kwargs):
        self = args[0] if len(args) else None
        with profile_ctx(getattr(self, "name", method.__name__)):
            return method(*args, **kwargs)
    return wrapper


def get_profile_info():
    global _profile_info
    return _profile_info


def print_profile():
    global _profile_info
    profile_data = sorted(_profile_info.items(), key=lambda x: x[1]['time'],
                          reverse=True)
    if len(_profile_info) == 0:
        print("No profiling information available")
        return
    print("Profiling info:")
    print("{:<40} {:<10} {:<10}".format('Function', 'N calls', 'Time'))
    tot_time = 0
    for kernel, data in profile_data:
        print("{:<40} {:<10} {:<10}".format(
            kernel,
            data['calls'],
            data['time']))
        tot_time += data['time']
    print("Total profiled time: %g secs" % tot_time)


def profile_kernel(kernel, name, backend=None):
    """For profiling raw PyCUDA/PyOpenCL kernels or cython functions
    """
    from compyle.array import get_backend
    backend = get_backend(backend)

    def _profile_knl(*args, **kwargs):
        if backend == 'opencl':
            start = time.time()
            event = kernel(*args, **kwargs)
            event.wait()
            end = time.time()
            _record_profile(name, end - start)
            return event
        elif backend == 'cuda':
            exec_time = kernel(*args, **kwargs, time_kernel=True)
            _record_profile(name, exec_time)
            return exec_time
        else:
            start = time.time()
            kernel(*args, **kwargs)
            end = time.time()
            _record_profile(name, end - start)

    if get_config().profile:
        wgi = getattr(kernel, 'get_work_group_info', None)
        if wgi is not None:
            _profile_knl.get_work_group_info = wgi
        return _profile_knl
    else:
        return kernel


def named_profile(name, backend=None):
    """Decorator for profiling raw PyOpenCL/PyCUDA kernels or cython functions.
    This can be used on a function that returns a raw PyCUDA/PyOpenCL kernel

    For example::

    @named_profile('prefix_sum')
    def _get_prefix_sum(ctx):
        return GenericScanKernel(ctx, np.int32,
                                 arguments="__global int *ary",
                                 input_expr="ary[i]",
                                 scan_expr="a+b", neutral="0",
                                 output_statement="ary[i] = prev_item")
    """
    from compyle.array import get_backend
    backend = get_backend(backend)

    def _decorator(f):
        if name is None:
            n = f.__name__
        else:
            n = name

        def _profiled_kernel_generator(*args, **kwargs):
            kernel = f(*args, **kwargs)
            return profile_kernel(kernel, n, backend=backend)

        return _profiled_kernel_generator

    return _decorator
