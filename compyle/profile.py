""" Utils for profiling kernels
"""

from contextlib import contextmanager
from collections import defaultdict
import time
from .config import get_config


def _make_default():
    return dict(calls=0, time=0.0)


_current_level = 0
_profile_info = defaultdict(
    lambda: defaultdict(_make_default)
)


def _record_profile(name, time):
    global _profile_info, _current_level
    li = _profile_info[_current_level]
    li[name]['time'] += time
    li[name]['calls'] += 1


@contextmanager
def profile_ctx(name):
    """ Context manager for profiling

    For profiling a function f, it can be used as follows::

    with profile_ctx('f'):
        f()
    """
    global _current_level
    _current_level += 1
    start = time.time()
    try:
        yield start
        end = time.time()
    finally:
        _current_level -= 1
    _record_profile(name, end - start)


def profile(method=None, name=None):
    """Decorator for profiling a function. Can be used as follows::

    @profile
    def f():
        pass


    If explicitly passed a name, with @profile(name='some name'), it will use
    the given name. Otherwise, if the function is a class method, and the class
    has a `self.name` attribute, it will use that. Otherwise, it will use the
    method's qualified name to record the profile.

    """
    def make_wrapper(method):
        def wrapper(*args, **kwargs):
            self = args[0] if len(args) else None
            if name is None:
                if hasattr(self, method.__name__) and hasattr(self, 'name'):
                    p_name = self.name
                else:
                    p_name = getattr(method, '__qualname__', method.__name__)
            else:
                p_name = name
            with profile_ctx(p_name):
                return method(*args, **kwargs)
        wrapper.__doc__ = method.__doc__
        return wrapper
    if method is None:
        return make_wrapper
    else:
        return make_wrapper(method)


class ProfileContext:
    """Used for a low-level profiling context.

    This is typically useful in Cython code where decorators are not usable and
    using a context manager makes the code hard to read.

    Example
    -------

    p = ProfileContext('some_func')
    do_something()
    p.stop()

    """
    def __init__(self, name):
        self.name = name
        global _current_level
        _current_level += 1
        self.start = time.time()

    def stop(self):
        global _current_level
        _current_level -= 1
        _record_profile(self.name, time.time() - self.start)


def get_profile_info():
    global _profile_info
    return _profile_info


def print_profile():
    global _profile_info
    hr = '-'*70
    print(hr)
    if len(_profile_info) == 0:
        print("No profiling information available")
        print(hr)
        return
    print("Profiling info:")
    print(
        "{:<6} {:<40} {:<10} {:<10}".format(
            'Level', 'Function', 'N calls', 'Time')
    )
    tot_time = 0
    for level in range(0, min(len(_profile_info), 2)):
        profile_data = sorted(
            _profile_info[level].items(), key=lambda x: x[1]['time'],
            reverse=True
        )
        for kernel, data in profile_data:
            print("{:<6} {:<40} {:<10} {:<10.3g}".format(
                level, kernel, data['calls'], data['time'])
            )
            if level == 0:
                tot_time += data['time']
    print("Total profiled time: %g secs" % tot_time)
    print(hr)


def profile2csv(fname, info=None):
    '''Write profile info to a CSV file.

    If the optional info argument is passed, it is used as the profile info.
    The `info` argument is a list, potentially one for each rank (for a
    parallel simulation).
    '''
    if info is None:
        info = [get_profile_info()]
    with open(fname, 'w') as f:
        f.write("{0},{1},{2},{3},{4}\n".format(
            'rank', 'level', 'function', 'calls', 'time')
        )
        for rank in range(len(info)):
            pdata = info[rank]
            for level in sorted(pdata.keys()):
                profile_data = sorted(
                    pdata[level].items(), key=lambda x: x[1]['time'],
                    reverse=True
                )
                for name, data in profile_data:
                    f.write("{0},{1},{2},{3},{4}\n".format(
                        rank, level, name, data['calls'], data['time']
                    ))


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
