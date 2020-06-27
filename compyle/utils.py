import inspect
import argparse


def getsourcelines(obj):
    '''Given an object return the source code that defines it as a list of
    lines along with the starting line.
    '''
    try:
        return inspect.getsourcelines(obj)
    except Exception:
        if hasattr(obj, 'source'):
            return obj.source.splitlines(True), 0
        else:
            raise


def getsource(obj):
    '''Given an object return the source that defines it.
    '''
    try:
        return inspect.getsource(obj)
    except Exception:
        if hasattr(obj, 'source'):
            return obj.source
        else:
            raise


class ArgumentParser(argparse.ArgumentParser):
    '''Standard argument parser for compyle applications.
    Includes arguments for backend, openmp and use_double
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # setup standard arguments
        self.add_argument(
            '-b', '--backend', action='store', dest='backend', default='cython',
            help='Choose the backend.'
        )
        self.add_argument(
            '--openmp', action='store_true', dest='openmp', default=False,
            help='Use OpenMP.'
        )
        self.add_argument(
            '--use-double', action='store_true', dest='use_double',
            default=False, help='Use double precision on the GPU.'
        )
