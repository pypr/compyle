import inspect
import argparse
import atexit
from compyle.config import get_config
from compyle.profile import print_profile


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
            choices = ['cython', 'opencl', 'cuda'],
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
        self.add_argument(
            '--suppress-warnings', action='store_true',
            dest='suppress_warnings',
            default=False, help='Suppress warnings'
        )
        self.add_argument(
            '--profile', action='store_true',
            dest='profile',
            default=False, help='Print profiling info'
        )
        self.profile_registered = False

    def _set_config_options(self, options):
        get_config().use_openmp = options.openmp
        get_config().use_double = options.use_double
        get_config().suppress_warnings = options.suppress_warnings
        if options.backend == 'opencl':
            get_config().use_opencl = True
        if options.backend == 'cuda':
            get_config().use_cuda = True
        if options.profile and not self.profile_registered:
            get_config().profile = True
            atexit.register(print_profile)
            self.profile_registered = True

    def parse_args(self, *args, **kwargs):
        options = super().parse_args(*args, **kwargs)
        self._set_config_options(options)
        return options

    def parse_known_args(self, *args, **kwargs):
        options, unknown = super().parse_known_args(*args, **kwargs)
        self._set_config_options(options)
        return options, unknown
