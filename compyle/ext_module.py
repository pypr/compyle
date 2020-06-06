# Standard library imports
from contextlib import contextmanager
from distutils.sysconfig import get_config_vars
from distutils.util import get_platform
from distutils.errors import CompileError, LinkError
import hashlib
import importlib
import io
import logging
import numpy
import os
from os.path import exists, expanduser, isdir, join
import platform
from pyximport import pyxbuild
import shutil
import sys
import time

# Conditional/Optional imports.
if sys.platform == 'win32':
    from setuptools.extension import Extension
else:
    from distutils.extension import Extension

PY3 = sys.version_info.major > 2

# Package imports.
from .config import get_config  # noqa: 402
from .capture_stream import CaptureMultipleStreams  # noqa: 402


logger = logging.getLogger(__name__)


def get_platform_dir():
    return 'py{version}-{platform_dir}'.format(
        version=sys.version[:3], platform_dir=get_platform()
    )


def get_ext_extension():
    """Return the system's file extension for Extension modules."""
    vars = get_config_vars()
    return vars.get('EXT_SUFFIX', vars.get('SO'))


def get_md5(data):
    """Return the MD5 sum of the given data.
    """
    return hashlib.md5(data.encode()).hexdigest()


def get_unicode(s):
    if PY3:
        return s
    else:
        return unicode(s)


def get_openmp_flags():
    """Return the OpenMP flags for the platform.

    This returns two lists, [extra_compile_args], [extra_link_args]
    """
    if sys.platform == 'win32':
        return ['/openmp'], []
    elif sys.platform == 'darwin':
        if (os.environ.get('CC') is not None and
           os.environ.get('CXX') is not None):
            return ['-fopenmp'], ['-fopenmp']
        else:
            return ['-Xpreprocessor', '-fopenmp'], ['-lomp']
    else:
        return ['-fopenmp'], ['-fopenmp']


class ExtModule(object):
    """Encapsulates the generated code, extension module etc.
    """
    def __init__(self, src, extension='pyx', root=None, verbose=False,
                 depends=None, extra_inc_dirs=None, extra_compile_args=None,
                 extra_link_args=None):
        """Initialize ExtModule.

        Parameters
        -----------

        src : str : source code.

        ext : str : extension for source code file.
            Do not specify the '.' (defaults to 'pyx').

        root : str: root of directory to store code and modules in.
            If not set it defaults to "~/.cpy/source/<platform-directory>".
            where <platform-directory> is platform specific.

        verbose : Bool : Print messages for convenience.

        depends : list : a list of modules that this extension depends on
            if any of these have an m_time greater than the compiled extension
            module, the extension will be recompiled.

        extra_inc_dirs : list : a list of directories to look for .pxd, .h
            and other files.

        extra_compile_args: list : a list of extra compilation flags.

        extra_link_args: list : a list of extra link flags.
        """
        self._setup_root(root)
        self.code = src
        self.hash = get_md5(src)
        self.extension = extension
        self.name = 'm_{0}'.format(self.hash)
        self._setup_filenames()
        self.verbose = verbose
        self.depends = depends
        self.extra_inc_dirs = extra_inc_dirs if extra_inc_dirs else []
        self._add_local_include()
        self.extra_compile_args = (
            extra_compile_args if extra_compile_args else []
        )
        self.extra_link_args = extra_link_args if extra_link_args else []

    def _add_local_include(self):
        if 'bsd' in platform.system().lower():
            local = '/usr/local/include'
            if local not in self.extra_inc_dirs:
                self.extra_inc_dirs.append(local)

    def _setup_filenames(self):
        base = self.name
        self.src_path = join(self.root, base + '.' + self.extension)
        self.ext_path = join(self.root, base + get_ext_extension())
        self.lock_path = join(self.root, base + '.lock')

    @contextmanager
    def _lock(self, timeout=90):
        t1 = time.time()

        def _is_timed_out():
            if timeout is None:
                return False
            else:
                return (time.time() - t1) > timeout

        def _try_to_lock():
            if not exists(self.lock_path):
                try:
                    os.mkdir(self.lock_path)
                except OSError:
                    return False
                else:
                    return True
            return False

        while not _try_to_lock():
            time.sleep(0.1)
            if _is_timed_out():
                break
        try:
            yield
        finally:
            os.rmdir(self.lock_path)

    def _write_source(self, path):
        if not exists(path):
            with io.open(path, 'w', encoding='utf-8') as f:
                f.write(get_unicode(self.code))

    def _setup_root(self, root):
        if root is None:
            plat_dir = get_platform_dir()
            self.root = expanduser(join('~', '.compyle', 'source', plat_dir))
        else:
            self.root = root

        self.build_dir = join(self.root, 'build')

        if not isdir(self.build_dir):
            try:
                os.makedirs(self.build_dir)
            except OSError:
                # The directory was created at the same time by another
                # process.
                pass

    def _dependencies_have_changed(self):
        depends = self.depends
        if not depends:
            return False
        else:
            ext_mtime = os.stat(self.ext_path).st_mtime
            for name in depends:
                try:
                    mod = importlib.import_module(name)
                    mod_mtime = os.stat(mod.__file__).st_mtime
                    if ext_mtime < mod_mtime:
                        return True
                except ImportError:
                    pass
            return False

    def should_recompile(self):
        if not exists(self.ext_path):
            return True
        elif self._dependencies_have_changed():
            return True
        else:
            return False

    def build(self, force=False):
        """Build source into an extension module.  If force is False
        previously compiled module is returned.
        """
        if force or self.should_recompile():
            self._message("Compiling code at:", self.src_path)
            inc_dirs = [numpy.get_include()]
            inc_dirs.extend(self.extra_inc_dirs)
            extra_compile_args, extra_link_args = (
                self._get_extra_args()
            )

            extension = Extension(
                name=self.name, sources=[self.src_path],
                include_dirs=inc_dirs,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                language="c++"
            )

            if not hasattr(sys.stdout, 'errors'):
                # FIXME: This happens when nosetests replaces the
                # stdout with the a Tee instance.  This Tee instance
                # does not have errors which breaks the tests so we
                # disable verbose reporting.
                script_args = []
            else:
                script_args = ['--verbose']
            try:
                with CaptureMultipleStreams() as stream:
                    mod = pyxbuild.pyx_to_dll(
                        self.src_path, extension,
                        pyxbuild_dir=self.build_dir,
                        force_rebuild=True,
                        setup_args={'script_args': script_args}
                    )
            except (CompileError, LinkError):
                hline = "*"*80
                print(hline + "\nERROR")
                print(stream.get_output()[0])
                print(stream.get_output()[1])
                msg = "Compilation of code failed, please check "\
                      "error messages above."
                print(hline + "\n" + msg)
                sys.exit(1)
            shutil.copy(mod, self.ext_path)
        else:
            self._message("Precompiled code from:", self.src_path)

    def write_source(self):
        """Writes source without compiling. Used for testing"""
        if not exists(self.src_path):
            with self._lock():
                self._write_source(self.src_path)

    def write_and_build(self):
        """Write source and build the extension module"""
        if not exists(self.ext_path):
            with self._lock():
                self._write_source(self.src_path)
                self.build()
        else:
            self._message("Precompiled code from:", self.src_path)

    def load(self):
        """Load the built extension module.

        Returns
        """
        self.write_and_build()
        spec = importlib.util.spec_from_file_location(self.name, self.ext_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _get_extra_args(self):
        ec, el = self.extra_compile_args, self.extra_link_args
        if get_config().use_openmp:
            _ec, _el = get_openmp_flags()
            return _ec + ec, _el + el
        else:
            return ec, el

    def _message(self, *args):
        msg = ' '.join(args)
        logger.info(msg)
        if self.verbose:
            print(msg)
