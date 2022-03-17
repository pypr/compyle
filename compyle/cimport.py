import os
from stat import S_ISREG
import io
import importlib
import shutil
import sys
from filelock import FileLock, Timeout

from os.path import exists, expanduser, isdir, join

import pybind11
from distutils.extension import Extension
from distutils.command import build_ext
from distutils.core import setup

from .ext_module import get_platform_dir, get_md5, get_ext_extension
from .capture_stream import CaptureMultipleStreams  # noqa: 402


class Cmodule:
    def __init__(self, src, hash_fn, root=None, verbose=False, extra_inc_dir=[pybind11.get_include()], extra_link_args=[], extra_compile_args=[]):
        self.src = src
        self.hash = hash_fn
        self.name = f'm_{self.hash}'
        self.verbose = verbose
        self.extra_inc_dir = extra_inc_dir
        self.extra_link_args = extra_link_args
        self.extra_compile_args = extra_compile_args

        self._setup_root(root)
        self._setup_filenames()
        self.lock = FileLock(self.lock_path, timeout=120)

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
                pass

    def _write_source(self):
        if not exists(self.src_path):
            with io.open(self.src_path, 'w', encoding='utf-8') as f:
                f.write(self.src)

    def _setup_filenames(self):
        self.src_path = join(self.root, self.name + '.cpp')
        self.ext_path = join(self.root, self.name + get_ext_extension())
        self.lock_path = join(self.root, self.name + '.lock')

    def is_build_needed(self):
        return not exists(self.ext_path)

    def build(self):
        ext = Extension(name=self.name,
                        sources=[self.src_path],
                        language='c++',
                        include_dirs=self.extra_inc_dir,
                        extra_link_args=self.extra_link_args,
                        extra_compile_args=self.extra_compile_args)
        args = [
            "build_ext",
            "--build-lib=" + self.build_dir,
            "--build-temp=" + self.build_dir,
            "-v",
        ]

        try:
            with CaptureMultipleStreams() as stream:
                setup(name=self.name,
                      ext_modules=[ext],
                      script_args=args,
                      cmdclass={"build_ext": build_ext.build_ext})
                shutil.move(join(self.build_dir, self.name +
                            get_ext_extension()), self.ext_path)

        except:
            hline = "*"*80
            print(hline + "\nERROR")
            s_out = stream.get_output()
            print(s_out[0])
            print(s_out[1])
            msg = "Compilation of code failed, please check "\
                "error messages above."
            print(hline + "\n" + msg)
            sys.exit(1)

    def write_and_build(self):
        """Write source and build the extension module"""
        if self.is_build_needed():
            with self.lock:
                self._write_source()
                self.build()
        else:
            self._message("Precompiled code from:", self.src_path)

    def load(self):
        self.write_and_build()
        spec = importlib.util.spec_from_file_location(self.name, self.ext_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _message(self, *args):
        msg = ' '.join(args)
        if self.verbose:
            print(msg)
