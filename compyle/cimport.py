import os
import hashlib
import json
from stat import S_ISREG
import struct
import io
import importlib
import logging
import shutil
import sys

from os.path import exists, expanduser, isdir, join

from distutils.sysconfig import get_config_vars, customize_compiler
from distutils.util import get_platform

from distutils.extension import Extension
from distutils.command import build_ext
from distutils.core import setup
from distutils.errors import CompileError, LinkError
from distutils.ccompiler import new_compiler, get_default_compiler
from webbrowser import get

from .ext_module import get_platform_dir, get_md5, get_ext_extension
from .capture_stream import CaptureMultipleStreams  # noqa: 402


_TAG = b"cmodule_compyle"
_FMT = struct.Struct("q" + str(len(_TAG)) + "s")

logger = logging.getLogger(__name__)


def is_checksum_valid(module_data):
    """
    Load the saved checksum from the extension file check if it matches the
    checksum computed from current source files.
    """
    deps, old_checksum = _load_checksum_trailer(module_data)
    if old_checksum is None:
        return False  # Already logged error in load_checksum_trailer.
    try:
        return old_checksum == get_md5(module_data)
    except OSError as e:
        return False


def _load_checksum_trailer(module_data):
    try:
        with open(module_data["ext_path"], "rb") as f:
            f.seek(-_FMT.size, 2)
            json_len, tag = _FMT.unpack(f.read(_FMT.size))
            if tag != _TAG:
                return None, None
            f.seek(-(_FMT.size + json_len), 2)
            json_s = f.read(json_len)
    except FileNotFoundError:
        return None, None

    try:
        old_checksum = json.loads(json_s)
    except ValueError:
        return None, None
    return old_checksum

def _save_checksum_trailer(ext_path, cur_checksum):
    # We can just append the checksum to the shared object; this is effectively
    # legal (see e.g. https://stackoverflow.com/questions/10106447).
    dump = json.dumps(cur_checksum).encode("ascii")
    dump += _FMT.pack(len(dump), _TAG)
    with open(ext_path, "ab") as file:
        file.write(dump)

def wget_tpnd_headers():
    import requests
    baseurl = 'https://gitlab.inria.fr/tapenade/tapenade/-/raw/3.16/ADFirstAidKit/'
    files = ['adBuffer.c', 'adBuffer.h', 'adStack.c', 'adStack.h']
    reqs = [requests.get(baseurl, file) for file in files]
    saveloc = get_tpnd_obj_dir()
    if not os.path.exists(saveloc):
        os.mkdir(saveloc)
    
    for file, r in zip(files, reqs):
        with open(join(saveloc, file), 'wb') as f:
            f.write(r.content)
    

def get_tpnd_obj_dir():
    plat_dir = get_platform_dir()
    root = expanduser(join('~', '.compyle', 'source', plat_dir))
    tpnd_dir = join(root, 'tapenade_src')
    return tpnd_dir

    
def compile_tapenade_source(verbose=0):
    try:
        with CaptureMultipleStreams() as stream:
            wget_tpnd_headers()
            os.environ["CC"]='g++'
            compiler = new_compiler(verbose=1)
            customize_compiler(compiler)
            compiler.compile([join(get_tpnd_obj_dir(), 'adBuffer.c')], output_dir=get_tpnd_obj_dir(), extra_preargs=['-c', '-fPIC'])
            compiler.compile([join(get_tpnd_obj_dir(), 'adStack.c')], output_dir=get_tpnd_obj_dir(), extra_preargs=['-c', '-fPIC'])
            objdir = join(get_tpnd_obj_dir(), get_tpnd_obj_dir()[1:])
            shutil.move(join(objdir, 'adBuffer.o'), join(get_tpnd_obj_dir(), 'adBuffer.o'))
            shutil.move(join(objdir, 'adStack.o'), join(get_tpnd_obj_dir(), 'adStack.o'))
    except (CompileError, LinkError):
        hline = "*"*80
        print(hline + "\nERROR")
        s_out = stream.get_output()
        print(s_out[0])
        print(s_out[1])
        msg = "Compilation of tapenade source failed, please check "\
                "error messages above."
        print(hline + "\n" + msg)
        sys.exit(1)


class Cmodule:
    def __init__(self, name, src, root=None, verbose=False, extra_inc_dir=[], extra_link_args=[], extra_compile_args=[]):
        self.name = name
        self.src = src
        self.hash = get_md5(src)
        self.verbose = verbose
        self.extra_inc_dir = extra_inc_dir
        self.extra_link_args = extra_link_args
        self.extra_compile_args = extra_compile_args
        
        self._setup_root(root)
        self._setup_filenames()
        
    
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
        base = 'm_' + self.hash
        self.src_path = join(self.root, base + '.cpp')
        self.ext_path = join(self.root, self.name + get_ext_extension())
        
    def is_build_needed(self):
        return True
    
    def build(self):
        if self.is_build_needed:
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
                shutil.move(join(self.build_dir, self.name + get_ext_extension()), self.ext_path)
               
        except:
            hline = "*"*80
            print(hline + "\nERROR")
            s_out = stream.get_output()
            print(s_out[0])
            print(s_out[1])
            msg = "Compilation of code failed, please check "\
                    "error messages above."
            print(hline + "\n" + msg)
            os.remove(self.src_path)
            sys.exit(1)
        
    def write_and_build(self):
        """Write source and build the extension module"""
        if not exists(self.src_path):
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
        logger.info(msg)
        if self.verbose:
            print(msg)