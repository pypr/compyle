import pybind11
import inspect
import os
import subprocess
import functools
import re
import numpy as np
from mako.template import Template
from distutils.errors import CompileError

from .profile import profile
from .translator import CConverter
from .transpiler import convert_to_float_if_needed
from . import array
from compyle.api import get_config

from .ext_module import get_md5
from .cimport import Cmodule, get_tpnd_obj_dir, compile_tapenade_source
from .transpiler import Transpiler


pyb11_setup_header = '''

// c code for with PyBind11 binding
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;


'''

pyb11_setup_header_rev = '''

// c code for with PyBind11 binding
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

\n
'''

pyb11_wrap_template = '''
PYBIND11_MODULE(m_${hash_fn}, m) {
    m.def("m_${hash_fn}", [](${pyb_call}){
        return ${name}${FUNC_SUFFIX_F}(${c_call});
    });
}
'''

pyb11_wrap_template_rev = '''
PYBIND11_MODULE(${modname}, m) {
    m.def("${modname}", [](${pyb_call}){
        return ${name}${FUNC_SUFFIX_F}(${c_call});
    });
}
'''

c_backend_template = '''
${c_kernel_defn}

void elementwise_${fn_name}(long int SIZE, ${fn_args})
{
    %if openmp:
        #pragma omp parallel for
    %endif
    for(long int iter = (SIZE - 1); iter >= 0; iter--)
    {
        ${fn_name}(iter, ${fn_call}) ;
    }
}
'''
VAR_SUFFIX_F = '__d'
FUNC_SUFFIX_F = '_d'

VAR_SUFFIX_R = '__b'
FUNC_SUFFIX_R = '_b'


def get_source(f):
    c = CConverter()
    source = c.parse(f)
    return source


def sig_to_pyb_call(par, typ):
    if typ[-1] == "*":
        call = "py::array_t<{}> {}".format(typ[:-1], par)
    else:
        call = typ + " " + str(par)
    return call


def sig_to_c_call(par, typ):
    if typ[-1] == '*':
        call = "({ctype}) {arg}.request().ptr".format(ctype=typ, arg=par)
    else:
        call = "{arg}".format(arg=par)
    return call


def get_diff_signature(f, active, mode='forward'):
    if mode == 'forward':
        VAR_SUFFIX = VAR_SUFFIX_F
    elif mode == 'reverse':
        VAR_SUFFIX = VAR_SUFFIX_R

    sig = inspect.signature(f)
    pyb_c = []
    pyb_py = []
    pure_c = []
    pure_py = []
    for s in sig.parameters:
        typ = str(sig.parameters[s].annotation.type)
        if s not in active:
            pyb_py.append([sig_to_pyb_call(s, typ)])
            pyb_c.append([sig_to_c_call(s, typ)])
            pure_c.append(["{typ} {i}".format(typ=typ, i=s)])
            pure_py.append([s])
        else:
            typstar = typ if typ[-1] == '*' else typ + '*'
            pyb_py.append([
                sig_to_pyb_call(s, typ),
                sig_to_pyb_call(s + VAR_SUFFIX, typstar)
            ])
            pyb_c.append([
                sig_to_c_call(s, typ),
                sig_to_c_call(s + VAR_SUFFIX, typstar)
            ])
            pure_c.append([
                "{typ} {i}".format(typ=typ, i=s),
                "{typ} {i}".format(typ=typstar, i=s + VAR_SUFFIX)
            ])
            pure_py.append([s, s + VAR_SUFFIX])

    pyb_py_all = functools.reduce(lambda x, y: x + y, pyb_py)
    pyb_c_all = functools.reduce(lambda x, y: x + y, pyb_c)
    pure_c = functools.reduce(lambda x, y: x + y, pure_c)
    pure_py = functools.reduce(lambda x, y: x + y, pure_py)

    return pyb_py_all, pyb_c_all, pure_py, pure_c


class GradBase:
    def __init__(self, func, wrt, gradof, mode='forward', backend='tapenade'):
        self.backend = backend
        self.func = func
        self.args = list(inspect.signature(self.func).parameters.keys())
        self.wrt = wrt
        self.gradof = gradof
        self.active = []
        self.mode = mode
        self.name = func.__name__
        self._config = get_config()
        self.source = 'Not yet generated'
        self.grad_source = 'Not yet generated'
        self.grad_all_source = 'Not yet generated'
        self.tapenade_op = 'Not yet generated'
        self.c_func = self.c_gen_error
        self.tp = Transpiler(backend='c')
        self._get_sources()
        self.grad_args, self.grad_types = self._get_grad_def(self.grad_source)
        self._get_active_vars()

    def _get_sources(self):
        self.tp.add(self.func)
        self.source = self.tp.get_code(incl_header=False)
        with open(self.name + '.c', 'w') as f:
            f.write(self.source)

        if self.mode == 'forward':
            command = [
                "tapenade", f"{self.name}.c", "-d", "-o",
                f"{self.name}_forward_diff", "-tgtvarname",
                f"{VAR_SUFFIX_F}", "-tgtfuncname",
                f"{FUNC_SUFFIX_F}",
                "-head",
                f'{self.name}({" ".join(self.wrt)})\({" ".join(self.gradof)})'
                "-nooptim", "recomputeintermediates",
                "-nooptim", "spareinit",
            ]
        elif self.mode == 'reverse':
            command = [
                "tapenade", f"{self.name}.c", "-b", "-o",
                f"{self.name}_reverse_diff", "-adjvarname",
                f"{VAR_SUFFIX_R}", "-adjfuncname",
                f"{FUNC_SUFFIX_R}",
                "-head",
                f'{self.name}({" ".join(self.wrt)})\({" ".join(self.gradof)})',
                "-nooptim", "adjointliveness",
            ]
                # "-nooptim", "diffliveness",
                # "-fixinterface"
                # "-nooptim", "recomputeintermediates",
        else:
            raise ValueError(f"supported modes are 'forward' and 'reverse', got {self.mode}")

        op_tapenade = ""
        try:
            proc = subprocess.run(command, capture_output=True, text=True)
            op_tapenade += proc.stdout
        except subprocess.CalledProcessError as e:
            print(e)
            raise RuntimeError(
                "Encountered errors while differentiating through Tapenade.")
        self.tapenade_op = op_tapenade

        if self.mode == 'forward':
            f_extn = "_forward_diff_d.c"
        elif self.mode == 'reverse':
            f_extn = "_reverse_diff_b.c"

        with open(self.name + f_extn, 'r') as f:
            self.grad_source = f.read()

    def c_gen_error(*args):
        raise RuntimeError("Differentiated function not yet generated")

    def _massage_arg(self, x):
        if isinstance(x, array.Array):
            return x.dev
        elif self.backend != 'cuda' or isinstance(x, np.ndarray):
            return x
        else:
            return np.asarray(x)

    def _get_grad_def(self, src):
        lines_src = src.split("\n")
        n_lines = len(lines_src)
        i = 0
        start = 0
        while i < n_lines:
            # if lines_src[i].strip().startswith(f'void {self.name}'):
            if f'void {self.name}' in lines_src[i].strip():
                start = i
                break
            i += 1
        if i == n_lines:
            raise CompileError('could not find fn definition for derivative')

        end = 0

        while i < n_lines:
            if lines_src[i].strip().endswith("{"):
                end = i
                break
            i += 1
        if i == n_lines:
            raise CompileError('could not find fn definition for derivative')

        src_def = " ".join([i.strip() for i in lines_src[start:end + 1]])
        args_type = re.search(r"\((.*?)\)", src_def).group(1).split(",")
        args = []
        types = []
        for val in args_type:
            temp = val.split()
            if len(temp) == 2:
                t1 = temp[0]
                t2 = temp[1]
            elif len(temp) == 3:
                t1 = temp[0]
                t2 = temp[1] + temp[2]
            else:
                raise CompileError('could not get arguments from generated fn')
            if t2.startswith("*"):
                t1 += "*"
                t2 = t2[1:]
            types.append(t1)
            args.append(t2)
        return args, types

    def _get_active_vars(self):
        if self.mode == 'forward':
            suff = VAR_SUFFIX_F
        elif self.mode == 'reverse':
            suff = VAR_SUFFIX_R

        for i, var in enumerate(self.grad_args):
            if var.endswith(suff):
                self.active.append(self.grad_args[i-1])

    @profile
    def __call__(self, *args, **kw):
        c_args = [self._massage_arg(x) for x in args]

        if self.backend == 'cuda':
            import pycuda.driver as drv
            event = drv.Event()
            self.c_func(*c_args, **kw)
            event.record()
            event.synchronize()

        elif self.backend == 'c':
            self.c_func(*c_args)

        else:
            raise RuntimeError("Given backend not supported, got '{}'".format(
                self.backend))


class ForwardGrad(GradBase):
    def __init__(self, func, wrt, gradof):
        super(ForwardGrad, self).__init__(func,
                                          wrt,
                                          gradof,
                                          mode='forward',
                                          backend='tapenade')
        self.c_func = self.get_c_forward_diff()

    def get_c_forward_diff(self):
        self.grad_source = pyb11_setup_header + self.grad_source
        hash_fn = get_md5(self.grad_source)
        modname = f'm_{hash_fn}'

        pyb_all, c_all, _, _ = get_diff_signature(self.func,
                                                  self.active,
                                                  mode='forward')
        pyb_call = ", ".join(pyb_all)
        c_call = ", ".join(c_all)

        pyb_temp = Template(pyb11_wrap_template)
        pyb_bind = pyb_temp.render(name=self.name,
                                   hash_fn=hash_fn,
                                   FUNC_SUFFIX_F=FUNC_SUFFIX_F,
                                   pyb_call=pyb_call,
                                   c_call=c_call)

        self.grad_all_source = self.grad_source + pyb_bind
        mod = Cmodule(self.grad_all_source, hash_fn)
        module = mod.load()
        return getattr(module, modname)

    def _get_len_wrt_args(self, args):
        len_wrt_args = []
        for i in range(len(self.args)):
            if self.args[i] in self.wrt:
                len_wrt_args.append(len(args[i]))
        return len_wrt_args

    def _add_wrt_args_fwd(self, args, wrt_var, len_wrt_arg):
        final_args = []
        gradof_args = []
        wrt_arg = None
        is_grad_var = []
        for i in range(len(args)):
            final_args.append(args[i])
            is_grad_var.append(0)
            if self.args[i] in self.active:
                temp = np.zeros((len(args[i]), len_wrt_arg))
                final_args.append(temp)
                is_grad_var.append(1)
                if self.args[i] == wrt_var:
                    wrt_arg = temp
                if self.args[i] in self.gradof:
                    gradof_args.append(temp)
        return final_args, gradof_args, wrt_arg, is_grad_var

    def _call_multi_fwd(self, final_args, wrt_arg, is_grad_var):
        for i in range(len(wrt_arg)):
            wrt_arg[i][i] = 1.0

        for i in range(len(wrt_arg)):
            temp_args = []
            for j, arg in enumerate(final_args):
                if not is_grad_var[j]:
                    temp_args.append(arg)
                else:
                    temp_args.append(arg[:, i])
            self.c_func(*temp_args)

    @profile
    def __call__(self, *args):
        c_args = [self._massage_arg(x) for x in args]

        len_g_args = self._get_len_wrt_args(c_args)

        ans = []
        for i, grad_var in enumerate(self.wrt):
            final_args, gradof_args, wrt_arg, is_grad_var = self._add_wrt_args_fwd(c_args, grad_var, len_g_args[i])

            self._call_multi_fwd(final_args, wrt_arg, is_grad_var)
            ans.append(gradof_args)
        return ans


class ElementwiseGrad(GradBase):
    def __init__(self, func, wrt, gradof, backend='c'):
        super(ElementwiseGrad, self).__init__(func,
                                              wrt,
                                              gradof,
                                              mode='forward',
                                              backend=backend)
        self._config = get_config()
        self.c_func = self._generate()

    def _generate(self):
        if self.backend == 'c':
            return self._c_gen()
        elif self.backend == 'cuda':
            return self._cuda_gen()

    def correct_initialization(self):
        for var in self.gradof:
            grad_var = var + VAR_SUFFIX_F
            prev = f"*{grad_var} = 0"
            after = f"{grad_var}[i] = 0"
            self.grad_all_source = self.grad_all_source.replace(prev, after)

    def _c_gen(self):
        pyb_args, pyb_c_args, py_args, c_args = get_diff_signature(
            self.func, self.active)

        c_templt = Template(c_backend_template)
        c_code = c_templt.render(c_kernel_defn=self.grad_source,
                                 fn_name='{fname}{suff}'.format(
                                     fname=self.name, suff=FUNC_SUFFIX_F),
                                 fn_args=", ".join(c_args[1:]),
                                 fn_call=", ".join(py_args[1:]),
                                 openmp=self._config.use_openmp)

        self.grad_all_source = pyb11_setup_header + c_code

        hash_fn = get_md5(self.grad_all_source)
        modname = f'm_{hash_fn}'

        pyb_templt = Template(pyb11_wrap_template)
        elwise_name = 'elementwise_' + self.name
        size = "{}.request().size".format(py_args[1])
        pyb_code = pyb_templt.render(name=elwise_name,
                                     hash_fn=hash_fn,
                                     FUNC_SUFFIX_F=FUNC_SUFFIX_F,
                                     pyb_call=", ".join(pyb_args[1:]),
                                     c_call=", ".join([size] + pyb_c_args[1:]))
        self.grad_all_source += pyb_code
        mod = Cmodule(self.grad_all_source, hash_fn)
        module = mod.load()
        return getattr(module, modname)

    def _cuda_gen(self):
        from .cuda import set_context
        set_context()
        from pycuda.elementwise import ElementwiseKernel
        from pycuda._cluda import CLUDA_PREAMBLE

        _, _, py_args, c_args = get_diff_signature(self.func, self.active)

        self.grad_source = self.convert_to_device_code(self.grad_source)
        expr = '{func}({args})'.format(func=self.name + FUNC_SUFFIX_F,
                                       args=", ".join(py_args))

        arguments = convert_to_float_if_needed(", ".join(c_args[1:]))
        preamble = convert_to_float_if_needed(self.grad_source)

        cluda_preamble = Template(text=CLUDA_PREAMBLE).render(
            double_support=True)
        self.grad_all_source = cluda_preamble + preamble
        self.correct_initialization()
        knl = ElementwiseKernel(name=self.name,
                                arguments=arguments,
                                operation=expr,
                                preamble="\n".join([cluda_preamble, preamble]))
        return knl

    def convert_to_device_code(self, code):
        code = re.sub(r'\bvoid\b', 'WITHIN_KERNEL void', code)
        code = re.sub(r'\bfloat\b', 'GLOBAL_MEM float ', code)
        code = re.sub(r'\bdouble\b', 'GLOBAL_MEM double ', code)
        return code


class ReverseGrad(GradBase):
    def __init__(self, func, wrt, gradof, backend='tapenade'):
        super().__init__(func, wrt, gradof, mode='reverse', backend=backend)
        self.c_func = self._c_reverse_diff()

    def _c_reverse_diff(self):
        self.grad_source = pyb11_setup_header_rev + self.grad_source
        hash_fn = get_md5(self.grad_source)
        modname = f'm_{hash_fn}'

        pyb_all, c_all, _, _ = get_diff_signature(self.func,
                                                  self.active,
                                                  mode=self.mode)
        pyb_call = ", ".join(pyb_all)
        c_call = ", ".join(c_all)

        pyb_temp = Template(pyb11_wrap_template_rev)
        pyb_bind = pyb_temp.render(name=self.name,
                                   modname=modname,
                                   FUNC_SUFFIX_F=FUNC_SUFFIX_R,
                                   pyb_call=pyb_call,
                                   c_call=c_call)

        self.grad_all_source = self.grad_source + pyb_bind
        tpnd_obj_dir = get_tpnd_obj_dir()

        if self.req_recomp_tpnd():
            compile_tapenade_source()
        extra_inc_dir = [pybind11.get_include(), tpnd_obj_dir]
        extra_link_args = [os.path.join(tpnd_obj_dir, 'adBuffer.o'),
                           os.path.join(tpnd_obj_dir, 'adStack.o')]
        mod = Cmodule(self.grad_all_source, hash_fn,
                      extra_inc_dir=extra_inc_dir,
                      extra_link_args=extra_link_args)
        module = mod.load()
        return getattr(module, modname)

    def req_recomp_tpnd(self):
        tpnd_obj_dir = get_tpnd_obj_dir()
        cond1 = not os.path.exists(os.path.join(tpnd_obj_dir, 'adBuffer.o'))
        cond2 = not os.path.exists(os.path.join(tpnd_obj_dir, 'adStack.o'))
        return cond1 or cond2

    def _get_len_gradof_args(self, args):
        len_gradof_args = []
        for i in range(len(self.args)):
            if self.args[i] in self.gradof:
                len_gradof_args.append(len(args[i]))
        return len_gradof_args

    def _add_grad_args_rev(self, args, gradof_var, len_gradof_arg):
        final_args = []
        wrt_args = []
        gradof_arg = None
        is_grad_var = []
        for i in range(len(args)):
            final_args.append(args[i])
            is_grad_var.append(0)
            if self.args[i] in self.active:
                temp = np.zeros((len_gradof_arg, len(args[i])))
                final_args.append(temp)
                is_grad_var.append(1)
                if self.args[i] == gradof_var:
                    gradof_arg = temp
                if self.args[i] in self.wrt:
                    wrt_args.append(temp)
        return final_args, wrt_args, gradof_arg, is_grad_var

    def _call_multi_rev(self, final_args, gradof_arg, is_grad_var):
        for i in range(len(gradof_arg)):
            gradof_arg[i][i] = 1.0

        for i in range(len(gradof_arg)):
            temp_args = []
            for j, arg in enumerate(final_args):
                if not is_grad_var[j]:
                    temp_args.append(arg)
                else:
                    temp_args.append(arg[i])
            self.c_func(*temp_args)

    @profile
    def __call__(self, *args):
        c_args = [self._massage_arg(x) for x in args]
        self.c_func(*c_args)
        return
        len_g_args = self._get_len_gradof_args(c_args)

        ans = []
        for i, grad_var in enumerate(self.gradof):
            final_args, wrt_args, gradof_arg, is_grad_var = self._add_grad_args_rev(c_args, grad_var, len_g_args[i])

            self._call_multi_rev(final_args, gradof_arg, is_grad_var)
            ans.append(wrt_args)
        return ans