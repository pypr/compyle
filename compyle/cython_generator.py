"""A simple code generator that generates high-performance Cython code
from equivalent Python code.

Note that this is not a general purpose code generator but one highly tailored
for use in PySPH for general use cases, Cython itself does a terrific job.
"""

from __future__ import absolute_import

import ast
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import inspect
import logging
from textwrap import dedent
import types

from mako.template import Template

from .types import KnownType, Undefined, get_declare_info
from .config import get_config
from .ast_utils import get_assigned, has_return
from .utils import getsourcelines

logger = logging.getLogger(__name__)


def get_parallel_range(start, stop=None, step=1, **kwargs):
    config = get_config()
    if stop is None:
        stop = start
        start = 0

    args = "{start}, {stop}, {step}"
    if config.use_openmp:
        schedule = config.omp_schedule[0]
        chunksize = config.omp_schedule[1]
        if 'schedule' in kwargs:
            schedule = kwargs.pop('schedule')
        if 'chunksize' in kwargs:
            chunksize = kwargs.pop('chunksize')

        if schedule is not None:
            args = args + ", schedule='{schedule}'"

        if chunksize is not None:
            args = args + ", chunksize={chunksize}"

        for k, v in kwargs.items():
            args = args + ", %s=%r" % (k, v)

        args = args.format(start=start, stop=stop, step=step,
                           schedule=schedule, chunksize=chunksize)
        return "prange({})".format(args)

    else:
        args = args.format(start=start, stop=stop, step=step)
        return "range({})".format(args)


class CythonClassHelper(object):
    def __init__(self, name='', public_vars=None, methods=None):
        self.name = name
        self.public_vars = public_vars
        self.methods = methods if methods is not None else []

    def generate(self):
        template = dedent("""
cdef class ${class_name}:
    %for name, type in public_vars.items():
    cdef public ${type} ${name}
    %endfor
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

%for defn, body in methods:
    ${defn}
    %for line in body.splitlines():
${line}
    %endfor

%endfor
        """)
        t = Template(text=template)
        return t.render(class_name=self.name,
                        public_vars=self.public_vars,
                        methods=self.methods)


def condense_multiline_calls(lines:list):
    """
    This function takes a list of lines and condenses multiline calls into
    single lines, while preserving indentation, and moving any inline comments
    to the end of the line.

    Parameters
    ----------
    lines : list of str
        The lines of code to condense.
    
    Returns
    -------
    condensed_lines : list of str
        The condensed lines of code.
    """
    def _line_is_complete(line:str):
        """
        This function checks if a line is complete, i.e. if it has the same
        number of opening and closing parentheses, and if it does not end with
        a backslash.

        Parameters
        ----------
        line : str
            The line to check.
        
        Returns
        -------
        is_complete : bool
            True if the line is complete, False otherwise.
        """
        tmp_line = line.strip()
        cond1 = tmp_line.count('(') == tmp_line.count(')')
        cond2 = not tmp_line.endswith('\\')
        return (cond1 and cond2)
    
    lines = [line.rstrip() for line in lines]
    condensed_lines = []
    line_count = 0
    for idx, line in enumerate(lines):
        if line.endswith('\n'):
            # Remove the newline character
            line = line[:-1]

        if line_count > idx:
            # This line has already been condensed
            continue
        
        cond = (line.count('(') == 0) and (line.count(')') >= 1)
        if _line_is_complete(line) or cond:
            # This line is complete, or it only has closing parentheses
            condensed_lines.append(line)
            line_count += 1
            continue
        else:
            # This line is incomplete, so it needs to be condensed
            current_line = line.rstrip()
            if current_line.endswith('\\'):
                # Remove the backslash
                current_line = current_line[:-1]
            
            if idx + 1 >= len(lines):
                # This line is incomplete, but there are no more lines
                raise SyntaxError(f"{idx}: {line}\n '(' was never closed")
            
            for next_line in lines[idx+1:]:
                # Iterate over the next lines, adding them to the current line
                # until the line is complete
                tmp = current_line.split('#')
                code_part = tmp[0].rstrip()
                comment_part = None
                
                if code_part.endswith('\\'):
                    # Remove the backslash
                    code_part = code_part[:-1]

                if len(tmp) > 1:
                    # There is an inline comment
                    comment_part = '#'.join(tmp[1:])

                if len(next_line.split('#')) >= 2:
                    # There is an inline comment in the next line. Deal with
                    # it by splitting the line into a code part and a comment
                    next_tmp = next_line.split('#')
                    next_line = next_tmp[0]
                    if comment_part is not None:
                        comment_part = comment_part.strip()
                        comment_part = " #".join(
                            next_tmp[1:] + [comment_part]
                        )
                    else:
                        comment_part = " #".join(next_tmp[1:])
                
                if next_line.rstrip().endswith('\\'):
                    # Remove the backslash
                    next_line = next_line.rstrip()[:-1]
                
                # Set the current line to the code part of the current line
                # plus the next line
                current_line = code_part.rstrip() + next_line.strip()
                line_count += 1

                if comment_part is not None:
                    # Add the inline comment to the end of the line
                    current_line += f" #{comment_part.strip()}"

                if _line_is_complete(current_line):
                    # The line is complete, so break out of the loop
                    break

            condensed_lines.append(current_line)
            line_count += 1
    
    # Add '\n' to the end of each line if it does not already have one
    condensed_lines = [
        line if line.endswith('\n') else f"{line}\n"
        for line in condensed_lines
    ]
    return condensed_lines

def get_func_definition(sourcelines):
    """Given a block of source lines for a method or function,
    get the lines for the function block.
    """
    # For now return the line after the first.
    count = 1
    for line in sourcelines:
        if line.rstrip().endswith(':'):
            break
        count += 1
    return sourcelines[:count], sourcelines[count:]


def all_numeric(seq):
    """Return true if all values in given sequence are numeric.
    """
    try:
        types = [int, float, long]
    except NameError:
        types = [int, float]
    return all(type(x) in types for x in seq)


class CodeGenerationError(Exception):
    pass


def parse_declare(code):
    """Given a string with the source for the declare method,
    return the type information.
    """
    m = ast.parse(code)
    call = m.body[0].value
    if call.func.id != 'declare':
        raise CodeGenerationError('Unknown declare statement: %s' % code)
    arg0 = call.args[0]
    if not isinstance(arg0, ast.Str):
        err = 'Type should be a string, given :%r' % arg0.s
        raise CodeGenerationError(err)

    return get_declare_info(arg0.s)


class CythonGenerator(object):
    def __init__(self, known_types=None, python_methods=False):
        """
        Parameters
        -----------

        - known_types: dict: provides default types for known arguments.

        - python_methods: bool: generate python wrapper functions.

             specifies if convenient Python friendly wrappers are to be
             generated in addition to the low-level c wrappers.
        """

        self.code = ''
        self.python_methods = python_methods
        # Methods to not wrap.
        self.ignore_methods = ['_cython_code_']
        self.known_types = known_types if known_types is not None else {}
        self._config = get_config()

    # ### Public protocol #####################################################

    def add_known(self, names):
        '''Just for API compatibility with the translator.
        '''
        pass

    def ctype_to_python(self, type_str):
        """Given a c-style type declaration obtained from the `detect_type`
        method, return a Python friendly type declaration.
        """
        return type_str.replace('*', '[:]')

    def detect_type(self, name, value):
        """Given the variable name and value, detect its type.
        """
        if isinstance(value, KnownType):
            return value.type.replace(
                'GLOBAL_MEM ', ''
            ).replace('LOCAL_MEM ', '')
        if name.startswith(('s_', 'd_')) and name not in ['s_idx', 'd_idx']:
            return 'double*'
        if name in ['s_idx', 'd_idx']:
            return 'long'
        if value is Undefined or isinstance(value, Undefined):
            msg = 'Unknown type, for function argument named: %s' % name
            raise CodeGenerationError(msg)

        if isinstance(value, bool):
            return 'int'
        elif isinstance(value, int):
            return 'long'
        elif isinstance(value, str):
            return 'str'
        elif isinstance(value, float):
            return 'double'
        elif isinstance(value, (list, tuple)):
            if all_numeric(value):
                # We don't deal with integer lists for now.
                return 'double*'
            else:
                return 'list' if isinstance(value, list) else 'tuple'
        else:
            return 'object'

    def get_code(self):
        return self.code

    def parse(self, obj, declarations=None, is_serial=False):
        obj_type = type(obj)
        if isinstance(obj, types.FunctionType):
            self._parse_function(obj, declarations=declarations,
                                 is_serial=is_serial)
        elif hasattr(obj, '__class__'):
            self._parse_instance(obj)
        else:
            raise TypeError('Unsupported type to wrap: %s' % obj_type)

    def get_func_signature(self, func):
        """Given a function that is wrapped, return the Python wrapper
        definition signature and the Python call signature and the C
        wrapper definition and C call signature.

        For example if we had

        def f(x=1, y=[1.0]):
            pass

        If this were passed we would get back:

        (['int x', 'double[:] y'], ['x', '&y[0]']),
        (['int x', 'double* y'], ['x', 'y'])

        """
        sourcelines = getsourcelines(func)[0]
        sourcelines = condense_multiline_calls(sourcelines)
        defn, lines = get_func_definition(sourcelines)
        f_name, returns, args = self._analyze_method(func, lines)
        py_args = []
        py_call = []
        c_args = []
        c_call = []
        for arg, value in args:
            c_type = self.detect_type(arg, value)
            c_args.append('{type} {arg}'.format(type=c_type, arg=arg))
            c_call.append(arg)
            py_type = self.ctype_to_python(c_type)
            py_args.append('{type} {arg}'.format(type=py_type, arg=arg))
            if c_type.endswith('*'):
                py_call.append('&{arg}[0]'.format(arg=arg))
            else:
                py_call.append('{arg}'.format(arg=arg))

        return (py_args, py_call), (c_args, c_call)

    def set_make_python_methods(self, value):
        """Turn on/off the generation of Python methods.
        """
        self.python_methods = value

    # #### Private protocol ###################################################

    def _analyze_method(self, meth, lines):
        """Returns information about the method.

        Specifically it returns the method name, if it has a return value,
        and a list of [(arg_name, value),...].
        """
        name = meth.__name__
        getfullargspec = getattr(
            inspect, 'getfullargspec', inspect.getargspec
        )
        argspec = getfullargspec(meth)
        args = argspec.args
        is_method = False
        if args and args[0] == 'self':
            args = args[1:]
            is_method = True

        if hasattr(argspec, 'annotations'):
            annotations = argspec.annotations
        else:
            annotations = getattr(meth, '__annotations__', {})

        call_args = {}
        # Type annotations always take first precendence even over known
        # types.
        if len(annotations) > 0:
            for arg in args:
                call_args[arg] = annotations.get(arg, Undefined)
            returns = annotations.get('return', False)
        else:
            body = ''.join(lines)
            returns = has_return(dedent(body))

            defaults = argspec.defaults if argspec.defaults is not None else []

            # The call_args dict is filled up with the defaults to detect
            # the appropriate type of the arguments.
            for i in range(1, len(defaults) + 1):
                call_args[args[-i]] = defaults[-i]

            # Set the rest to Undefined
            for i in range(len(args) - len(defaults)):
                call_args[args[i]] = Undefined

            # Make sure any predefined quantities are suitably typed.
            call_args.update(self.known_types)

        new_args = [('self', None)] if is_method else []
        for arg in args:
            value = call_args[arg]
            new_args.append((arg, value))

        return name, returns, new_args

    def _get_c_method_spec(self, name, returns, args):
        """Returns a C definition for the method.
        """
        c_args = []
        if args and args[0][0] == 'self':
            args = args[1:]
            c_args.append('self')

        for arg, value in args:
            c_type = self.detect_type(arg, value)
            c_args.append('{type} {arg}'.format(type=c_type, arg=arg))

        if isinstance(returns, KnownType):
            c_ret = returns.type
        else:
            c_ret = 'double' if returns else 'void'
        c_arg_def = ', '.join(c_args)
        if self._config.use_openmp:
            ignore = ['reduce', 'converged']
            gil = " nogil" if name not in ignore else ""
        else:
            gil = ""
        cdefn = 'cdef inline {ret} {name}({arg_def}){gil}:'.format(
            ret=c_ret, name=name, arg_def=c_arg_def, gil=gil
        )

        return cdefn

    def _get_methods(self, cls):
        methods = []
        for name in dir(cls):
            if name.startswith(('_', 'py_')):
                continue
            meth = getattr(cls, name)
            if callable(meth):
                if name in self.ignore_methods:
                    continue

                c_code, py_code = self._get_method_wrapper(
                    meth, indent=' ' * 8)
                methods.append(c_code)
                if self.python_methods:
                    methods.append(py_code)

        return methods

    def _get_method_body(self, meth, lines, indent=' ' * 8, declarations=None,
                         is_serial=False):
        getfullargspec = getattr(
            inspect, 'getfullargspec', inspect.getargspec
        )
        args = set(getfullargspec(meth).args)
        src = [self._process_body_line(line, is_serial=is_serial)
               for line in lines]
        if declarations:
            cy_decls = []
            for var, decl in declarations.items():
                dtype, name = decl[:-1].split(' ')
                if dtype[0] == 'u':
                    dtype = 'unsigned %s' % dtype[1:]
                modified_decl = '%s %s' % (dtype, name)
                cy_decls.append((var, indent + 'cdef %s\n' % modified_decl))
            src = cy_decls + src
        declared = [] if not declarations else list(declarations.keys())
        for names, defn in src:
            if names:
                declared.extend(x.strip() for x in names.split(','))
        cython_body = ''.join([x[1] for x in src])
        body = ''.join(lines)
        dedented_body = dedent(body)
        symbols = get_assigned(dedented_body)
        undefined = symbols - set(declared) - args
        declare = [indent + 'cdef double %s\n' % x for x in sorted(undefined)]
        code = ''.join(declare) + cython_body
        return code

    def _get_method_wrapper(self, meth, indent=' ' * 8, declarations=None,
                            is_serial=False):
        sourcelines = getsourcelines(meth)[0]
        sourcelines = condense_multiline_calls(sourcelines)
        defn, lines = get_func_definition(sourcelines)
        m_name, returns, args = self._analyze_method(meth, lines)
        c_defn = self._get_c_method_spec(m_name, returns, args)
        c_body = self._get_method_body(meth, lines, indent=indent,
                                       declarations=declarations,
                                       is_serial=is_serial)
        self.code = '{defn}\n{body}'.format(defn=c_defn, body=c_body)
        if self.python_methods:
            defn, body = self._get_py_method_spec(m_name, returns, args,
                                                  indent=indent)
        else:
            defn, body = None, None
        return (c_defn, c_body), (defn, body)

    def _get_public_vars(self, obj):
        # For now get it all from the dict.
        data = obj.__dict__
        vars = OrderedDict((name, self.detect_type(name, data[name]))
                           for name in sorted(data.keys()))
        return vars

    def _get_py_method_spec(self, name, returns, args, indent=' ' * 8):
        """Returns a Python friendly definition for the method along with the
        wrapper function.
        """
        py_args = []
        is_method = False
        if args and args[0][0] == 'self':
            is_method = True
            args = args[1:]
            py_args.append('self')

        call_sig = []
        for arg, value in args:
            c_type = self.detect_type(arg, value)
            py_type = self.ctype_to_python(c_type)
            py_args.append('{type} {arg}'.format(type=py_type, arg=arg))
            if c_type.endswith('*'):
                call_sig.append('&{arg}[0]'.format(arg=arg))
            else:
                call_sig.append('{arg}'.format(arg=arg))

        if isinstance(returns, KnownType):
            py_ret = returns.type + ' '
        else:
            py_ret = 'double ' if returns else ''
        py_arg_def = ', '.join(py_args)
        pydefn = 'cpdef {ret}py_{name}({arg_def}):'.format(
            ret=py_ret, name=name, arg_def=py_arg_def
        )
        call = ', '.join(call_sig)
        py_ret = 'return ' if returns else ''
        py_self = 'self.' if is_method else ''
        body = indent + '{ret}{self}{name}({call})\n'.format(
            name=name, call=call, ret=py_ret, self=py_self
        )

        return pydefn, body

    def _handle_declare_statement(self, name, declare):
        def matrix(size):
            if not isinstance(size, tuple):
                size = (size,)
            sz = ''.join(['[%d]' % n for n in size])
            return sz

        # Parse the declare statement.
        kind, _address_space, ctype, shape = parse_declare(declare)
        if kind == 'matrix':
            sz = matrix(shape)
            vars = ['%s%s' % (x.strip(), sz) for x in name.split(',')]
            defn = 'cdef {type} {vars}'.format(
                type=ctype, vars=', '.join(vars)
            )
            return defn
        else:
            defn = 'cdef {type} {name}'.format(type=ctype, name=name)
            return defn

    def _handle_cast_statement(self, name, call):
        # FIXME: This won't handle casting to pointers
        # using something like 'intp'
        call_args = call[5:-1].split(',')
        if len(call_args) <= 2:
            # Maintainś backward compatibility
            expr = call_args[0].strip()
            ctype = call_args[-1].strip()[1:-1]
        else:
            # Deals with cases like `cast(max(abs(x), abs(y)), 'int')
            # where the expression is a function call with multiple arguments
            # separated by commas
            expr = ','.join(call_args[0:-1]).strip()
            ctype = call_args[-1].strip()[1:-1]
        stmt = '%s = <%s> (%s)' % (name, ctype, expr)
        return stmt

    def _handle_atomic_statement_inc(self, name, call, is_serial):
        # FIXME: This won't handle casting to pointers
        # using something like 'intp'
        call_arg = call[11:-1].strip()
        if self._config.use_openmp and not is_serial:
            return['openmp.omp_set_lock(&cy_lock)',
                   '%s = %s' % (name, call_arg),
                   '%s += 1' % call_arg, 'openmp.omp_unset_lock(&cy_lock)']
        else:
            return ['%s = %s' % (name, call_arg), '%s += 1' % call_arg]

    def _handle_atomic_statement_dec(self, name, call, is_serial):
        # FIXME: This won't handle casting to pointers
        # using something like 'intp'
        call_arg = call[11:-1].strip()
        if self._config.use_openmp and not is_serial:
            return['openmp.omp_set_lock(&cy_lock)',
                   '%s = %s' % (name, call_arg),
                   '%s -= 1' % call_arg, 'openmp.omp_unset_lock(&cy_lock)']
        else:
            return ['%s = %s' % (name, call_arg), '%s -= 1' % call_arg]

    def _parse_function(self, obj, declarations=None, is_serial=False):
        c_code, py_code = self._get_method_wrapper(obj, indent=' ' * 4,
                                                   declarations=declarations,
                                                   is_serial=is_serial)
        code = '{defn}\n{body}'.format(defn=c_code[0], body=c_code[1])
        if self.python_methods:
            code += '\n'
            code += '{defn}\n{body}'.format(defn=py_code[0], body=py_code[1])
        self.code = code

    def _parse_instance(self, obj):
        cls = obj.__class__
        name = cls.__name__
        public_vars = self._get_public_vars(obj)
        methods = self._get_methods(cls)
        helper = CythonClassHelper(name=name, public_vars=public_vars,
                                   methods=methods)
        self.code = helper.generate()

    def _process_body_line(self, line, is_serial=False):
        """Returns the name defined and the processed line itself.

        This hack primarily lets us declare variables from Python and inject
        them into Cython code.
        """
        if '=' in line:
            words = [x.strip() for x in line.split('=')]
            if words[1].startswith('declare') and \
               not line.strip().startswith('#'):
                name = words[0]
                declare = words[1]
                defn = self._handle_declare_statement(name, declare)
                indent = line[:line.index(name)]
                return name, indent + defn + '\n'
            elif words[1].startswith('cast') and \
                    not line.strip().startswith('#'):
                name = words[0]
                call = words[1]
                stmt = self._handle_cast_statement(name, call)
                indent = line[:line.index(name)]
                return '', indent + stmt + '\n'
            elif words[1].startswith('atomic_inc') and \
                    not line.strip().startswith('#'):
                name = words[0]
                call = words[1]
                indent = line[:line.index(name)]
                stmts = self._handle_atomic_statement_inc(
                    name, call, is_serial)
                result = ''
                for stmt in stmts:
                    result += indent + stmt + '\n'
                return '', result + '\n'
            elif words[1].startswith('atomic_dec') and \
                    not line.strip().startswith('#'):
                name = words[0]
                call = words[1]
                indent = line[:line.index(name)]
                stmts = self._handle_atomic_statement_dec(
                    name, call, is_serial)
                result = ''
                for stmt in stmts:
                    result += indent + stmt + '\n'
                return '', result + '\n'
            else:
                return '', line
        else:
            return '', line
