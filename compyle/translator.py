'''Simple Python to C converter.

While this is a fresh implementation, it is highly inspired from

https://github.com/mdipierro/ocl

This code does not use meta and uses the standard ast visitor, it is also
tested and modified to be suitable for use with PySPH.

'''

from __future__ import absolute_import

import ast
import re
import sys
from textwrap import dedent, wrap
import types

import numpy as np
from mako.template import Template

from .config import get_config
from .types import get_declare_info
from .cython_generator import (
    CodeGenerationError, KnownType, Undefined, all_numeric
)
from .utils import getsource
from . import transpiler as tp

PY_VER = sys.version_info.major


def detect_type(name, value):
    if isinstance(value, KnownType):
        return value.type
    if name.startswith(('s_', 'd_')) and name not in ['s_idx', 'd_idx']:
        return 'double*'
    if name in ['s_idx', 'd_idx']:
        return 'long'
    if value is Undefined or isinstance(value, Undefined):
        raise CodeGenerationError('Unknown type, for %s' % name)

    if isinstance(value, bool):
        return 'int'
    elif isinstance(value, int):
        return 'long'
    elif isinstance(value, float):
        return 'double'
    elif isinstance(value, (list, tuple)):
        if all_numeric(value):
            # We don't deal with integer lists for now.
            return 'double*'
        else:
            raise CodeGenerationError(
                'Unknown type, for %s with value %s' % (name, value)
            )
    else:
        raise CodeGenerationError(
            'Unknown type, for %s with value %s' % (name, value)
        )


def py2c(src, detect_type=detect_type, known_types=None, count_flops=False):
    converter = CConverter(detect_type=detect_type, known_types=known_types,
                           count_flops=count_flops)
    result = converter.convert(src)
    r = converter.get_declarations() + result
    print(r)
    return r


class CStructHelper(object):
    def __init__(self, obj):
        self._use_double = get_config().use_double
        self.parse(obj)

    def _get_public_vars(self):
        data = self.obj.__dict__
        vars = {}
        for name in data:
            if name.startswith('_'):
                continue
            value = data[name]
            if isinstance(value, (int, bool)):
                vars[name] = 'int'
            elif isinstance(value, float):
                vars[name] = 'double'

        return vars

    def parse(self, obj):
        self.name = obj.__class__.__name__
        self.obj = obj
        self.vars = self._get_public_vars()

    def get_array(self):
        f_dtype = np.float64 if self._use_double else np.float32
        types = {'int': np.int32, 'double': f_dtype, 'long': np.int64}
        if len(self.vars) > 0:
            obj = self.obj
            fields = []
            for var in sorted(self.vars):
                fields.append((var, types[self.vars[var]]))
            dtype = np.dtype(fields)
            ary = np.empty(1, dtype)
            for var in self.vars:
                ary[var][0] = getattr(obj, var)
            return ary
        else:
            return None

    def get_code(self):
        template = dedent("""
        typedef struct ${class_name} {
        %for name, type in sorted(vars.items()):
            ${type} ${name};
        %endfor
        } ${class_name};
        """)
        t = Template(text=template)
        return t.render(class_name=self.name, vars=self.vars)


class CConverter(ast.NodeVisitor):
    def __init__(self, detect_type=detect_type, known_types=None,
                 count_flops=False):
        self._declares = {}
        self._known = set((
            'M_E', 'M_LOG2E', 'M_LOG10E', 'M_LN2', 'M_LN10',
            'M_PI', 'M_PI_2', 'M_PI_4', 'M_1_PI', 'M_2_PI',
            'M_2_SQRTPI', 'M_SQRT2', 'M_SQRT1_2',
            'INFINITY', 'NAN', 'HUGE_VALF'
        ))
        self._name_ctx = (ast.Load, ast.Store)
        self._indent = ''
        self._detect_type = detect_type
        self._known_types = known_types if known_types is not None else {}
        self._class_name = ''
        self._src = ''
        self._for_count = 0
        self._added_loop_vars = set()
        self._annotations = {}
        self._declarations = None
        self._ignore_methods = []
        self._count_flops = count_flops
        self._flop_count = [0]
        self._depth = 0
        self._last_for_depth = -1
        self._external_funcs = []
        self._replacements = {
            'True': '1', 'False': '0', 'None': 'NULL',
            True: '1', False: '0', None: 'NULL',
        }
        self.function_address_space = ''
        self.backend = None

    def _body_has_return(self, body):
        return re.search(r'\breturn\b', body) is not None

    def _get_return_type(self, body, node):
        annotations = self._annotations.get(node.name)
        if annotations:
            kt = annotations.get('return')
            return kt.type if kt is not None else 'void'
        else:
            return 'double' if self._body_has_return(body) else 'void'

    def _get_self_type(self):
        return KnownType('%s*' % self._class_name)

    def _get_local_arg(self, arg, type):
        return arg, type

    def _get_function_args(self, node):
        node_args = node.args.args
        if PY_VER == 2:
            args = [x.id for x in node_args]
        else:
            args = [x.arg for x in node_args]
        annotations = self._annotations.get(node.name)
        call_args = {}
        if annotations:
            for arg in args:
                call_args[arg] = annotations.get(arg, Undefined)
        else:
            defaults = [ast.literal_eval(x) for x in node.args.defaults]

            # Fill up the call_args dict with the defaults.
            for i in range(1, len(defaults) + 1):
                call_args[args[-i]] = defaults[-i]

            # Set the rest to Undefined.
            for i in range(len(args) - len(defaults)):
                call_args[args[i]] = Undefined

            call_args.update(self._known_types)

        if len(self._class_name) > 0:
            call_args['self'] = self._get_self_type()

        call_sig = []
        for arg in args:
            value = call_args[arg]
            type = self._detect_type(arg, value)
            if 'LOCAL_MEM' in type:
                arg, type = self._get_local_arg(arg, type)
            call_sig.append('{type} {arg}'.format(type=type, arg=arg))

        if self._count_flops:
            call_sig.append(self._get_flop_counter_arg())

        return ', '.join(call_sig)

    def _get_variable_declaration(self, type_str, names):
        kind, address_space, ctype, shape = get_declare_info(type_str)
        if address_space:
            address_space += ' '

        if kind == 'matrix':
            if not isinstance(shape, tuple):
                shape = (shape,)
            sz = ''.join('[%d]' % x for x in shape)
            vars = ['%s%s' % (x, sz) for x in names]
            return '{address}{type} {vars};'.format(
                address=address_space, type=ctype, vars=', '.join(vars)
            )
        else:
            return '{address}{type} {vars};'.format(
                address=address_space, type=ctype, vars=', '.join(names)
            )

    def _indent_block(self, code):
        lines = code.splitlines()
        pad = ' ' * 4
        return '\n'.join(pad + x for x in lines)

    def _remove_docstring(self, body):
        if body and isinstance(body[0], ast.Expr) and \
                isinstance(body[0].value, ast.Str):
            return body[1:]
        else:
            return body

    def _get_local_info(self, obj):
        return None

    def _get_local_declarations(self):
        return ''

    def _get_flop_counter_arg(self):
        return 'long* cpy_flop_counter'

    def add_known(self, names):
        '''Add a known name that should not be auto-declared.

        This is useful when we are declaring global constants for which we
        should not redeclare the types in the generated code.
        '''
        self._known.update(names)

    def convert(self, src, ignore_methods=None):
        if ignore_methods is not None:
            self._ignore_methods = ignore_methods
        self._src = src.splitlines()
        code = ast.parse(src)
        result = self.visit(code)
        if self._depth:
            raise ValueError("Non-zero function depth at the end of parsing")
        self._ignore_methods = []
        return result

    def error(self, message, node):
        msg = '\nError in code in line %d:\n' % node.lineno
        if self._src:  # pragma: no branch
            if node.lineno > 1:  # pragma no branch
                msg += self._src[node.lineno - 2] + '\n'
            msg += self._src[node.lineno - 1] + '\n'
            msg += ' ' * node.col_offset + '^' + '\n\n'
        msg += message
        raise NotImplementedError(msg)

    def get_declarations(self):
        if len(self._declares) > 0:
            return '\n'.join(
                sorted(self._declares.values())
            ) + '\n'
        else:
            return ''

    def get_struct_from_instance(self, obj):
        helper = CStructHelper(obj)
        return helper.get_code() + '\n'

    def parse(self, obj, declarations=None):
        obj_type = type(obj)
        if isinstance(obj, types.FunctionType):
            code = self.parse_function(obj, declarations=declarations)
        elif hasattr(obj, '__class__'):
            code = self.parse_instance(obj)
        else:
            raise TypeError('Unsupported type to wrap: %s' % obj_type)
        return code

    def parse_instance(self, obj, ignore_methods=None):
        code = self.get_struct_from_instance(obj)
        src = dedent(getsource(obj.__class__))
        ignore_methods = [] if ignore_methods is None else ignore_methods
        for method in dir(obj):
            if not method.startswith(('_', 'py_')) \
               and method not in ignore_methods:
                ann = getattr(getattr(obj, method), '__annotations__', None)
                self._annotations[method] = ann
        code += self.convert(src, ignore_methods)
        self._annotations = {}
        return code

    def parse_function(self, obj, declarations=None):
        src = dedent(getsource(obj))
        fname = obj.__name__
        symbols, implicits, funcs, externs = tp.get_external_symbols_and_calls(
            obj, self.backend
        )
        self._external_funcs = [f.__name__ for f in funcs]
        self._declarations = declarations
        self._annotations[fname] = getattr(obj, '__annotations__', {})
        self._local_decl = self._get_local_info(obj)
        code = self.convert(src)
        self._local_decl = None
        self._annotations = {}
        self._declarations = None
        self._external_funcs = []
        return code

    def render_atomic(self, func, arg):
        raise NotImplementedError(
            "Atomics only supported by CUDA/OpenCL backends")

    def _get_flop_increment(self):
        return self._indent_block(
            'cpy_flop_counter[${offset}] += %i;' %
            self._flop_count[self._depth]
        )

    def _increment_depth(self):
        self._depth += 1
        if len(self._flop_count) <= self._depth:
            self._flop_count.append(0)

    def _get_flop_increment_cumulative(self, final_depth):
        if not self._count_flops:
            return ''
        current_depth = self._depth
        total_increment = 0
        while current_depth >= final_depth:
            total_increment += self._flop_count[current_depth]
            current_depth -= 1
        return 'cpy_flop_counter[${offset}] += %i;' % \
            total_increment + '\n'

    def visit_LShift(self, node):
        return '<<'

    def visit_RShift(self, node):
        return '>>'

    def visit_BitOr(self, node):
        return '|'

    def visit_BitXor(self, node):
        return '^'

    def visit_BitAnd(self, node):
        return '&'

    def visit_Add(self, node):
        return '+'

    def visit_And(self, node):
        return '&&'

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            self.error("Assignments can have only one target.", node)
        left, right = node.targets[0], node.value
        if isinstance(right, ast.Call) and \
           isinstance(right.func, ast.Name) and right.func.id == 'declare':
            if not isinstance(right.args[0], ast.Str):
                self.error("Argument to declare should be a string.", node)
            type = right.args[0].s
            if isinstance(left, ast.Name):
                self._known.add(left.id)
                return self._get_variable_declaration(type, [self.visit(left)])
            elif isinstance(left, ast.Tuple):
                names = [x.id for x in left.elts]
                self._known.update(names)
                return self._get_variable_declaration(type, names)

        return '%s = %s;' % (self.visit(left), self.visit(right))

    def visit_Attribute(self, node):
        return '%s->%s' % (self.visit(node.value), node.attr)

    def visit_AugAssign(self, node):
        self._flop_count[self._depth] += 1
        return '%s %s= %s;' % (self.visit(node.target), self.visit(node.op),
                               self.visit(node.value))

    def visit_BinOp(self, node):
        self._flop_count[self._depth] += 1
        if isinstance(node.op, ast.Pow):
            return 'pow(%s, %s)' % (
                self.visit(node.left), self.visit(node.right)
            )
        else:
            result = tuple(self.visit(x)
                           for x in (node.left, node.op, node.right))
            return '(%s %s %s)' % result

    def visit_BoolOp(self, node):
        self._flop_count[self._depth] += 1
        op = ' %s ' % self.visit(node.op)
        return '(%s)' % (op.join(self.visit(x) for x in node.values))

    def visit_Break(self, node):
        flop_increments = self._get_flop_increment_cumulative(
            self._last_for_depth)
        return flop_increments + 'break;'

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id == 'address':
                return '(&%s)' % self.visit(node.args[0])
            elif 'atomic' in node.func.id:
                return self.render_atomic(node.func.id, node.args[0])
            elif node.func.id == 'cast':
                return '(%s) (%s)' % (node.args[1].s, self.visit(node.args[0]))
            else:
                args = ', '.join(self.visit(x) for x in node.args)
                if self._count_flops and node.func.id in self._external_funcs:
                    args = args + ', cpy_flop_counter + ${offset}'
                return '{func}({args})'.format(
                    func=node.func.id,
                    args=args
                )
        elif isinstance(node.func, ast.Attribute):
            if node.func.value.id in self._known_types:
                name = node.func.value.id
                cls = self._known_types[name].base_type
                args = [name] + [self.visit(x) for x in node.args]
                return '{func}({args})'.format(
                    func='%s_%s' % (cls, node.func.attr),
                    args=', '.join(args)
                )
            elif len(self._class_name) > 0:
                args = ['self'] + [self.visit(x) for x in node.args]
                return '{func}({args})'.format(
                    func='%s_%s' % (self._class_name, node.func.attr),
                    args=', '.join(args)
                )
            else:
                self.error('Unsupported function call', node)
        else:
            self.error('Unsupported function call', node)

    def visit_ClassDef(self, node):
        self._class_name = node.name
        # FIXME: Does not handle base class methods.
        code = [self.visit(x) for x in self._remove_docstring(node.body)]
        self._class_name = ''
        return '\n'.join(code)

    def visit_Compare(self, node):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            self.error('Only simple comparisons are allowed.', node)
        return '(%s %s %s)' % (self.visit(node.left),
                               self.visit(node.ops[0]),
                               self.visit(node.comparators[0]))

    def visit_Continue(self, node):
        flop_increments = self._get_flop_increment_cumulative(
            self._last_for_depth)
        return flop_increments + 'continue;'

    def visit_Div(self, node):
        return '/'

    def visit_Eq(self, node):
        return '=='

    def visit_Expr(self, node):
        return self.visit(node.value) + ';'

    def _check_if_integer(self, s):
        try:
            int(ast.literal_eval(s))
        except ValueError:
            return False
        else:
            return True

    def visit_For(self, node):
        if node.iter.func.id != 'range':
            self.error(
                'Only for var in range syntax supported.', node.iter
            )
        if node.orelse:
            self.error('For/else not supported.', node.orelse[0])
        args = node.iter.args

        # If the stop or step elements are not numbers, then the semantics of a
        # for i in range can be very different from the translated C as in C,
        # one could change the stop or increment at each step.  This is not
        # possible in Python.
        simple = True
        positive_step = True
        int_step = True
        int_stop = True
        self._increment_depth()
        prev_last_for_depth = self._last_for_depth
        self._last_for_depth = self._depth
        if len(args) == 1:
            start, stop, incr = 0, self.visit(args[0]), 1
            int_stop = simple = self._check_if_integer(stop)
        elif len(args) == 2:
            start, stop, incr = self.visit(args[0]), self.visit(args[1]), 1
            int_stop = simple = self._check_if_integer(stop)
        elif len(args) == 3:
            start, stop, incr = [self.visit(x) for x in args]
            int_step = self._check_if_integer(incr)
            int_stop = self._check_if_integer(stop)
            simple = (int_stop and int_step)
            if int_step:
                positive_step = ast.literal_eval(incr) > 0
        else:
            self.error('range should have either 1, 2, or 3 args', node.iter)

        local_scope = False
        if isinstance(node.target, ast.Name):
            if node.target.id not in self._known:
                target_type = 'long '
                self._known.add(node.target.id)
                local_scope = True
            else:
                target_type = ''

        target = self.visit(node.target)

        if simple:
            block = '\n'.join(self._indent_block(self.visit(x))
                              for x in node.body)
            if self._count_flops:
                block += '\n' + self._get_flop_increment()

            comparator = '<' if positive_step else '>'
            r = ('for ({type}{i}={start}; {i}{comp}{stop}; {i}+={incr})'
                 ' {{\n{block}\n}}\n').format(
                     i=target, type=target_type,
                     start=start, stop=stop, incr=incr, comp=comparator,
                     block=block
            )
        else:
            count = self._for_count
            self._for_count += 1
            block = '\n'.join(self._indent_block(self.visit(x))
                              for x in node.body)
            if self._count_flops:
                block += '\n' + self._get_flop_increment()

            r = ''
            if not int_stop:
                stop_var = '__cpy_stop_{count}'.format(count=count)
                type = 'long ' if stop_var not in self._known else ''
                self._known.add(stop_var)
                if count > 0:
                    self._added_loop_vars.add(stop_var)
                r += '{type}{stop_var} = {stop};\n'.format(
                    type=type, stop_var=stop_var, stop=stop
                )
                stop = stop_var
            if int_step:
                comparator = '<' if positive_step else '>'
                r += ('for ({type}{i}={start}; {i}{comp}{stop}; {i}+={incr})'
                      ' {{\n{block}\n}}\n').format(
                          i=target, type=target_type,
                          start=start, stop=stop, incr=incr,
                          comp=comparator, block=block
                )
            else:
                step_var = '__cpy_step_{count}'.format(count=count)
                type = 'long ' if step_var not in self._known else ''
                self._known.add(step_var)
                if count > 0:
                    self._added_loop_vars.add(step_var)
                r += '{type}{step_var} = {incr};\n'.format(
                    type=type, step_var=step_var, incr=incr
                )
                incr = step_var
                r += dedent('''\
                if ({incr} < 0) {{
                    for ({type}{i}={start}; {i}>{stop}; {i}+={incr}) {{
                    {block}
                    }}
                }}
                else {{
                    for ({type}{i}={start}; {i}<{stop}; {i}+={incr}) {{
                    {block}
                    }}
                }}
                ''').format(
                    i=target, type=target_type,
                    start=start, stop=stop, incr=incr,
                    block=block
                )

            self._for_count -= 1
            if count == 0:
                self._known -= self._added_loop_vars
                self._added_loop_vars = set()

        if local_scope:
            self._known.remove(node.target.id)
        self._flop_count[self._depth] = 0
        self._depth -= 1
        self._last_for_depth = prev_last_for_depth
        return r

    def visit_FunctionDef(self, node):
        assert node.args.vararg is None, \
            "Functions with varargs nor supported in line %d." % node.lineno
        assert node.args.kwarg is None, \
            "Functions with kwargs not supported in line %d." % node.lineno

        self._depth = 0
        self._flop_count = [0]
        if self._class_name and (node.name.startswith(('_', 'py_')) or
                                 node.name in self._ignore_methods):
            return ''

        orig_declares = self._declares
        self._declares = {} if not self._declarations else self._declarations
        orig_known = set(self._known)
        if PY_VER == 2:
            self._known.update(x.id for x in node.args.args)
        else:
            self._known.update(x.arg for x in node.args.args)

        args = self._get_function_args(node)
        bodylines = [self._indent_block(self.visit(item))
                     for item in self._remove_docstring(node.body)]

        if self._count_flops and not isinstance(node.body[-1], ast.Return):
            bodylines.append(self._get_flop_increment())

        body = '\n'.join(bodylines)
        local_decl = self._get_local_declarations()
        if len(self._class_name) > 0:
            func_name = self._class_name + '_' + node.name
        else:
            func_name = node.name
        return_type = self._get_return_type(body, node)
        sig = self.function_address_space + '{ret} {name}({args})'.format(
            ret=return_type, name=func_name, args=args
        )

        declares = self._indent_block(self.get_declarations())
        if len(declares) > 0:
            declares += '\n'

        sig = '\n'.join(wrap(
            sig, width=78, subsequent_indent=' ' * 4, break_long_words=False
        ))
        self._known = orig_known
        self._declares = orig_declares
        self._flop_count = [0]
        return sig + '\n{\n' + local_decl + declares + body + '\n}\n'

    def visit_Gt(self, node):
        return '>'

    def visit_GtE(self, node):
        return '>='

    def visit_If(self, node):
        cond = self.visit(node.test)
        self._increment_depth()
        block = '\n'.join(
                self._indent_block(self.visit(x)) for x in node.body
        )
        if self._count_flops and not isinstance(
                node.body[-1], (ast.Continue, ast.Break, ast.Return)):
            block += '\n' + self._get_flop_increment()
        code = 'if ({cond}) {{\n{block}\n}}\n'.format(
            cond=cond,
            block=block
        )
        self._flop_count[self._depth] = 0
        if node.orelse:
            block = '\n'.join(
                self._indent_block(self.visit(x)) for x in node.orelse
            )
            if self._count_flops and not isinstance(
                    node.orelse[-1], (ast.Continue, ast.Break, ast.Return)):
                block += '\n' + self._get_flop_increment()
            code += 'else {{\n{block}\n}}\n'.format(
                block=block
            )
        self._flop_count[self._depth] = 0
        self._depth -= 1
        return code

    def visit_IfExp(self, node):
        code = '{cond} ? {true} : {false}'.format(
            cond=self.visit(node.test),
            true=self.visit(node.body),
            false=self.visit(node.orelse)
        )
        return code

    def visit_Is(self, node):
        return '=='

    def visit_IsNot(self, node):
        return '!='

    def visit_Lt(self, node):
        return '<'

    def visit_LtE(self, node):
        return '<='

    def visit_Mod(self, node):
        return '%'

    def visit_Module(self, node):
        return '\n'.join(
            self.visit(item) for item in node.body
        )

    def visit_Mult(self, node):
        return '*'

    def visit_Name(self, node):
        assert isinstance(node.ctx, self._name_ctx)
        id = node.id
        if id in self._replacements:
            return self._replacements[id]
        if id not in self._declares and id not in self._known:
            self._declares[id] = 'double %s;' % id
        return id

    def visit_NameConstant(self, node):
        value = node.value
        if value in self._replacements:
            return self._replacements[value]
        else:
            return value

    def visit_Not(self, node):
        return '!'

    def visit_NotEq(self, node):
        return '!='

    def visit_Num(self, node):
        return str(node.n)

    def visit_Or(self, node):
        return '||'

    def visit_Pass(self, node):
        return ';'

    def visit_Return(self, node):
        if node.value:
            ret_value = self.visit(node.value)
        flop_increments = self._get_flop_increment_cumulative(0)
        if node.value:
            return flop_increments + 'return %s;' % ret_value
        else:
            return flop_increments + 'return;'

    def visit_Sub(self, node):
        return '-'

    def visit_Str(self, node):
        return r'"%s"' % node.s

    def visit_Subscript(self, node):
        return '%s[%s]' % (
            self.visit(node.value), self.visit(node.slice.value)
        )

    def visit_TryExcept(self, node):
        self.error('Try/except not implemented.', node)

    visit_Try = visit_TryExcept

    def visit_UnaryOp(self, node):
        return '%s%s' % (self.visit(node.op), self.visit(node.operand))

    def visit_USub(self, node):
        return '-'

    def visit_While(self, node):
        self._increment_depth()
        if node.orelse:
            self.error('Does not support while/else clauses.', node.orelse[0])
        block = '\n'.join(
            self._indent_block(self.visit(x)) for x in node.body
        )
        if self._count_flops:
            block += '\n' + self._get_flop_increment()
        code = 'while ({cond}) {{\n{block}\n}}\n'.format(
            cond=self.visit(node.test), block=block
        )
        self._flop_count[self._depth] = 0
        self._depth -= 1
        return code


def ocl_detect_pointer_base_type(name, value):
    if isinstance(value, KnownType):
        if value.base_type:
            return value.base_type
        else:
            # Valid pointer type ends with a '*'
            # Exceptions like `int a[]` are possible but such definitions are
            # not generated by the translator
            pointer_type = value.type.rstrip()
            pointer_type = pointer_type.replace('__global', '')
            pointer_type = pointer_type.replace('GLOBAL_MEM', '')
            pointer_type = pointer_type.replace('__local', '')
            pointer_type = pointer_type.replace('LOCAL_MEM', '')

            if pointer_type[-1] != '*':
                raise Exception("Invalid pointer type: %s" % value.type)

            base_type = pointer_type[:-1].rstrip()
            return base_type
    elif name.startswith(('s_', 'd_')) and name not in ['s_idx', 'd_idx']:
        return 'double'
    else:
        raise NotImplementedError()


def ocl_detect_type(name, value):
    if isinstance(value, KnownType):
        return value.type
    elif name.startswith(('s_', 'd_')) and name not in ['s_idx', 'd_idx']:
        return 'GLOBAL_MEM double*'
    else:
        return detect_type(name, value)


class OpenCLConverter(CConverter):
    def __init__(self, detect_type=ocl_detect_type, known_types=None,
                 count_flops=False):
        super(OpenCLConverter, self).__init__(detect_type, known_types,
                                              count_flops=count_flops)
        self.function_address_space = 'WITHIN_KERNEL '
        self._known.update((
            'LID_0', 'LID_1', 'LID_2',
            'GID_0', 'GID_1', 'GID_2',
            'LDIM_0', 'LDIM_1', 'LDIM_2',
            'GDIM_0', 'GDIM_1', 'GDIM_2'
        ))
        self.backend = 'opencl'

    def _get_self_type(self):
        return KnownType('GLOBAL_MEM %s*' % self._class_name)

    def _get_flop_counter_arg(self):
        return 'GLOBAL_MEM long* cpy_flop_counter'

    def render_atomic(self, func, arg):
        if func == 'atomic_inc':
            return 'atomic_inc(&%s)' % self.visit(arg)
        else:
            raise NotImplementedError("Only atomic_inc supported right now")


class CUDAConverter(OpenCLConverter):
    def __init__(self, detect_type=ocl_detect_type, known_types=None,
                 count_flops=False):
        super(CUDAConverter, self).__init__(detect_type, known_types,
                                            count_flops=count_flops)
        self._local_decl = None
        self.backend = 'cuda'

    def _get_local_arg(self, arg, type):
        return 'size_%s' % arg, 'int'

    def _get_local_info(self, obj):
        fname = obj.__name__
        annotations = self._annotations[fname]
        local_info = {}
        for arg, kt in annotations.items():
            if 'LOCAL_MEM' in kt.type:
                local_info[arg] = kt.base_type
        if local_info:
            return local_info
        return None

    def _get_local_declarations(self):
        local_decl = ''
        if self._local_decl:
            decls = ['extern LOCAL_MEM float shared_buff[];']
            # Reference:
            # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared
            for arg, dtype in self._local_decl.items():
                if len(decls) == 1:
                    local_decl = ('%(dtype)s* %(arg)s = '
                                  '(%(dtype)s*) shared_buff;')
                    local_decl = local_decl % {'dtype': dtype, 'arg': arg}
                    decls.append(local_decl)
                    prev_arg = arg
                else:
                    local_decl = ('%(dtype)s* %(arg)s = (%(dtype)s*) '
                                  '&%(prev_arg)s[size_%(prev_arg)s];')
                    local_decl = local_decl % {'dtype': dtype, 'arg': arg,
                                               'prev_arg': prev_arg}
                    decls.append(local_decl)
                    prev_arg = arg
            local_decl = self._indent_block('\n'.join(decls))
            local_decl += '\n'
        return local_decl

    def render_atomic(self, func, arg):
        if func == 'atomic_inc':
            return 'atomicAdd(&%s, 1)' % self.visit(arg)
        else:
            raise NotImplementedError("Only atomic_inc supported right now")
