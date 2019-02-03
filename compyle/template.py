import ast
import inspect
from textwrap import dedent

import mako.template


getfullargspec = getattr(
    inspect, 'getfullargspec', inspect.getargspec
)


class Template(object):
    def __init__(self, name):
        self.name = name
        self._function = None

    @property
    def function(self):
        if self._function is None:
            self._function = self._make_function()
        return self._function

    def _make_function(self):
        src, annotations = self._get_code()
        self._source = src
        namespace = {}
        exec(src, namespace)
        f = namespace[self.name]
        f.__module__ = self.__module__
        try:
            f.__annotations__ = annotations
        except AttributeError:
            f.im_func.__annotations__ = annotations
        f.source = src
        return f

    def _get_code(self):
        m = ast.parse(dedent(inspect.getsource(self.template)))
        argspec = getfullargspec(self.template)
        args = argspec.args
        if args[0] == 'self':
            args = args[1:]
        extra_args, extra_annotations = self.extra_args()
        args += extra_args
        arg_string = ', '.join(args)
        body = m.body[0].body
        template = body[-1].value.s
        docstring = body[0].value.s if len(body) == 2 else ''
        name = self.name
        sig = 'def {name}({args}):\n    """{docs}\n    """'.format(
            name=name, args=arg_string, docs=docstring
        )
        src = sig + self.render(template)
        annotations = getattr(self.template, '__annotations__', {})
        annotations.update(extra_annotations)
        return src, annotations

    def inject(self, func, indent=1):
        '''Returns the source code of the body of `func`.

        The optional `indent` parameter is the indentation to be used for the
        code.  When indent is 1, 4 spaces are added to each line.

        This is meant to be used from the mako template. The idea is that one
        can define the code to be injected as a method and have the body be
        directly injected.
        '''
        lines = inspect.getsourcelines(func)[0]
        src = dedent(''.join(lines))
        m = ast.parse(src)
        # We do this so as to not inject any docstrings.
        body_start_index = 1 if isinstance(m.body[0].body[0], ast.Expr) else 0
        body_start = m.body[0].body[body_start_index].lineno - 1
        body_lines = lines[body_start:]
        first = body_lines[0]
        leading = first.index(first.lstrip())
        diff = indent*4 - leading
        if diff < 0:
            indented_body = [x[-diff:] for x in body_lines]
        else:
            indented_body = [' '*diff + x for x in body_lines]
        return ''.join(indented_body)

    def render(self, src):
        t = mako.template.Template(text=src)
        return t.render(obj=self)

    def extra_args(self):
        '''Override this to provide configurable arguments.

        Return a list of strings which are the arguments and a dictionary with
        the type annotations.

        '''
        return [], {}

    def template(self):
        '''Override this to write your mako template.

        `obj` is mapped to self.
        '''
        '''
        ## Mako code here.
        '''
