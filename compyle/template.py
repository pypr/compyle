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
        arg_string = ', '.join(args)
        body = m.body[0].body
        template = body[-1].value.s
        docstring = body[0].value.s if len(body) == 2 else ''
        name = self.name
        sig = 'def {name}({args}):\n    """{docs}\n    """'.format(
            name=name, args=arg_string, docs=docstring
        )
        src = sig + self.render(template)
        annotations = self.template.__annotations__ or {}
        return src, annotations

    def inject(self, s):
        return s

    def render(self, src):
        t = mako.template.Template(text=src)
        return t.render(obj=self)

    def extra_args(self):
        # XXX
        return None

    def template(self):
        '''Override this to write your mako template.

        `obj` is mapped to self.
        '''
        '''
        ## Mako code here.
        '''
