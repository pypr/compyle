import inspect
from textwrap import dedent
from unittest import TestCase

from .. import utils


def func(x):
    return x


class TestUtils(TestCase):
    def test_getsource_works_with_normal_function(self):
        # Given/When
        src = utils.getsource(func)

        # Then
        self.assertEqual(src, inspect.getsource(func))

    def test_getsource_works_with_generated_function(self):
        # Given
        src = dedent('''
        def gfunc(x):
            return x
        ''')
        ns = {}
        exec(src, ns)
        gfunc = ns['gfunc']
        gfunc.source = src

        # When
        result = utils.getsource(gfunc)

        # Then
        self.assertEqual(result, src)

    def test_getsourcelines_works_with_normal_function(self):
        # Given/When
        result = utils.getsourcelines(func)

        # Then
        self.assertEqual(result, inspect.getsourcelines(func))

    def test_getsourcelines_works_with_generated_function(self):
        # Given
        src = dedent('''
        def gfunc(x):
            return x
        ''')
        ns = {}
        exec(src, ns)
        gfunc = ns['gfunc']
        gfunc.source = src

        # When
        result = utils.getsourcelines(gfunc)

        # Then
        self.assertEqual(result, (src.splitlines(True), 0))
