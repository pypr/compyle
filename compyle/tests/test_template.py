from textwrap import dedent

import numpy as np

from ..array import wrap
from ..types import annotate
from ..template import Template
from ..parallel import Elementwise


class SimpleTemplate(Template):
    def __init__(self, name, cond=False):
        super(SimpleTemplate, self).__init__(name=name)
        self.cond = cond

    def template(self, x, y):
        '''Docstring text'''
        '''
        % for i in range(5):
        print(${i})
        % endfor
        % if obj.cond:
        return 'hello'
        % else:
        return 'bye'
        % endif
        '''


class Dummy(Template):
    def template(self):
        '''Docs'''
        '''
        print(123)
        '''


class ParallelExample(Template):
    @annotate(i='int', x='doublep', y='doublep')
    def template(self, i, x, y):
        '''
        y[i] = x[i]*2.0
        '''


class ExtraArgs(Template):
    def extra_args(self):
        return ['x'], {'x': 'int'}

    def template(self):
        '''
        return x + 1
        '''


def test_simple_template():
    # Given
    t = SimpleTemplate(name='simple')

    # When
    simple = t.function
    x = simple(1, 2)

    # Then
    assert x == 'bye'

    # Given
    t = SimpleTemplate(name='simple', cond=True)

    # When
    simple = t.function
    x = simple(1, 2)

    # Then
    assert x == 'hello'


def test_that_source_code_is_available():
    # Given/When
    dummy = Dummy('dummy').function

    # Then
    expect = dedent('''\
    def dummy():
        """Docs
        """
        print(123)
    ''')
    assert dummy.source.strip() == expect.strip()


def test_template_usable_in_code_generation():
    # Given
    twice = ParallelExample('twice').function

    x = np.linspace(0, 1, 10)
    y = np.zeros_like(x)
    x, y = wrap(x, y)

    # When
    e = Elementwise(twice)
    e(x, y)

    # Then
    y.pull()
    np.testing.assert_almost_equal(y, 2.0*x.data)


def test_template_with_extra_args():
    # Given
    extra = ExtraArgs('extra').function

    # When
    result = extra(1)

    # Then
    assert result == 2
    assert extra.__annotations__ == {'x': 'int'}


def test_template_inject_works():
    # Given
    def f(x):
        '''Docs
        '''
        for i in range(5):
            x += i
        return x + 1

    # When
    t = Template('t')
    result = t.inject(f, indent=1)

    # Then
    lines = ['for i in range(5):\n', '    x += i\n', 'return x + 1\n']
    expect = ''.join([' '*4 + x for x in lines])
    assert result == expect

    # When
    result = t.inject(f, indent=2)

    # Then
    lines = ['for i in range(5):\n', '    x += i\n', 'return x + 1\n']
    expect = ''.join([' '*8 + x for x in lines])
    assert result == expect

    # When
    result = t.inject(f, indent=0)

    # Then
    lines = ['for i in range(5):\n', '    x += i\n', 'return x + 1\n']
    expect = ''.join(lines)
    assert result == expect
