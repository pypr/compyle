import unittest
import numpy as np

from pytest import importorskip

from ..config import get_config, use_config
from ..array import wrap, zeros, ones
from ..profile import (
    get_profile_info, named_profile, profile, profile_ctx, ProfileContext
)


def axpb():
    a, b = 7, 13
    x = np.random.rand(1000)
    return a * x + b


class A:
    @profile
    def f(self):
        pass


class B:
    def __init__(self):
        self.name = 'my_name'

    @profile
    def f(self):
        pass

    @profile(name='explicit_name')
    def named(self):
        pass


@profile
def profiled_axpb():
    axpb()


@profile
def nested():
    profiled_axpb()


@named_profile('prefix_sum', backend='opencl')
def get_prefix_sum_knl():
    from ..opencl import get_queue, get_context
    from pyopencl.scan import GenericScanKernel
    ctx = get_context()
    queue = get_queue()
    return GenericScanKernel(ctx, np.int32,
                             arguments="__global int *ary",
                             input_expr="ary[i]",
                             scan_expr="a+b", neutral="0",
                             output_statement="ary[i] = prev_item")


def test_profile_ctx():
    with profile_ctx('axpb'):
        axpb()

    profile_info = get_profile_info()
    assert profile_info[0]['axpb']['calls'] == 1


def test_profile():
    for i in range(100):
        profiled_axpb()

    profile_info = get_profile_info()
    assert profile_info[0]['profiled_axpb']['calls'] == 100


def test_profile_method():
    # Given
    a = A()
    b = B()

    # When
    for i in range(5):
        a.f()
        b.f()
        b.named()

    # Then
    profile_info = get_profile_info()
    assert profile_info[0]['A.f']['calls'] == 5

    # For b.f(), b.name is my_name.
    assert profile_info[0]['my_name']['calls'] == 5

    # profile was given an explicit name for b.named()
    assert profile_info[0]['explicit_name']['calls'] == 5


def test_named_profile():
    importorskip('pyopencl')
    get_config().profile = True
    knl = get_prefix_sum_knl()
    x = ones(100, np.int32, backend='opencl')
    knl(x.dev)

    profile_info = get_profile_info()
    assert profile_info[0]['prefix_sum']['calls'] == 1


def test_nesting_and_context():
    # When
    p = ProfileContext('main')
    nested()
    p.stop()

    # Then
    prof = get_profile_info()
    assert len(prof) == 3
    assert prof[0]['main']['calls'] == 1
    assert prof[1]['nested']['calls'] == 1
    assert prof[2]['profiled_axpb']['calls'] == 1
