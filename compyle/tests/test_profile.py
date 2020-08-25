import unittest
import numpy as np

from pytest import importorskip

from ..config import get_config, use_config
from ..array import wrap, zeros, ones
from ..profile import get_profile_info, named_profile, profile, profile_ctx


def axpb():
    a, b = 7, 13
    x = np.random.rand(1000)
    return a * x + b


@profile
def profiled_axpb():
    axpb()


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
    assert profile_info['axpb']['calls'] == 1


def test_profile():
    for i in range(100):
        profiled_axpb()

    profile_info = get_profile_info()
    assert profile_info['profiled_axpb']['calls'] == 100


def test_named_profile():
    importorskip('pyopencl')
    get_config().profile = True
    knl = get_prefix_sum_knl()
    x = ones(100, np.int32, backend='opencl')
    knl(x.dev)

    profile_info = get_profile_info()
    assert profile_info['prefix_sum']['calls'] == 1
