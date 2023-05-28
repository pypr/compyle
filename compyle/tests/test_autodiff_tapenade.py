import pytest
import numpy as np
from math import exp, log

from ..autodiff_tapenade import ForwardGrad, ReverseGrad, ElementwiseGrad
from ..types import annotate
from ..array import wrap, declare



@annotate(n_1='int', doublep='x, y')
def simple_pow(x, y, n_1):
    y[0] = 1
    for i in range(n_1):
        y[0] *= x[0]

@annotate(doublep='x, y', n_1='int')
def ifelse(x, y, n_1):
    if n_1 < 0:
        x[0] = 1 / x[0]
        n_1 = -n_1
    y[0] = 1
    for i in range(n_1):
        y[0] *= x[0]


@annotate(doublep='ip, W, loss', double='b, label', n_1='int')
def log_reg(ip, W, b, loss, label, n_1):
    h = declare('double')
    pred = declare('double')
    h = 0
    for i in range(n_1):
        h += ip[i] * W[i]
    h += b
    pred = 1 / (1 + exp(-h))
    los_prob = pred * label + (1 - pred) * (1 - label)
    loss[0] = -log(los_prob)


def grad_log_reg(ip, W, label):
    h = np.dot(ip, W)
    pred = 1 / (1 + exp(-h))
    return ip * (pred - label)


def test_simple_pow():
    grad_pow = ForwardGrad(simple_pow, ['x'], ['y'])

    x = np.array([2])
    y = np.empty([1])
    [[yfd]] = grad_pow(x, y, 5)
    assert yfd == 80

    grad_pow = ReverseGrad(simple_pow, ['x'], ['y'])

    x = np.array([2])
    y = np.empty(1)
    xd = np.zeros(1)
    yd = np.array([1])

    grad_pow(x, xd, y, yd, 5)
    assert xd[0] == 80


def test_if_else():
    grad_pow = ForwardGrad(ifelse, ['x'], ['y'])

    x = np.array([2])
    y = np.empty(1)
   
    [[yfd]] = grad_pow(x, y, -5)
    assert yfd == (-5 * (2 ** -6))

    grad_pow = ReverseGrad(ifelse, ['x'], ['y'])

    x = np.array([2])
    y = np.empty(1)
    xd = np.zeros(1)
    yd = np.array([1])

    grad_pow(x, xd, y, yd, -5)
    assert xd[0] == (-5 * (2 ** -6))


def t_log_red():
    g_log_reg = ReverseGrad(log_reg, ['W'], ['loss'])
    n = 5
    ip = np.linspace(0, 1, n)
    W = np.random.randn(n)
    b = 1
    loss = np.array([0])
    label = 1
    
    loss_d = np.ones(1)
    W_d = np.empty_like(W)
    
    g_log_reg(ip, W, W_d, b, loss, loss_d, label, n)
    
    assert np.allclose(W_d, grad_log_reg(ip, W, label))