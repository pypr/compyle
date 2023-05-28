import optax
import jax.numpy as jnp
import jax
import numpy as np
from optax import softmax_cross_entropy
from mnist import MNIST

import numpy as np
from math import exp, log
import matplotlib.pyplot as plt

from compyle.autodiff_tapenade import ReverseGrad
from compyle.api import annotate, Elementwise, declare, get_config
from compyle.array import empty, ones
from pytools import argmax

np.random.seed(2)
# get_config().use_openmp = True

BATCH_SIZE = 200
from mnist import MNIST

mndata = MNIST('/home/rohit/Documents/Pypr/examples/samples')

images, labels = mndata.load_training()
TRAIN = np.array(images, dtype=np.float32) / 255
TRAIN_LABELS = np.array(labels)

imt, lt = mndata.load_testing()
TEST = np.array(imt, dtype=np.float32) / 255
TEST_FLAT = TEST.flatten()
TEST_LABELS = np.array(lt, dtype=np.int32)

def initialise(n_0, n_1, n_2):
    w_01 = np.random.random(n_0 * n_1).astype(np.float32) * 0.01
    w_12 = np.random.random(n_1 * n_2).astype(np.float32) * 0.01
    b_1 = np.random.random(n_1).astype(np.float32) * 0.01
    b_2 = np.random.random(n_2).astype(np.float32) * 0.01
    v_1 = np.zeros(n_1, dtype=np.float32)
    v_2 = np.zeros(n_2, dtype=np.float32)
    
    g_w_01 = np.zeros_like(w_01, dtype=np.float32)
    g_w_12 = np.zeros_like(w_12, dtype=np.float32)
    
    g_b_1 = np.zeros_like(b_1, dtype=np.float32)
    g_b_2 = np.zeros_like(b_2, dtype=np.float32)
    
    return w_01, b_1, v_1, w_12, b_2, v_2, g_w_01, g_w_12, g_b_1, g_b_2



n_0 = 784
n_1 = 128
n_2 = 10
alpha = 0.001
n_train = 60000

w_01, b_1, v_1, w_12, b_2, v_2, g_w_01, g_w_12, g_b_1, g_b_2 = initialise(n_0, n_1,n_2)
g_v_1 = np.zeros_like(v_1)
g_v_2 = np.zeros_like(v_2)
g_ip = np.zeros(n_0)
loss = np.zeros(1, dtype=np.float32)
loss_b = np.ones(1, dtype=np.float32)

initial_params = {}
initial_params['w01'] = w_01
initial_params['w12'] = w_12
initial_params['b1'] = b_1
initial_params['b2'] = b_2

grads = {}
grads['w01'] = g_w_01
grads['w12'] = g_w_12
grads['b1'] = g_b_1
grads['b2'] = g_b_2

###############################################################################
###############################################################################
###############################################################################

@annotate(i='int', v='floatp')
def reset(i, v):
    v[i] = 0.0
reset_all = Elementwise(reset, backend='c')

@annotate(int='i, batch_size', floatp='g, gsum')
def addgrad(i, g, gsum):
    gsum[i] += g[i]
addgrad_elwise = Elementwise(addgrad, backend='c')

def reset_grads(grads):
    for key in grads:
        reset_all(grads[key])

def add_grads(grads, gradsums):
    for key in grads:
        addgrad_elwise(grads[key], gradsums[key])
def avg_grads(gradsums, batch_size):
    for key in gradsums:
        gradsums[key] /= batch_size

@annotate(int='n_0, n_1, n_2, n_3, expected', floatp='input, w_01, b_1, v_1, w_12, b_2, v_2, loss')
def fwd_pass_final(n_0, input, w_01, n_1, b_1, v_1, w_12, n_2, b_2, v_2, loss, expected):
    i, j = declare('int')
    
    for i in range(n_1):
        v_1[i] = b_1[i]
        for j in range(n_0):
            v_1[i] += w_01[i * n_0 + j] * input[j]
    
    for i in range(n_1):
        if v_1[i] < 0:
            v_1[i] = 0
    
    for i in range(n_2):
        v_2[i] = b_2[i]
        for j in range(n_1):
            v_2[i] += w_12[i * n_1 + j] * v_1[j]

    den = declare('float')
    den = 0
    
    for i in range(n_2):
        v_2[i] = exp(v_2[i])
        den += v_2[i]
    for i in range(n_2):
        v_2[i] = v_2[i] / den
        
    loss[0] = -log(v_2[expected])

grad_forward = ReverseGrad(fwd_pass_final,
                           ['w_01', 'b_1', 'w_12', 'b_2'], ['loss'])

def fit(optimizer, params, grads, n_train, ip_ar, op_ar):
    opt_state = optimizer.init(params)
    
    gradsums = {}
    gradsums['w01'] = np.zeros_like(w_01)
    gradsums['w12'] = np.zeros_like(w_12)
    gradsums['b1'] = np.zeros_like(b_1)
    gradsums['b2'] = np.zeros_like(b_2)

    losssum = 0.0
    
    
    def step(opt_state, params, grads):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state,  params
    
    for _ in range(1):
        reset_grads(grads)
        reset_grads(gradsums)
        for i in range(n_train):
            loss_b[0] = 1.0
            grad_forward(n_0, ip_ar[i, :], params['w01'], grads['w01'], n_1, params['b1'], grads['b1'], v_1, g_v_1,
                        params['w12'], grads['w12'], n_2, params['b2'], grads['b2'], v_2, g_v_2,
                        loss, loss_b, op_ar[i])
            add_grads(grads, gradsums)
            losssum += loss[0]
            reset_grads(grads)

            if i!= 0 and i % BATCH_SIZE == 0:
                print(i, losssum / BATCH_SIZE)
                avg_grads(gradsums, BATCH_SIZE)
                opt_state, params = step(opt_state, params, gradsums)
                reset_grads(gradsums)
                losssum = 0.0

    return params


def net(params, x):
    # l1
    x = jnp.dot(x, params['w01']) + params['b1']
    x = jax.nn.relu(x)
    #l2
    x = jnp.dot(x, params['w12']) + params['b2']
    x = jax.nn.softmax(x)
    return x
    
def test(params, test_images, test_labels):
    correct = 0
    for i, (batch, labels) in enumerate(zip(TEST, TEST_LABELS)):
        y_hat = net(params, batch)
        if argmax(y_hat) == labels:
            correct += 1

    return correct / len(TEST)
###############################################################################
###############################################################################
###############################################################################

optimizer = optax.adam(learning_rate=1e-3)
params = fit(optimizer, initial_params, grads, n_train, TRAIN, TRAIN_LABELS)

params['w01'] = params['w01'].reshape(n_1, n_0).T
params['w12'] = params['w12'].reshape(n_2, n_1).T

op = test(params, TEST, TEST_LABELS)
print("Accuracy: ", op)