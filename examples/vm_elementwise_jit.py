import numpy as np
from math import pi
import time

from compyle.config import get_config
from compyle.api import declare, annotate
from compyle.parallel import Elementwise
from compyle.array import wrap


@annotate
def point_vortex(xi, yi, xj, yj, gamma, result):
    xij = xi - xj
    yij = yi - yj
    r2ij = xij*xij + yij*yij
    if r2ij < 1e-14:
        result[0] = 0.0
        result[1] = 0.0
    else:
        tmp = gamma/(2.0*pi*r2ij)
        result[0] = -tmp*yij
        result[1] = tmp*xij


@annotate
def velocity(i, x, y, gamma, u, v, nv):
    tmp = declare('matrix(2)')
    xi = x[i]
    yi = y[i]
    u[i] = 0.0
    v[i] = 0.0
    for j in range(nv):
        point_vortex(xi, yi, x[j], y[j], gamma[j], tmp)
        u[i] += tmp[0]
        v[i] += tmp[1]


def make_vortices(nv, backend):
    x = np.linspace(-1, 1, nv)
    y = x.copy()
    gamma = np.ones(nv)
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    x, y, gamma, u, v = wrap(x, y, gamma, u, v, backend=backend)
    return x, y, gamma, u, v, nv


def run(nv, backend):
    e = Elementwise(velocity, backend=backend)
    args = make_vortices(nv, backend)
    t1 = time.time()
    e(*args)
    print(time.time() - t1)
    u = args[-3]
    u.pull()
    return e, args


if __name__ == '__main__':
    from compyle.utils import ArgumentParser
    p = ArgumentParser()
    p.add_argument('-n', action='store', type=int, dest='n',
                   default=10000, help='Number of particles.')
    o = p.parse_args()
    run(o.n, o.backend)
