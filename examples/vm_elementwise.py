import numpy as np
from math import pi
import time

from compyle.config import get_config
from compyle.api import declare, annotate
from compyle.low_level import cast
from compyle.parallel import Elementwise
from compyle.array import wrap


@annotate(double='xi, yi, xj, yj, gamma', result='doublep')
def point_vortex(xi, yi, xj, yj, gamma, result):
    xij = xi - xj
    yij = yi - yj
    r2ij = xij*xij + yij*yij
    EPS = cast(1.0e-14, "float")
    two = cast(2.0, "float")
    mypi = cast(pi, "float")
    if r2ij < EPS:
        result[0] = 0.0
        result[1] = 0.0
    else:
        tmp = gamma/(two*mypi*r2ij)
        result[0] = -tmp*yij
        result[1] = tmp*xij


@annotate(int='i, nv', gdoublep='x, y, gamma, u, v')
def velocity(i, x, y, gamma, u, v, nv):
    j = declare('int')
    tmp = declare('matrix(2)')
    vx = 0.0
    vy = 0.0
    xi = x[i]
    yi = y[i]
    for j in range(nv):
        point_vortex(xi, yi, x[j], y[j], gamma[j], tmp)
        vx += tmp[0]
        vy += tmp[1]
    u[i] = vx
    v[i] = vy


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
