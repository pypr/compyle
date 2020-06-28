import numpy as np
from math import pi
import time

from compyle.config import get_config
from compyle.api import declare, annotate
from compyle.parallel import Elementwise
from compyle.array import get_backend, wrap
from compyle.low_level import cast

import compyle.array as carr


def bc(x, y):
    return np.sin(np.pi * (x + y))


@annotate
def laplace_step(i, u, res, err, nx, ny, dx2, dy2, dnr_inv):
    xid = cast(i % nx, "int")
    yid = cast(i / nx, "int")

    if xid == 0 or xid == nx - 1 or yid == 0 or yid == ny - 1:
        return

    res[i] = ((u[i - 1] + u[i + 1]) * dx2 +
              (u[i - nx] + u[i + nx]) * dy2) * dnr_inv

    diff = res[i] - u[i]

    err[i] = diff * diff


class Grid(object):
    def __init__(self, nx=10, ny=10, xmin=0., xmax=1.,
                 ymin=0., ymax=1., bc=lambda x: 0, backend=None):
        self.backend = get_backend(backend)
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.nx, self.ny = nx, ny
        self.dx = (xmax - xmin) / (nx - 1)
        self.dy = (ymax - ymin) / (ny - 1)
        self.x = np.arange(self.xmin, self.xmax + self.dx * 0.5, self.dx)
        self.y = np.arange(self.ymin, self.ymax + self.dy * 0.5, self.dy)
        self.bc = bc
        self.setup()

    def setup(self):
        u_host = np.zeros((self.nx, self.ny)).astype(np.float32)

        u_host[0, :] = self.bc(self.xmin, self.y)
        u_host[-1, :] = self.bc(self.xmax, self.y)
        u_host[:, 0] = self.bc(self.x, self.ymin)
        u_host[:, -1] = self.bc(self.x, self.ymax)

        self.u = wrap(u_host.flatten(), backend=self.backend)
        self.err = carr.zeros_like(self.u)

    def get(self):
        u_host = self.u.get()
        return np.resize(u_host, (self.nx, self.ny))

    def compute_err(self):
        return np.sqrt(carr.dot(self.err, self.err))

    def plot(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.get())
        plt.show()


class LaplaceSolver(object):
    def __init__(self, grid, backend=None):
        self.grid = grid
        self.backend = get_backend(backend)
        self.step_method = Elementwise(laplace_step, backend=self.backend)
        self.res = self.grid.u.copy()

    def solve(self, max_iter=None, eps=1.0e-8):
        err = np.inf

        g = self.grid

        dx2 = g.dx ** 2
        dy2 = g.dy ** 2
        dnr_inv = 0.5 / (dx2 + dy2)

        count = 0

        while err > eps:
            if max_iter and count >= max_iter:
                return err, count
            self.step_method(g.u, self.res, g.err, g.nx, g.ny,
                             dx2, dy2, dnr_inv)
            err = g.compute_err()

            tmp = g.u
            g.u = self.res
            self.res = tmp

            count += 1

        return err, count


if __name__ == '__main__':
    from compyle.utils import ArgumentParser
    p = ArgumentParser()
    p.add_argument('--nx', action='store', type=int, dest='nx',
                   default=100, help='Number of grid points in x.')
    p.add_argument('--ny', action='store', type=int, dest='ny',
                   default=100, help='Number of grid points in y.')
    p.add_argument(
        '--show', action='store_true', dest='show',
        default=False, help='Show plot at the end of simulation'
    )
    o = p.parse_args()

    grid = Grid(nx=o.nx, ny=o.ny, bc=bc, backend=o.backend)

    solver = LaplaceSolver(grid, backend=o.backend)

    start = time.time()
    err, count = solver.solve(eps=1e-6)
    end = time.time()

    print("Number of iterations = %s" % count)
    print("Time taken = %g secs" % (end - start))

    if o.show:
        solver.grid.plot()
