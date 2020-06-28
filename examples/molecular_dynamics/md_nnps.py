import numpy as np
from math import pi
import time

from compyle.config import get_config
from compyle.api import declare, annotate
from compyle.parallel import Elementwise, Reduction
from compyle.array import get_backend, wrap
import compyle.array as carr

from nnps import NNPSCountingSort, NNPSRadixSort
from md_simple import integrate_step1, integrate_step2, \
    boundary_condition, MDSolverBase


@annotate
def calculate_force(i, x, y, fx, fy, pe, nbr_starts, nbr_lengths, nbrs):
    start_idx = nbr_starts[i]
    length = nbr_lengths[i]
    for k in range(start_idx, start_idx + length):
        j = nbrs[k]
        if i == j:
            continue
        xij = x[i] - x[j]
        yij = y[i] - y[j]
        rij2 = xij * xij + yij * yij
        irij2 = 1.0 / rij2
        irij6 = irij2 * irij2 * irij2
        irij12 = irij6 * irij6
        pe[i] += (2 * (irij12 - irij6))
        f_base = 24 * irij2 * (2 * irij12 - irij6)

        fx[i] += f_base * xij
        fy[i] += f_base * yij


@annotate
def step_method1(i, x, y, vx, vy, fx, fy, pe, xmin, xmax,
                 ymin, ymax, m, dt, nbr_starts, nbr_lengths,
                 nbrs):
    integrate_step1(i, m, dt, x, y, vx, vy, fx, fy)
    boundary_condition(i, x, y, vx, vy, fx, fy, pe, xmin, xmax,
                       ymin, ymax)


@annotate
def step_method2(i, x, y, vx, vy, fx, fy, pe, xmin, xmax,
                 ymin, ymax, m, dt, nbr_starts, nbr_lengths,
                 nbrs):
    calculate_force(i, x, y, fx, fy, pe, nbr_starts, nbr_lengths, nbrs)
    integrate_step2(i, m, dt, x, y, vx, vy, fx, fy)


class MDNNPSSolver(MDSolverBase):
    def __init__(self, num_particles, x=None, y=None, vx=None, vy=None,
                 xmax=100., ymax=100., dx=1.5, init_T=0.,
                 backend=None, use_count_sort=False):
        super().__init__(num_particles, x=x, y=y, vx=vx, vy=vy,
                         xmax=xmax, ymax=ymax, dx=dx, init_T=init_T,
                         backend=backend)
        self.init_forces = Elementwise(calculate_force, backend=self.backend)
        self.step1 = Elementwise(step_method1, backend=self.backend)
        self.step2 = Elementwise(step_method2, backend=self.backend)
        self.nnps_algorithm = NNPSCountingSort \
            if use_count_sort else NNPSRadixSort
        self.nnps = self.nnps_algorithm(self.x, self.y, 3., self.xmax,
                                        self.ymax, backend=self.backend)

    def solve(self, t, dt):
        num_steps = int(t // dt)
        self.nnps.build()
        self.nnps.get_neighbors()
        self.init_forces(self.x, self.y, self.fx, self.fy,
                         self.pe, self.nnps.nbr_starts,
                         self.nnps.nbr_lengths, self.nnps.nbrs)
        for i in range(num_steps):
            self.step1(self.x, self.y, self.vx, self.vy, self.fx,
                       self.fy, self.pe, self.xmin, self.xmax, self.ymin,
                       self.ymax, self.m, dt, self.nnps.nbr_starts,
                       self.nnps.nbr_lengths, self.nnps.nbrs)
            self.nnps.build()
            self.nnps.get_neighbors()
            self.step2(self.x, self.y, self.vx, self.vy, self.fx,
                       self.fy, self.pe, self.xmin, self.xmax, self.ymin,
                       self.ymax, self.m, dt, self.nnps.nbr_starts,
                       self.nnps.nbr_lengths, self.nnps.nbrs)
            if i % 100 == 0:
                self.post_step(i)


if __name__ == '__main__':
    from compyle.utils import ArgumentParser
    p = ArgumentParser()
    p.add_argument(
        '--use-count-sort', action='store_true', dest='use_count_sort',
        default=False, help='Use count sort instead of radix sort'
    )
    p.add_argument(
        '--show', action='store_true', dest='show',
        default=False, help='Show plot at end of simulation'
    )

    p.add_argument('-n', action='store', type=int, dest='n',
                   default=100, help='Number of particles')

    p.add_argument('--tf', action='store', type=float, dest='t',
                   default=40., help='Final time')

    p.add_argument('--dt', action='store', type=float, dest='dt',
                   default=0.02, help='Time step')

    o = p.parse_args()

    solver = MDNNPSSolver(
        o.n,
        backend=o.backend,
        use_count_sort=o.use_count_sort)

    start = time.time()
    solver.solve(o.t, o.dt)
    end = time.time()
    print("Time taken for N = %i is %g secs" % (o.n, (end - start)))
    if o.show:
        solver.pull()
        solver.plot()
