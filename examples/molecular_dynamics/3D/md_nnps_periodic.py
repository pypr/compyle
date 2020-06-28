import numpy as np
from math import pi
import time

from compyle.config import get_config
from compyle.api import declare, annotate
from compyle.parallel import Elementwise, Reduction
from compyle.array import get_backend, wrap
from compyle.low_level import cast
import compyle.array as carr

from nnps import NNPSCountingSortPeriodic, NNPSRadixSortPeriodic
from md_simple import integrate_step1, integrate_step2, MDSolverBase


@annotate
def calculate_force(i, x, y, z, xmax, ymax, zmax, fx, fy, fz, pe,
                    nbr_starts, nbr_lengths, nbrs):
    start_idx = nbr_starts[i]
    length = nbr_lengths[i]
    halfx = 0.5 * xmax
    halfy = 0.5 * ymax
    halfz = 0.5 * zmax
    for k in range(start_idx, start_idx + length):
        j = nbrs[k]
        if i == j:
            continue
        xij = x[i] - x[j]
        yij = y[i] - y[j]
        zij = z[i] - z[j]
        signx = 1 if xij > 0 else -1
        signy = 1 if yij > 0 else -1
        signz = 1 if zij > 0 else -1
        xij = xij if abs(xij) < halfx else xij - signx * xmax
        yij = yij if abs(yij) < halfy else yij - signy * ymax
        zij = zij if abs(zij) < halfz else zij - signz * zmax
        rij2 = xij * xij + yij * yij + zij * zij
        irij2 = 1.0 / rij2
        irij6 = irij2 * irij2 * irij2
        irij12 = irij6 * irij6
        pe[i] += (2 * (irij12 - irij6))
        f_base = 24 * irij2 * (2 * irij12 - irij6)

        fx[i] += f_base * xij
        fy[i] += f_base * yij
        fz[i] += f_base * zij


@annotate
def step_method1(i, x, y, z, vx, vy, vz, fx, fy, fz, pe, xmin, xmax,
                 ymin, ymax, zmin, zmax, m, dt, nbr_starts, nbr_lengths,
                 nbrs):
    integrate_step1(i, m, dt, x, y, z, vx, vy, vz, fx, fy, fz)
    boundary_condition(i, x, y, z, fx, fy, fz, pe, xmin, xmax,
                       ymin, ymax, zmin, zmax)


@annotate
def step_method2(i, x, y, z, vx, vy, vz, fx, fy, fz, pe, xmin, xmax,
                 ymin, ymax, zmin, zmax, m, dt, nbr_starts, nbr_lengths,
                 nbrs):
    calculate_force(i, x, y, z, xmax, ymax, zmax, fx, fy, fz, pe,
                    nbr_starts, nbr_lengths, nbrs)
    integrate_step2(i, m, dt, x, y, z, vx, vy, vz, fx, fy, fz)


@annotate
def boundary_condition(i, x, y, z, fx, fy, fz, pe, xmin, xmax, ymin, ymax,
                       zmin, zmax):
    fx[i] = 0.
    fy[i] = 0.
    fz[i] = 0.
    pe[i] = 0.

    xwidth = xmax - xmin
    ywidth = ymax - ymin
    zwidth = zmax - zmin

    xoffset = cast(floor(x[i] / xmax), "int")
    yoffset = cast(floor(y[i] / ymax), "int")
    zoffset = cast(floor(z[i] / zmax), "int")

    x[i] -= xoffset * xwidth
    y[i] -= yoffset * ywidth
    z[i] -= zoffset * zwidth


class MDNNPSSolverPeriodic(MDSolverBase):
    def __init__(self, num_particles, x=None, y=None, z=None,
                 vx=None, vy=None, vz=None,
                 xmax=100., ymax=100., zmax=100., dx=2., init_T=0.,
                 backend=None, use_count_sort=False):
        super().__init__(num_particles, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
                         xmax=xmax, ymax=ymax, zmax=zmax, dx=dx, init_T=init_T,
                         backend=backend)
        self.nnps_algorithm = NNPSCountingSortPeriodic \
            if use_count_sort else NNPSRadixSortPeriodic
        self.nnps = self.nnps_algorithm(self.x, self.y, self.z, 3., 0.01,
                                        self.xmax, self.ymax, self.zmax,
                                        backend=self.backend)
        self.init_forces = Elementwise(calculate_force, backend=self.backend)
        self.step1 = Elementwise(step_method1, backend=self.backend)
        self.step2 = Elementwise(step_method2, backend=self.backend)

    def solve(self, t, dt, log_output=False):
        num_steps = int(t // dt)
        self.nnps.build()
        self.nnps.get_neighbors()
        self.init_forces(self.x, self.y, self.z, self.xmax, self.ymax,
                         self.zmax, self.fx, self.fy, self.fz,
                         self.pe, self.nnps.nbr_starts,
                         self.nnps.nbr_lengths, self.nnps.nbrs)
        for i in range(num_steps):
            self.step1(self.x, self.y, self.z, self.vx, self.vy, self.vz,
                       self.fx, self.fy, self.fz,
                       self.pe, self.xmin, self.xmax, self.ymin, self.ymax,
                       self.zmin, self.zmax, self.m, dt, self.nnps.nbr_starts,
                       self.nnps.nbr_lengths, self.nnps.nbrs)
            self.nnps.build()
            self.nnps.get_neighbors()
            self.step2(self.x, self.y, self.z, self.vx, self.vy, self.vz,
                       self.fx, self.fy, self.fz,
                       self.pe, self.xmin, self.xmax, self.ymin, self.ymax,
                       self.zmin, self.zmax, self.m, dt, self.nnps.nbr_starts,
                       self.nnps.nbr_lengths, self.nnps.nbrs)

            if i % 100 == 0:
                self.post_step(i, log_output=log_output)


if __name__ == '__main__':
    from compyle.utils import ArgumentParser
    p = ArgumentParser()
    p.add_argument(
        '--use-count-sort', action='store_true', dest='use_count_sort',
        default=False, help='Use count sort instead of radix sort'
    )
    p.add_argument(
        '--show', action='store_true', dest='show',
        default=False, help='Show plot'
    )
    p.add_argument(
        '--log-output', action='store_true', dest='log_output',
        default=False, help='Log output'
    )

    p.add_argument('-n', action='store', type=int, dest='n',
                   default=100, help='Number of particles')

    p.add_argument('--tf', action='store', type=float, dest='t',
                   default=40., help='Final time')

    p.add_argument('--dt', action='store', type=float, dest='dt',
                   default=0.02, help='Time step')

    o = p.parse_args()

    solver = MDNNPSSolverPeriodic(
        o.n,
        backend=o.backend,
        use_count_sort=o.use_count_sort)

    start = time.time()
    solver.solve(o.t, o.dt, o.log_output)
    end = time.time()
    print("Time taken for N = %i is %g secs" % (o.n, (end - start)))
    if o.log_output:
        solver.write_log('nnps_periodic.log')
    if o.show:
        solver.pull()
        solver.plot()
