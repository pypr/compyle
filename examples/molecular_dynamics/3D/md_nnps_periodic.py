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
from md_simple import calculate_energy, integrate_step1, \
        integrate_step2, MDSolver


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
        pe[i] += (4 * (irij12 - irij6))
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


class MDNNPSSolverPeriodic(MDSolver):
    def __init__(self, num_particles, x=None, y=None, z=None,
                 vx=None, vy=None, vz=None,
                 xmax=100., ymax=100., zmax=100., dx=2., init_T=0.,
                 backend=None, use_count_sort=False):
        self.nnps_algorithm = NNPSCountingSortPeriodic \
            if use_count_sort else NNPSRadixSortPeriodic
        self.backend = get_backend(backend)
        self.num_particles = num_particles
        self.xmin, self.xmax = 0., xmax
        self.ymin, self.ymax = 0., ymax
        self.zmin, self.zmax = 0., zmax
        self.m = 1.
        if x is None and y is None and z is None:
            self.x, self.y, self.z = self.setup_positions(num_particles, dx)
        if vx is None and vy is None and vz is None:
            self.vx, self.vy, self.vz = self.setup_velocities(
                init_T, num_particles)
        self.fx = carr.zeros_like(self.x, backend=self.backend)
        self.fy = carr.zeros_like(self.y, backend=self.backend)
        self.fz = carr.zeros_like(self.z, backend=self.backend)
        self.pe = carr.zeros_like(self.x, backend=self.backend)
        self.nnps = self.nnps_algorithm(self.x, self.y, self.z, 3., 0.01, self.xmax,
                                        self.ymax, self.zmax, backend=self.backend)
        self.init_forces = Elementwise(calculate_force, backend=self.backend)
        self.step1 = Elementwise(step_method1, backend=self.backend)
        self.step2 = Elementwise(step_method2, backend=self.backend)
        self.energy_calc = Reduction("a+b", map_func=calculate_energy,
                                     backend=self.backend)

    def solve(self, t, dt):
        num_steps = int(t // dt)
        curr_t = 0.
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

            curr_t += dt
            if i % 100 == 0:
                self.post_step(curr_t)


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument(
        '-b', '--backend', action='store', dest='backend', default='cython',
        help='Choose the backend.'
    )
    p.add_argument(
        '--openmp', action='store_true', dest='openmp', default=False,
        help='Use OpenMP.'
    )
    p.add_argument(
        '--use-double', action='store_true', dest='use_double',
        default=False, help='Use double precision on the GPU.'
    )
    p.add_argument(
        '--use-count-sort', action='store_true', dest='use_count_sort',
        default=False, help='Use count sort instead of radix sort'
    )
    p.add_argument(
        '--show', action='store_true', dest='show',
        default=False, help='Show plot'
    )

    p.add_argument('-n', action='store', type=int, dest='n',
                   default=100, help='Number of particles')

    p.add_argument('--tf', action='store', type=float, dest='t',
                   default=40., help='Final time')

    p.add_argument('--dt', action='store', type=float, dest='dt',
                   default=0.02, help='Time step')

    o = p.parse_args()
    get_config().use_openmp = o.openmp
    get_config().use_double = o.use_double

    solver = MDNNPSSolverPeriodic(
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
