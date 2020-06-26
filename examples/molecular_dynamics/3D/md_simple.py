import numpy as np
from math import pi
import time

from compyle.config import get_config
from compyle.api import declare, annotate
from compyle.parallel import Elementwise, Reduction
from compyle.array import get_backend, wrap

import compyle.array as carr


@annotate
def calculate_energy(i, vx, vy, vz, pe, num_particles):
    ke = 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i])
    return pe[i] + ke


@annotate
def calculate_force(i, x, y, z, fx, fy, fz, pe, num_particles):
    force_cutoff = 3.
    force_cutoff2 = force_cutoff * force_cutoff
    for j in range(num_particles):
        if i == j:
            continue
        xij = x[i] - x[j]
        yij = y[i] - y[j]
        zij = z[i] - z[j]
        rij2 = xij * xij + yij * yij + zij * zij
        if rij2 > force_cutoff2:
            continue
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
                 ymin, ymax, zmin, zmax, m, dt, num_particles):
    integrate_step1(i, m, dt, x, y, z, vx, vy, vz, fx, fy, fz)
    boundary_condition(i, x, y, z, vx, vy, vz, fx, fy, fz, pe, xmin, xmax,
                       ymin, ymax, zmin, zmax)


@annotate
def step_method2(i, x, y, z, vx, vy, vz, fx, fy, fz, pe, xmin, xmax,
                 ymin, ymax, zmin, zmax, m, dt, num_particles):
    calculate_force(i, x, y, z, fx, fy, fz, pe, num_particles)
    integrate_step2(i, m, dt, x, y, z, vx, vy, vz, fx, fy, fz)


@annotate
def integrate_step1(i, m, dt, x, y, z, vx, vy, vz, fx, fy, fz):
    x[i] += vx[i] * dt + 0.5 * fx[i] * dt * dt
    y[i] += vy[i] * dt + 0.5 * fy[i] * dt * dt
    z[i] += vz[i] * dt + 0.5 * fz[i] * dt * dt
    vx[i] += 0.5 * fx[i] * dt
    vy[i] += 0.5 * fy[i] * dt
    vz[i] += 0.5 * fz[i] * dt


@annotate
def integrate_step2(i, m, dt, x, y, z, vx, vy, vz, fx, fy, fz):
    vx[i] += 0.5 * fx[i] * dt
    vy[i] += 0.5 * fy[i] * dt
    vz[i] += 0.5 * fz[i] * dt


@annotate
def boundary_condition(i, x, y, z, vx, vy, vz, fx, fy, fz, pe,
                       xmin, xmax, ymin, ymax, zmin, zmax):
    xwidth = xmax - xmin
    ywidth = ymax - ymin
    zwidth = zmax - zmin
    stiffness = 50.
    pe[i] = 0.
    if x[i] < 0.5:
        fx[i] = stiffness * (0.5 - x[i])
        pe[i] += 0.5 * stiffness * (0.5 - x[i]) * (0.5 - x[i])
    elif x[i] > xwidth - 0.5:
        fx[i] = stiffness * (xwidth - 0.5 - x[i])
        pe[i] += 0.5 * stiffness * (xwidth - 0.5 - x[i]) * (xwidth - 0.5 - x[i])
    else:
        fx[i] = 0.

    if y[i] < 0.5:
        fy[i] = stiffness * (0.5 - y[i])
        pe[i] += 0.5 * stiffness * (0.5 - y[i]) * (0.5 - y[i])
    elif y[i] > ywidth - 0.5:
        fy[i] = stiffness * (ywidth - 0.5 - y[i])
        pe[i] += 0.5 * stiffness * (ywidth - 0.5 - y[i]) * (ywidth - 0.5 - y[i])
    else:
        fy[i] = 0.

    if z[i] < 0.5:
        fz[i] = stiffness * (0.5 - z[i])
        pe[i] += 0.5 * stiffness * (0.5 - z[i]) * (0.5 - z[i])
    elif z[i] > zwidth - 0.5:
        fz[i] = stiffness * (zwidth - 0.5 - z[i])
        pe[i] += 0.5 * stiffness * (zwidth - 0.5 - z[i]) * (zwidth - 0.5 - z[i])
    else:
        fz[i] = 0.


class MDSolver(object):
    def __init__(self, num_particles, x=None, y=None, z=None,
                 vx=None, vy=None, vz=None,
                 xmax=100., ymax=100., zmax=100., dx=2., init_T=0.,
                 backend=None):
        self.backend = backend
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
        self.init_forces = Elementwise(calculate_force, backend=self.backend)
        self.step1 = Elementwise(step_method1, backend=self.backend)
        self.step2 = Elementwise(step_method2, backend=self.backend)
        self.energy_calc = Reduction("a+b", map_func=calculate_energy,
                                     backend=self.backend)

    def setup_velocities(self, T, num_particles):
        np.random.seed(123)
        vx = np.random.uniform(0, 1., size=num_particles).astype(np.float64)
        vy = np.random.uniform(0, 1., size=num_particles).astype(np.float64)
        vz = np.random.uniform(0, 1., size=num_particles).astype(np.float64)
        T_current = np.average(vx ** 2 + vy ** 2 + vz ** 2)
        scaling_factor = (T / T_current) ** 0.5
        vx = vx * scaling_factor
        vy = vy * scaling_factor
        vz = vz * scaling_factor
        return wrap(vx, vy, vz, backend=self.backend)

    def setup_positions(self, num_particles, dx):
        ndim = np.ceil(num_particles ** (1 / 3.))
        dim_length = ndim * dx

        self.xmax = 3 * (1 + round(dim_length * 1.5 / 3.))
        self.ymax = 3 * (1 + round(dim_length * 1.5 / 3.))
        self.zmax = 3 * (1 + round(dim_length * 1.5 / 3.))

        xmin_eff = ((self.xmax - self.xmin) - dim_length) / 2.
        xmax_eff = ((self.xmax - self.xmin) + dim_length) / 2.

        x, y, z = np.mgrid[xmin_eff:xmax_eff:dx, xmin_eff:xmax_eff:dx,
                           xmin_eff:xmax_eff:dx]
        x = x.ravel().astype(np.float64)[:num_particles]
        y = y.ravel().astype(np.float64)[:num_particles]
        z = z.ravel().astype(np.float64)[:num_particles]
        return wrap(x, y, z, backend=self.backend)

    def post_step(self, t):
        energy = self.energy_calc(self.vx, self.vy, self.vz, self.pe,
                                  self.num_particles)
        print("Energy at time =", t, "is", energy)

    def solve(self, t, dt):
        num_steps = int(t // dt)
        curr_t = 0.
        self.init_forces(self.x, self.y, self.z, self.fx, self.fy, self.fz,
                         self.pe, self.num_particles)
        for i in range(num_steps):
            self.step1(self.x, self.y, self.z, self.vx, self.vy, self.vz,
                       self.fx, self.fy, self.fz,
                       self.pe, self.xmin, self.xmax, self.ymin, self.ymax,
                       self.zmin, self.zmax, self.m, dt, self.num_particles)
            self.step2(self.x, self.y, self.z, self.vx, self.vy, self.vz,
                       self.fx, self.fy, self.fz,
                       self.pe, self.xmin, self.xmax, self.ymin, self.ymax,
                       self.zmin, self.zmax, self.m, dt, self.num_particles)
            curr_t += dt
            if i % 100 == 0:
                self.post_step(curr_t)

    def pull(self):
        self.x.pull()
        self.y.pull()
        self.z.pull()

    def plot(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_zlim(self.zmin, self.zmax)
        ax.scatter(self.x.data, self.y.data, self.z.data)
        plt.show()


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

    p.add_argument('-n', action='store', type=int, dest='n',
                   default=100, help='Number of particles')

    p.add_argument('--tf', action='store', type=float, dest='t',
                   default=40., help='Final time')

    p.add_argument('--dt', action='store', type=float, dest='dt',
                   default=0.02, help='Time step')

    o = p.parse_args()
    get_config().use_openmp = o.openmp
    get_config().use_double = o.use_double

    solver = MDSolver(o.n, backend=o.backend)

    start = time.time()
    solver.solve(o.t, o.dt)
    end = time.time()
    print("Time taken for N = %i is %g secs" % (o.n, (end - start)))
    solver.pull()
    solver.plot()
