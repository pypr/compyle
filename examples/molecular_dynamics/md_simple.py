import numpy as np
from math import pi
import time

from compyle.config import get_config
from compyle.api import declare, annotate
from compyle.parallel import Elementwise, Reduction
from compyle.array import get_backend, wrap

import compyle.array as carr


@annotate
def calculate_energy(i, vx, vy, pe, num_particles):
    ke = 0.5 * (vx[i] * vx[i] + vy[i] * vy[i])
    return pe[i] + ke


@annotate
def calculate_force(i, x, y, fx, fy, pe, num_particles):
    force_cutoff = 3.
    force_cutoff2 = force_cutoff * force_cutoff
    for j in range(num_particles):
        if i == j:
            continue
        xij = x[i] - x[j]
        yij = y[i] - y[j]
        rij2 = xij * xij + yij * yij
        if rij2 > force_cutoff2:
            continue
        irij2 = 1.0 / rij2
        irij6 = irij2 * irij2 * irij2
        irij12 = irij6 * irij6
        pe[i] += (2 * (irij12 - irij6))
        f_base = 24 * irij2 * (2 * irij12 - irij6)

        fx[i] += f_base * xij
        fy[i] += f_base * yij


@annotate
def step_method1(i, x, y, vx, vy, fx, fy, pe, xmin, xmax,
                 ymin, ymax, m, dt, num_particles):
    integrate_step1(i, m, dt, x, y, vx, vy, fx, fy)
    boundary_condition(i, x, y, vx, vy, fx, fy, pe, xmin, xmax,
                       ymin, ymax)


@annotate
def step_method2(i, x, y, vx, vy, fx, fy, pe, xmin, xmax,
                 ymin, ymax, m, dt, num_particles):
    calculate_force(i, x, y, fx, fy, pe, num_particles)
    integrate_step2(i, m, dt, x, y, vx, vy, fx, fy)


@annotate
def integrate_step1(i, m, dt, x, y, vx, vy, fx, fy):
    axi = fx[i]
    ayi = fy[i]
    x[i] += vx[i] * dt + 0.5 * axi * dt * dt
    y[i] += vy[i] * dt + 0.5 * ayi * dt * dt
    vx[i] += 0.5 * axi * dt
    vy[i] += 0.5 * ayi * dt


@annotate
def integrate_step2(i, m, dt, x, y, vx, vy, fx, fy):
    axi = fx[i]
    ayi = fy[i]
    vx[i] += 0.5 * axi * dt
    vy[i] += 0.5 * ayi * dt


@annotate
def boundary_condition(i, x, y, vx, vy, fx, fy, pe, xmin, xmax, ymin, ymax):
    xwidth = xmax - xmin
    ywidth = ymax - ymin
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


class MDSolverBase(object):
    def __init__(self, num_particles, x=None, y=None, vx=None, vy=None,
                 xmax=100., ymax=100., dx=1.5, init_T=0.,
                 backend=None):
        self.backend = get_backend(backend)
        self.num_particles = num_particles
        self.xmin, self.xmax = 0., xmax
        self.ymin, self.ymax = 0., ymax
        self.m = 1.
        if x is None and y is None:
            self.x, self.y = self.setup_positions(num_particles, dx)
        if vx is None and vy is None:
            self.vx, self.vy = self.setup_velocities(init_T, num_particles)
        self.fx = carr.zeros_like(self.x, backend=self.backend)
        self.fy = carr.zeros_like(self.x, backend=self.backend)
        self.pe = carr.zeros_like(self.x, backend=self.backend)
        self.energy_calc = Reduction("a+b", map_func=calculate_energy,
                                     backend=self.backend)

    def setup_velocities(self, T, num_particles):
        np.random.seed(123)
        vx = np.random.uniform(0, 1., size=num_particles).astype(np.float64)
        vy = np.random.uniform(0, 1., size=num_particles).astype(np.float64)
        T_current = np.average(vx ** 2 + vy ** 2)
        scaling_factor = (T / T_current) ** 0.5
        vx = vx * scaling_factor
        vy = vy * scaling_factor
        return wrap(vx, vy, backend=self.backend)

    def setup_positions(self, num_particles, dx):
        ndim = np.ceil(num_particles ** 0.5)
        dim_length = ndim * dx

        self.xmax = dim_length * 3
        self.ymax = dim_length * 3

        xmin_eff = ((self.xmax - self.xmin) - dim_length) / 2.
        xmax_eff = ((self.xmax - self.xmin) + dim_length) / 2.

        x, y = np.mgrid[xmin_eff:xmax_eff:dx, xmin_eff:xmax_eff:dx]
        x = x.ravel().astype(np.float64)[:num_particles]
        y = y.ravel().astype(np.float64)[:num_particles]
        return wrap(x, y, backend=self.backend)

    def post_step(self, step):
        energy = self.energy_calc(self.vx, self.vy, self.pe,
                                  self.num_particles)
        print("Energy at time step =", step, "is", energy)

    def pull(self):
        self.x.pull()
        self.y.pull()

    def plot(self):
        import matplotlib.pyplot as plt
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.scatter(self.x.data, self.y.data, 4.2)
        plt.show()


class MDSolver(MDSolverBase):
    def __init__(self, num_particles, x=None, y=None, vx=None, vy=None,
                 xmax=100., ymax=100., dx=1.5, init_T=0.,
                 backend=None):
        super().__init__(num_particles, x=x, y=y, vx=vx, vy=vy,
                         xmax=xmax, ymax=ymax, dx=dx, init_T=init_T,
                         backend=backend)
        self.init_forces = Elementwise(calculate_force, backend=self.backend)
        self.step1 = Elementwise(step_method1, backend=self.backend)
        self.step2 = Elementwise(step_method2, backend=self.backend)

    def solve(self, t, dt):
        num_steps = int(t // dt)
        self.init_forces(self.x, self.y, self.fx, self.fy, self.pe,
                         self.num_particles)
        for i in range(num_steps):
            self.step1(self.x, self.y, self.vx, self.vy, self.fx, self.fy,
                       self.pe, self.xmin, self.xmax, self.ymin, self.ymax,
                       self.m, dt, self.num_particles)
            self.step2(self.x, self.y, self.vx, self.vy, self.fx, self.fy,
                       self.pe, self.xmin, self.xmax, self.ymin, self.ymax,
                       self.m, dt, self.num_particles)
            if i % 100 == 0:
                self.post_step(i)


if __name__ == '__main__':
    from compyle.utils import ArgumentParser
    p = ArgumentParser()
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

    solver = MDSolver(o.n, backend=o.backend)

    start = time.time()
    solver.solve(o.t, o.dt)
    end = time.time()
    print("Time taken for N = %i is %g secs" % (o.n, (end - start)))
    if o.show:
        solver.pull()
        solver.plot()
