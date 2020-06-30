import hoomd
import hoomd.md
import numpy as np
import time


def setup_positions(num_particles, dx):
    ndim = np.ceil(num_particles ** (1 / 3.))
    dim_length = ndim * dx

    xmax = 3 * (1 + round(dim_length * 1.5 / 3.))
    ymax = 3 * (1 + round(dim_length * 1.5 / 3.))
    zmax = 3 * (1 + round(dim_length * 1.5 / 3.))

    print(dim_length, xmax)

    xmin_eff = (xmax - dim_length) / 2.
    xmax_eff = (xmax + dim_length) / 2.

    x, y, z = np.mgrid[xmin_eff:xmax_eff:dx, xmin_eff:xmax_eff:dx,
                       xmin_eff:xmax_eff:dx]
    x = x.ravel().astype(np.float32)[:num_particles]
    y = y.ravel().astype(np.float32)[:num_particles]
    z = z.ravel().astype(np.float32)[:num_particles]
    return x, y, z, xmax


def simulate(num_particles, dt, tf, profile=False, log=False):
    x, y, z, L = setup_positions(num_particles, 2.)
    positions = np.array((x, y, z)).T
    hoomd.context.initialize("")

    snapshot = hoomd.data.make_snapshot(N=len(positions),
                                        box=hoomd.data.boxdim(
                                            Lx=L, Ly=L, Lz=L),
                                        particle_types=['A'],
                                        )
    # need to get automated positions...
    snapshot.particles.position[:] = positions - 0.5 * L

    snapshot.particles.typeid[:] = 0

    hoomd.init.read_snapshot(snapshot)

    nl = hoomd.md.nlist.cell(r_buff=0)
    lj = hoomd.md.pair.lj(r_cut=3.0, nlist=nl)
    lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)

    if log:
        hoomd.analyze.log(filename="hoomd-output.log",
                          quantities=['potential_energy', 'kinetic_energy'],
                          period=100,
                          overwrite=True)

    # Create integrator and forces
    hoomd.md.integrate.mode_standard(dt=dt)
    hoomd.md.integrate.nve(group=hoomd.group.all())

    nsteps = int(tf // dt)
    start = time.time()
    hoomd.run(nsteps, profile=profile)
    end = time.time()
    return end - start


if __name__ == '__main__':
    import sys
    print(simulate(int(sys.argv[1]), 0.02, 200., profile=True, log=True))
