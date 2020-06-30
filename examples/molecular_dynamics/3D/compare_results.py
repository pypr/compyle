from hoomd_periodic import simulate
from md_nnps_periodic import MDNNPSSolverPeriodic
import numpy as np
import matplotlib.pyplot as plt

def run_simulations(num_particles, tf, dt):
    # run hoomd simulation
    simulate(num_particles, dt, tf, log=True)

    # run compyle simulation
    solver = MDNNPSSolverPeriodic(num_particles)
    solver.solve(tf, dt, log_output=True)
    solver.write_log('compyle-output.log')


def plot_props(hoomd_fname, comp_fname):
    data_hoomd = np.genfromtxt(fname=hoomd_fname, skip_header=True)
    data_compyle = np.genfromtxt(fname=comp_fname)


    plt.plot(data_hoomd[:,0], data_hoomd[:,1], label="HooMD")
    plt.plot(data_hoomd[:,0], data_compyle[:,1], label="Compyle")
    plt.xlabel("Timestep")
    plt.ylabel("Potential Energy")
    plt.legend()
    plt.savefig("hoomd_pe.png", dpi=300)

    plt.clf()

    plt.plot(data_hoomd[:,0], data_hoomd[:,2], label="HooMD")
    plt.plot(data_hoomd[:,0], data_compyle[:,2], label="Compyle")
    plt.xlabel("Timestep")
    plt.ylabel("Kinetic Energy")
    plt.legend()
    plt.savefig("hoomd_ke.png", dpi=300)


if __name__ == '__main__':
    run_simulations(2000, 200, 0.02)
    plot_props('hoomd-output.log', 'compyle-output.log')

