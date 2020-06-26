import numpy as np
import time
from md_nnps_periodic import MDNNPSSolverPeriodic

from compyle.config import get_config
from hoomd_periodic import simulate


def solve(n, backend, tf=4., dt=0.02, use_count_sort=False):
    if backend == 'hoomd':
        return simulate(n, dt, tf)
    else:
        solver = MDNNPSSolverPeriodic(
            n, dx=2., backend=backend, use_count_sort=use_count_sort)
        start = time.time()
        solver.solve(tf, dt)
        end = time.time()
        print("Time taken for backend = %s, N = %i is %g secs" %
              (backend, n, (end - start)))
        return end - start


def compare(backends, n_list, niter=3, use_count_sort=False):
    t_list = {b: [] for b in backends}
    speedups = {b: [] for b in backends}
    for backend in backends:
        for n in n_list:
            print("Running for N = %i" % n)
            t = 1e9
            for it in range(niter):
                t = min(t, solve(n, backend, use_count_sort=use_count_sort))
            t_list[backend].append(t)

    if 'hoomd' in backends:
        for backend in backends:
            for i, n in enumerate(n_list):
                speedups[backend].append(
                    t_list['hoomd'][i] / t_list[backend][i])
    else:
        speedups = None

    return speedups, t_list


def plot(n_list, speedups, t_list, label):
    backend_label_map = {'hoomd': 'HooMD',
                         'opencl': 'OpenCL', 'cuda': 'CUDA'}
    import matplotlib.pyplot as plt
    plt.figure()

    if speedups:
        for backend, arr in speedups.items():
            if backend == "hoomd":
                continue
            plt.semilogx(n_list, arr, 'x-', label=backend_label_map[backend])

        plt.xlabel("Number of particles")
        plt.ylabel("Speedup")
        plt.legend()
        plt.grid(True)
        plt.savefig("%s_speedup_%s.png" %
                    (label, "_".join(speedups.keys())), dpi=300)

    plt.clf()

    for backend, arr in t_list.items():
        plt.loglog(n_list, arr, 'x-', label=backend_label_map[backend])

    plt.xlabel("Number of particles")
    plt.ylabel("Time (secs)")
    plt.legend()
    plt.grid(True)
    plt.savefig("%s_time_%s.png" % (label, "_".join(t_list.keys())), dpi=300)


if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser()

    p.add_argument(
        '--use-count-sort', action='store_true', dest='use_count_sort',
        default=False, help='Use count sort instead of radix sort'
    )
    o = p.parse_args()

    n_list = [1000 * (2 ** i) for i in range(11)]
    backends = ["cuda", "hoomd"]
    print("Running for", n_list)
    speedups, t_list = compare(backends, n_list,
                               use_count_sort=o.use_count_sort)
    plot(n_list, speedups, t_list, "hoomd")
