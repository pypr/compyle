import numpy as np
import time
from md_nnps import MDSolver

from compyle.config import get_config


def solve(n, backend, tf=0.5, dt=0.02, use_count_sort=False):
    solver = MDSolver(n, backend=backend.replace("_omp", ""),
                      use_count_sort=use_count_sort)

    start = time.time()
    solver.solve(tf, dt)
    end = time.time()
    print("Time taken for backend = %s, N = %i is %g secs" %
          (backend, n, (end - start)))
    return end - start


def compare(backends, n_list, niter=3):
    t_list = {b: [] for b in backends}
    speedups = {b: [] for b in backends}
    for n in n_list:
        for backend in backends:
            if "omp" in backend:
                get_config().use_openmp = True
            t = 1e9
            for it in range(niter):
                t = min(t, solve(n, backend))
            t_list[backend].append(t)
            if "omp" in backend:
                get_config().use_openmp = False

    if 'cython' in backends:
        for backend in backends:
            for i, n in enumerate(n_list):
                speedups[backend].append(t_list["cython"][i] / t_list[backend][i])
    else:
        speedups = None

    return speedups, t_list


def plot(n_list, speedups, t_list):
    backend_label_map = {'cython': 'Cython', 'cython_omp': 'OpenMP',
                         'opencl': 'OpenCL', 'cuda': 'CUDA'}
    import matplotlib.pyplot as plt
    plt.figure()

    if speedups:
        for backend, arr in speedups.items():
            if backend == "cython":
                continue
            plt.semilogx(n_list, arr, 'x-', label=backend_label_map[backend])

        plt.xlabel("Number of particles")
        plt.ylabel("Speedup")
        plt.legend()
        plt.savefig("speedup_%s.png" % "_".join(speedups.keys()), dpi=300)

    plt.clf()

    for backend, arr in t_list.items():
        plt.loglog(n_list, arr, 'x-', label=backend_label_map[backend])

    plt.xlabel("Number of particles")
    plt.ylabel("Time (secs)")
    plt.legend()
    plt.savefig("time_%s.png" % "_".join(t_list.keys()), dpi=300)


if __name__ == "__main__":
    backends = ["opencl", "cuda"]
    n_list = [10000, 40000, 160000, 640000, 2560000, 10240000, 40960000]
    speedups, t_list = compare(backends, n_list)
    plot(n_list, speedups, t_list)
