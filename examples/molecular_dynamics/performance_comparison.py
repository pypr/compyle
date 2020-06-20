import numpy as np
import time
import md_simple
import md_nnps

from compyle.config import get_config


def solve(n, backend, solver_algo, tf=0.5, dt=0.02, use_count_sort=False):
    solver = solver_algo(n, backend=backend.replace("_omp", ""))
    start = time.time()
    solver.solve(tf, dt)
    end = time.time()
    print("Time taken for backend = %s, N = %i is %g secs" %
          (backend, n, (end - start)))
    return end - start


def compare(backends, n_list, solver_algo, niter=3):
    t_list = {b: [] for b in backends}
    speedups = {b: [] for b in backends}
    for n in n_list:
        print("Running for N = %i" % n)
        for backend in backends:
            if "omp" in backend:
                get_config().use_openmp = True
            t = 1e9
            for it in range(niter):
                t = min(t, solve(n, backend, solver_algo))
            t_list[backend].append(t)
            if "omp" in backend:
                get_config().use_openmp = False

    if 'cython' in backends:
        for backend in backends:
            for i, n in enumerate(n_list):
                speedups[backend].append(
                    t_list["cython"][i] / t_list[backend][i])
    else:
        speedups = None

    return speedups, t_list


def compare_implementations(backend, n_list, niter=3):
    import matplotlib.pyplot as plt
    sp, nnps_tlist = compare([backend], n_list,
                             md_nnps.MDSolver, niter=niter)
    sp, simple_tlist = compare([backend], n_list,
                               md_simple.MDSolver, niter=niter)

    speedup = [simple_tlist[backend][i] / nnps_tlist[backend][i]
               for i in range(len(n_list))]

    plt.loglog(n_list, nnps_tlist[backend], 'x-', label="Linear")
    plt.loglog(n_list, simple_tlist[backend], 'x-', label="Simple")

    plt.xlabel("Number of particles")
    plt.ylabel("Time (secs)")
    plt.legend()
    plt.grid(True)
    plt.savefig("time_comp_impl.png", dpi=300)

    plt.clf()

    plt.loglog(n_list, speedup, 'x-')

    plt.xlabel("Number of particles")
    plt.ylabel("Speedup")
    plt.grid(True)
    plt.savefig("speedup_comp_impl.png", dpi=300)


def plot(n_list, speedups, t_list, label):
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
        '-c', '--comparison', action='store', dest='comp', default='gpu_comp',
        choices=['gpu_comp', 'omp_comp', 'comp_algo'],
        help='Choose the comparison.'
    )
    p.add_argument(
        '--nnps', action='store', dest='nnps', default='linear',
        choices=['linear', 'simple'],
        help='Choose algorithm.'
    )
    p.add_argument(
        '--use-double', action='store_true', dest='use_double',
        default=False,  help='Use double precision on the GPU.'
    )

    o = p.parse_args()
    get_config().use_double = o.use_double
    solver_algo = (md_nnps.MDNNPSSolver if o.nnps == 'linear'
                   else md_simple.MDSolver)
    n_list = [10000 * (2 ** i) for i in range(10)] if o.nnps == 'linear' else \
        [500 * (2 ** i) for i in range(8)]

    if o.comp == "gpu_comp":
        backends = ["opencl", "cuda", "cython"]
        print("Running for", n_list)
        speedups, t_list = compare(backends, n_list, solver_algo)
        plot(n_list, speedups, t_list, o.nnps)
    elif o.comp == "omp_comp":
        backends = ["cython_omp", "cython"]
        print("Running for", n_list)
        speedups, t_list = compare(backends, n_list, solver_algo)
        plot(n_list, speedups, t_list, o.nnps)
    elif o.comp == "comp_algo":
        backend = "cython"
        n_list = [500, 1000, 2000, 4000, 8000, 16000, 32000]
        print("Running for", n_list)
        compare_implementations(backend, n_list)
