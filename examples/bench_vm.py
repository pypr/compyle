import numpy as np
import time

from compyle.config import get_config
import vm_numba as VN
import vm_elementwise as VE
import vm_kernel as VK


def setup(mod, backend, openmp):
    get_config().use_openmp = openmp
    if mod == VE:
        e = VE.Elementwise(VE.velocity, backend)
    elif mod == VN:
        e = VN.velocity
    elif mod == VK:
        e = VK.Kernel(VK.velocity, backend)

    return e


def data(n, mod, backend):
    if mod == VN:
        args = mod.make_vortices(n)
    else:
        args = mod.make_vortices(n, backend)
    return args


def compare(m=5):
    # Warm up the jit to prevent the timing from going off for the first point.
    VN.velocity(*VN.make_vortices(100))
    N = np.array([10, 50, 100, 200, 500, 1000, 2000, 4000, 6000,
                  8000, 10000, 15000, 20000])
    backends = [(VN, '', False), (VE, 'cython', False), (VE, 'cython', True),
                (VE, 'opencl', False), (VK, 'opencl', False),
                (VE, 'c', False), (VE, 'c', True)]
    timing = []
    for backend in backends:
        e = setup(*backend)
        times = []
        for n in N:
            args = data(n, backend[0], backend[1])
            t = []
            for j in range(m):
                start = time.time()
                e(*args)
                t.append(time.time() - start)
            times.append(np.min(t))
        timing.append(times)

    return N, np.array(timing)


def plot_timing(n, timing):
    from matplotlib import pyplot as plt
    plt.plot(n, timing[0]/timing[1], label='numba/cython', marker='+')
    plt.plot(n, timing[0]/timing[2], label='numba/openmp', marker='+')
    plt.plot(n, timing[0]/timing[3], label='numba/opencl', marker='+')
    plt.plot(n, timing[0]/timing[4], label='numba/opencl local', marker='+')
    plt.plot(n, timing[5]/timing[1], label='c/cython', marker='+')
    plt.plot(n, timing[6]/timing[2], label='c-openmp/openmp', marker='+')
    plt.grid()
    plt.xlabel('N')
    plt.ylabel('Speedup')
    plt.legend()
    plt.figure()
    gflop = 12*n*n/1e9
    plt.plot(n, gflop/timing[0], label='numba', marker='+')
    plt.plot(n, gflop/timing[1], label='Cython', marker='+')
    plt.plot(n, gflop/timing[2], label='OpenMP', marker='+')
    plt.plot(n, gflop/timing[3], label='OpenCL', marker='+')
    plt.plot(n, gflop/timing[4], label='OpenCL Local', marker='+')
    plt.plot(n, gflop/timing[5], label='C', marker='+')
    plt.plot(n, gflop/timing[6], label='C-OpenMP', marker='+')
    plt.grid()
    plt.xlabel('N')
    plt.ylabel('GFLOPS')
    plt.legend()
    plt.show()
    best = timing[:, -1].min()
    print("Fastest time for n=", n[-1], best, "secs")


if __name__ == '__main__':
    n, t = compare()
    plot_timing(n, t)
