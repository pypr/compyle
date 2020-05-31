from compyle.config import get_config
from compyle.api import declare, annotate
from compyle.parallel import serial, Elementwise, Reduction, Scan
from compyle.array import get_backend, wrap
from compyle.low_level import atomic_inc, cast
from math import floor
from time import time

import numpy as np
import compyle.array as carr


@annotate
def find_cell_id(x, y, h, c):
    c[0] = cast(floor((x) / h), "int")
    c[1] = cast(floor((y) / h), "int")


@annotate(ulong='p, q', return_='ulong')
def interleave2(p, q):
    m1, m2, m3, m4, m5, m6 = declare('unsigned long', 6)

    m1 = 0xffffffff
    m2 = 0x0000ffff0000ffff
    m3 = 0x00ff00ff00ff00ff
    m4 = 0x0f0f0f0f0f0f0f0f
    m5 = 0x3333333333333333
    m6 = 0x5555555555555555

    p = p & m1
    p = (p | (p << 16)) & m2
    p = (p | (p << 8)) & m3
    p = (p | (p << 4)) & m4
    p = (p | (p << 2)) & m5
    p = (p | (p << 1)) & m6

    q = q & m1
    q = (q | (q << 16)) & m2
    q = (q | (q << 8)) & m3
    q = (q | (q << 4)) & m4
    q = (q | (q << 2)) & m5
    q = (q | (q << 1)) & m6

    return (p | (q << 1))


@serial
@annotate
def count_bins(i, x, y, h, keys, bin_counts, sort_offsets):
    c = declare('matrix(2, "int")')
    key = declare('unsigned long')
    find_cell_id(x[i], y[i], h, c)
    key = interleave2(c[0], c[1])
    keys[i] = key
    idx = atomic_inc(bin_counts[key])
    sort_offsets[i] = idx


@annotate
def sort_indices(i, keys, sort_offsets, start_indices, sorted_indices):
    key = keys[i]
    offset = sort_offsets[i]
    start_idx = start_indices[key]
    sorted_indices[start_idx + offset] = i


@annotate
def input_start_indices(i, counts):
    return 0 if i == 0 else counts[i - 1]


@annotate
def output_start_indices(i, item, indices):
    indices[i] = item


@annotate
def find_neighbor_lengths_knl(i, x, y, h, start_indices, sorted_indices,
                              bin_counts, nbr_lengths, max_key):
    d = h * h
    q_c = declare('matrix(2, "int")')
    key = declare('unsigned long')
    find_cell_id(x[i], y[i], h, q_c)

    for p in range(-1, 2):
        for q in range(-1, 2):
            cx = q_c[0] + p
            cy = q_c[1] + q

            key = interleave2(cx, cy)

            if key >= max_key:
                continue

            start_idx = start_indices[key]
            np = bin_counts[key]

            for k in range(np):
                j = sorted_indices[start_idx + k]
                xij = x[i] - x[j]
                yij = y[i] - y[j]
                rij2 = xij * xij + yij * yij

                if rij2 < d:
                    nbr_lengths[i] += 1


@annotate
def find_neighbors_knl(i, x, y, h, start_indices, sorted_indices,
                       bin_counts, nbr_starts, nbrs, max_key):
    d = h * h
    q_c = declare('matrix(2, "int")')
    key = declare('unsigned long')
    find_cell_id(x[i], y[i], h, q_c)
    length = 0
    nbr_start_idx = nbr_starts[i]

    for p in range(-1, 2):
        for q in range(-1, 2):
            cx = q_c[0] + p
            cy = q_c[1] + q

            key = interleave2(cx, cy)

            if key >= max_key:
                continue

            start_idx = start_indices[key]
            np = bin_counts[key]

            for k in range(np):
                j = sorted_indices[start_idx + k]
                xij = x[i] - x[j]
                yij = y[i] - y[j]
                rij2 = xij * xij + yij * yij

                if rij2 < d:
                    nbrs[nbr_start_idx + length] = j
                    length += 1


class ParticleArrayWrapper(object):
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        self.vx = None
        self.vy = None
        self.fx = None
        self.fy = None
        self.pe = None


class NNPS(object):
    def __init__(self, pa, h, xmax, ymax, backend=None):
        self.backend = backend
        self.num_particles = pa.x.length
        self.pa = pa
        self.h = h

        cmax = np.array([floor(xmax / h), floor(ymax / h)], dtype=np.int32)
        self.max_key = 1 + interleave2(cmax[0], cmax[1])

        # sort kernels
        self.count_bins = Elementwise(count_bins, backend=self.backend)
        self.sort_indices = Elementwise(sort_indices, backend=self.backend)
        self.scan_start_indices = Scan(input=input_start_indices,
                                       output=output_start_indices,
                                       scan_expr="a+b", dtype=np.int32,
                                       backend=self.backend)

        # neighbor kernels
        self.find_neighbor_lengths = Elementwise(find_neighbor_lengths_knl,
                                                 backend=self.backend)
        self.find_neighbors = Elementwise(find_neighbors_knl,
                                          backend=self.backend)

        self.init_arrays()

    def init_arrays(self):
        # sort arrays
        self.bin_counts = carr.zeros(self.max_key, dtype=np.int32,
                                     backend=self.backend)
        self.sort_offsets = carr.zeros(self.num_particles, dtype=np.int32,
                                       backend=self.backend)
        self.start_indices = carr.zeros(self.max_key, dtype=np.int32,
                                        backend=self.backend)
        self.keys = carr.zeros(self.num_particles, dtype=np.uint64,
                               backend=self.backend)
        self.pids = carr.zeros(self.num_particles, dtype=np.int32,
                               backend=self.backend)
        self.sorted_indices = carr.zeros(self.num_particles, dtype=np.int32,
                                         backend=self.backend)

        # neighbor arrays
        self.nbr_lengths = carr.zeros(self.num_particles, dtype=np.int32,
                                      backend=self.backend)
        self.nbr_starts = carr.zeros(self.num_particles, dtype=np.int32,
                                     backend=self.backend)
        self.nbrs = carr.zeros(2 * self.num_particles, dtype=np.int32,
                                     backend=self.backend)

    def reset_arrays(self):
        # sort arrays
        self.bin_counts.fill(0)
        self.sort_offsets.fill(0)
        self.start_indices.fill(0)
        self.sorted_indices.fill(0)

        # neighbors array
        self.nbr_lengths.fill(0)
        self.nbr_starts.fill(0)

    def build(self):
        self.reset_arrays()
        self.count_bins(self.pa.x, self.pa.y, self.h, self.keys, self.bin_counts,
                        self.sort_offsets)
        self.scan_start_indices(counts=self.bin_counts,
                                indices=self.start_indices)
        self.sort_indices(self.keys, self.sort_offsets, self.start_indices,
                          self.sorted_indices)

    def get_neighbors(self):
        self.find_neighbor_lengths(self.pa.x, self.pa.y, self.h, self.start_indices,
                                   self.sorted_indices, self.bin_counts, self.nbr_lengths,
                                   self.max_key)
        self.scan_start_indices(counts=self.nbr_lengths,
                                indices=self.nbr_starts)
        self.total_neighbors = int(self.nbr_lengths[-1] + self.nbr_starts[-1])
        self.nbrs.resize(self.total_neighbors)
        self.find_neighbors(self.pa.x, self.pa.y, self.h, self.start_indices,
                            self.sorted_indices, self.bin_counts, self.nbr_starts,
                            self.nbrs, self.max_key)


if __name__ == "__main__":
    backend = 'opencl'
    np.random.seed(123)
    num_particles = 200
    x = np.random.uniform(0, 100., size=num_particles).astype(np.float32)
    y = np.random.uniform(0, 100., size=num_particles).astype(np.float32)
    x, y = wrap(x, y, backend=backend)
    pa = ParticleArrayWrapper(x, y)
    nnps = NNPS(pa, 3., 100., 100., backend=backend)
    nnps.build()
    nnps.get_neighbors()
    print(nnps.keys.dev[nnps.sorted_indices.dev])

