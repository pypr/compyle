from compyle.api import declare, annotate
from compyle.parallel import serial
from compyle.low_level import atomic_inc, cast
from math import floor
import numpy as np


@annotate
def find_cell_id(x, y, h, c):
    c[0] = cast(floor((x) / h), "int")
    c[1] = cast(floor((y) / h), "int")


@annotate
def flatten(p, q, qmax):
    return p * qmax + q


@serial
@annotate
def count_bins(i, x, y, h, cmax, keys, bin_counts,
               sort_offsets):
    c = declare('matrix(2, "int")')
    find_cell_id(x[i], y[i], h, c)
    key = flatten(c[0], c[1], cmax)
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
def fill_keys(i, x, y, h, cmax, indices, keys):
    c = declare('matrix(2, "int")')
    find_cell_id(x[i], y[i], h, c)
    key = flatten(c[0], c[1], cmax)
    keys[i] = key
    indices[i] = i


@annotate
def input_scan_keys(i, keys):
    return 1 if i == 0 or keys[i] != keys[i - 1] else 0


@annotate
def output_scan_keys(i, item, prev_item, keys, start_indices):
    key = keys[i]
    if item != prev_item:
        start_indices[key] = i


@annotate
def fill_bin_counts(i, keys, start_indices, bin_counts, num_particles):
    if i == num_particles - 1:
        last_key = keys[num_particles - 1]
        bin_counts[last_key] = num_particles - start_indices[last_key]
    if i == 0 or keys[i] == keys[i - 1]:
        return
    key = keys[i]
    prev_key = keys[i - 1]
    bin_counts[prev_key] = start_indices[key] - start_indices[prev_key]


@annotate
def find_neighbor_lengths_knl(i, x, y, h, cmax, start_indices, sorted_indices,
                              bin_counts, nbr_lengths, max_key):
    d = h * h
    q_c = declare('matrix(2, "int")')
    find_cell_id(x[i], y[i], h, q_c)

    for p in range(-1, 2):
        for q in range(-1, 2):
            cx = q_c[0] + p
            cy = q_c[1] + q

            key = flatten(cx, cy, cmax)

            if key >= max_key or key < 0:
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
def find_neighbors_knl(i, x, y, h, cmax, start_indices, sorted_indices,
                       bin_counts, nbr_starts, nbrs, max_key):
    d = h * h
    q_c = declare('matrix(2, "int")')
    find_cell_id(x[i], y[i], h, q_c)
    length = 0
    nbr_start_idx = nbr_starts[i]

    for p in range(-1, 2):
        for q in range(-1, 2):
            cx = q_c[0] + p
            cy = q_c[1] + q

            key = flatten(cx, cy, cmax)

            if key >= max_key or key < 0:
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
