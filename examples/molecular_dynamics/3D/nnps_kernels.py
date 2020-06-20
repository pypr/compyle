from compyle.api import declare, annotate
from compyle.parallel import serial
from compyle.low_level import atomic_inc, cast
from math import floor
import numpy as np


@annotate
def find_cell_id(x, y, z, h, eps, c):
    c[0] = cast(floor((x + eps) / h), "int")
    c[1] = cast(floor((y + eps) / h), "int")
    c[2] = cast(floor((z + eps) / h), "int")


@annotate
def flatten(p, q, r, qmax, rmax):
    return (p * qmax + q) * rmax + r


@serial
@annotate
def count_bins(i, x, y, z, h, eps, qmax, rmax, keys, bin_counts,
               sort_offsets):
    c = declare('matrix(3, "int")')
    find_cell_id(x[i], y[i], z[i], h, eps, c)
    key = flatten(c[0], c[1], c[2], qmax, rmax)
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
def fill_keys(i, x, y, z, h, eps, qmax, rmax, indices, keys):
    c = declare('matrix(3, "int")')
    find_cell_id(x[i], y[i], z[i], h, eps, c)
    key = flatten(c[0], c[1], c[2], qmax, rmax)
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
def find_neighbor_lengths_knl(i, x, y, z, h, eps, qmax, rmax, start_indices,
                              sorted_indices, bin_counts, nbr_lengths,
                              max_key):
    d = h * h
    q_c = declare('matrix(3, "int")')
    find_cell_id(x[i], y[i], z[i], h, eps, q_c)

    for p in range(-1, 2):
        for q in range(-1, 2):
            for r in range(-1, 2):
                cx = q_c[0] + p
                cy = q_c[1] + q
                cz = q_c[2] + r

                key = flatten(cx, cy, cz, qmax, rmax)

                if key >= max_key or key < 0:
                    continue

                start_idx = start_indices[key]
                np = bin_counts[key]

                for k in range(np):
                    j = sorted_indices[start_idx + k]
                    xij = x[i] - x[j]
                    yij = y[i] - y[j]
                    zij = z[i] - z[j]
                    rij2 = xij * xij + yij * yij + zij * zij

                    if rij2 < d:
                        nbr_lengths[i] += 1


@annotate
def find_neighbors_knl(i, x, y, z, h, eps, qmax, rmax, start_indices, sorted_indices,
                       bin_counts, nbr_starts, nbrs, max_key):
    d = h * h
    q_c = declare('matrix(3, "int")')
    find_cell_id(x[i], y[i], z[i], h, eps, q_c)
    length = 0
    nbr_start_idx = nbr_starts[i]

    for p in range(-1, 2):
        for q in range(-1, 2):
            for r in range(-1, 2):
                cx = q_c[0] + p
                cy = q_c[1] + q
                cz = q_c[2] + r

                key = flatten(cx, cy, cz, qmax, rmax)

                if key >= max_key or key < 0:
                    continue

                start_idx = start_indices[key]
                np = bin_counts[key]

                for k in range(np):
                    j = sorted_indices[start_idx + k]
                    xij = x[i] - x[j]
                    yij = y[i] - y[j]
                    zij = z[i] - z[j]
                    rij2 = xij * xij + yij * yij + zij * zij

                    if rij2 < d:
                        nbrs[nbr_start_idx + length] = j
                        length += 1


@annotate
def find_neighbor_lengths_periodic_knl(i, x, y, z, h, eps, xmax, ymax, zmax,
                                       pmax, qmax, rmax, start_indices,
                                       sorted_indices, bin_counts, nbr_lengths,
                                       max_key):
    d = h * h
    q_c = declare('matrix(3, "int")')
    xij, yij, zij = declare('double', 3)
    find_cell_id(x[i], y[i], z[i], h, eps, q_c)

    for p in range(-1, 2):
        for q in range(-1, 2):
            for r in range(-1, 2):
                cx = q_c[0] + p
                cy = q_c[1] + q
                cz = q_c[2] + r

                cx_f = cast(cx, "float")
                cy_f = cast(cy, "float")
                cz_f = cast(cz, "float")

                xoffset = cast(floor(cx_f / pmax), "int")
                yoffset = cast(floor(cy_f / qmax), "int")
                zoffset = cast(floor(cz_f / rmax), "int")

                cx -= xoffset * pmax
                cy -= yoffset * qmax
                cz -= zoffset * rmax

                key = flatten(cx, cy, cz, qmax, rmax)

                if key >= max_key or key < 0:
                    continue

                start_idx = start_indices[key]
                np = bin_counts[key]

                for k in range(np):
                    j = sorted_indices[start_idx + k]
                    xij = abs(x[i] - x[j])
                    yij = abs(y[i] - y[j])
                    zij = abs(z[i] - z[j])
                    xij = min(xij, xmax - xij)
                    yij = min(yij, ymax - yij)
                    zij = min(zij, zmax - zij)
                    rij2 = xij * xij + yij * yij + zij * zij

                    if rij2 < d:
                        nbr_lengths[i] += 1


@annotate
def find_neighbors_periodic_knl(i, x, y, z, h, eps, xmax, ymax, zmax,
                                pmax, qmax, rmax, start_indices, sorted_indices,
                                bin_counts, nbr_starts, nbrs, max_key):
    d = h * h
    q_c = declare('matrix(3, "int")')
    xij, yij, zij = declare('double', 3)
    find_cell_id(x[i], y[i], z[i], h, eps, q_c)
    length = 0
    nbr_start_idx = nbr_starts[i]

    for p in range(-1, 2):
        for q in range(-1, 2):
            for r in range(-1, 2):
                cx = q_c[0] + p
                cy = q_c[1] + q
                cz = q_c[2] + r

                cx_f = cast(cx, "float")
                cy_f = cast(cy, "float")
                cz_f = cast(cz, "float")

                xoffset = cast(floor(cx_f / pmax), "int")
                yoffset = cast(floor(cy_f / qmax), "int")
                zoffset = cast(floor(cz_f / rmax), "int")

                cx -= xoffset * pmax
                cy -= yoffset * qmax
                cz -= zoffset * rmax

                key = flatten(cx, cy, cz, qmax, rmax)

                if key >= max_key or key < 0:
                    continue

                start_idx = start_indices[key]
                np = bin_counts[key]

                for k in range(np):
                    j = sorted_indices[start_idx + k]
                    xij = abs(x[i] - x[j])
                    yij = abs(y[i] - y[j])
                    zij = abs(z[i] - z[j])
                    xij = min(xij, xmax - xij)
                    yij = min(yij, ymax - yij)
                    zij = min(zij, zmax - zij)
                    rij2 = xij * xij + yij * yij + zij * zij

                    if rij2 < d:
                        nbrs[nbr_start_idx + length] = j
                        length += 1
