import numpy as np

from .config import get_config
from .cython_generator import get_parallel_range, CythonGenerator
from .transpiler import Transpiler, convert_to_float_if_needed
from .types import dtype_to_ctype, annotate
from .parallel import Scan
from .template import Template

from . import array


class OutputSortBit(Template):
    def __init__(self, name, num_arys):
        super(OutputSortBit, self).__init__(name=name)
        self.num_arys = num_arys

    def extra_args(self):
        args = ['inp_%s' % num for num in range(self.num_arys)]
        args += ['out_%s' % num for num in range(self.num_arys)]
        return args, {}

    def template(self, i, item, prev_item, last_item, bit_number, indices,
                 sorted_indices):
        '''
        key_bit = (inp_0[i] >> bit_number) & 1
        t = last_item + i - prev_item
        idx = t if key_bit else prev_item

        sorted_indices[idx] = indices[i]
        % for num in range(obj.num_arys):
        out_${num}[idx] = inp_${num}[i]
        % endfor
        '''


@annotate
def input_sort_bit(i, inp_0, bit_number):
    return 1 if (inp_0[i] >> bit_number) & 1 == 0 else 0


def radix_sort(ary_list, out_list=None, max_key_bits=None, backend=None):
    keys = ary_list[0]
    backend = array.get_backend(backend)
    if not np.issubdtype(keys.dtype, np.integer):
        raise ValueError("RadixSort can only sort integer types")
    if max_key_bits is None:
        max_key_bits = 8 * keys.dtype.itemsize

    # temp arrays
    sorted_indices = array.zeros(keys.length, np.int32, backend=backend)
    temp_indices = array.zeros_like(sorted_indices)

    indices = array.arange(0, keys.length, 1, backend=backend)

    # allocate temp arrays
    if out_list:
        temp_ary_list = out_list
    else:
        temp_ary_list = [array.zeros_like(ary) for ary in ary_list]
    sorted_ary_list = [array.zeros_like(ary) for ary in ary_list]

    # kernel
    output_sort_bit = OutputSortBit('output_sort_bit', len(ary_list))

    sort_bit_knl = Scan(input_sort_bit, output_sort_bit.function,
                        'a+b', dtype=keys.dtype, backend=backend)

    for bit_number in range(max_key_bits):
        if bit_number == 0:
            inp_indices = indices
            inp_ary_list = ary_list
        else:
            inp_indices = temp_indices
            inp_ary_list = temp_ary_list

        args = {'bit_number': bit_number, 'indices': indices,
                'sorted_indices': sorted_indices}
        args.update({'inp_%i' % i: ary for i, ary in enumerate(inp_ary_list)})
        args.update({'out_%i' %
                     i: ary for i, ary in enumerate(sorted_ary_list)})

        sort_bit_knl(**args)

        temp_indices, sorted_indices = sorted_indices, temp_indices
        temp_ary_list, sorted_ary_list = sorted_ary_list, temp_ary_list

    return temp_ary_list, temp_indices
