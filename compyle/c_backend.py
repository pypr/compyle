from textwrap import dedent
import pybind11
from .translator import CConverter
from mako.template import Template
from .translator import ocl_detect_type, OpenCLConverter, KnownType
from .cython_generator import CythonGenerator, get_func_definition, getsourcelines

pybind11_wrap_fn = '''
PYBIND11_MODULE(${name}, m) {
    m.def("${name}", &${name}, "${doc}");
}
'''


class CBackend(CythonGenerator):
    def __init__(self, detect_type=ocl_detect_type, known_types=None):
        super(CBackend, self).__init__()
        # self.function_address_space = 'WITHIN_KERNEL '

    def add_pybind11_func(self, node):
        args = self._get_function_args(node)
        name = node.body[0].name
        doc = ''
        template = Template(pybind11_wrap_fn)
        src = template.render(
            name = name,
            doc = doc
        )
        return src

    def get_func_signature_pyb11(self, func):
        sourcelines = getsourcelines(func)[0]
        defn, lines = get_func_definition(sourcelines)
        f_name, returns, args = self._analyze_method(func, lines)
        pyb11_args = []
        pyb11_call = []
        c_args = []
        c_call = []
        for arg, value in args:
            c_type = self.detect_type(arg, value)
            c_args.append('{type} {arg}'.format(type=c_type, arg=arg))

            c_call.append(arg)
            pyb11_type = self.ctype_to_pyb11(c_type)
            pyb11_args.append('{type} {arg}'.format(type=pyb11_type, arg=arg))
            if c_type.endswith('*'):
                pyb11_call.append('({ctype}){arg}.request().ptr'.format(arg = arg, ctype = c_type))
            else:
                pyb11_call.append('{arg}'.format(arg=arg))

        return (pyb11_args, pyb11_call), (c_args, c_call)

    def ctype_to_pyb11(self, c_type):
        if c_type[-1] == '*':
            return 'py::array_t<{}>'.format(c_type[:-1])
        else:
            return c_type

    def _get_self_type(self):
        return KnownType('GLOBAL_MEM %s*' % self._class_name)



reduction_c_template = '''
template<typename T>
T combine(T a, T b){
    return ${red_expr};
}

template<typename T>
T reduce_one_ar(int offset, int n, T initial_val, T* ary){
    T a, b, temp;
    temp = initial_val;

    for (int i = offset; i < (n + offset); i++){
        a = temp;
        b = ary[i];

        temp = combine<T>(a, b);
    }
    return temp;
}

template<typename T>
T reduce(int offset, int n, T initial_val${args_extra}){
    T a, b, temp;
    temp = initial_val;

    for (int i = offset; i < (n + offset); i++){
        a = temp;
        b = ${map_expr};

        temp = combine<T>(a, b);
    }
    return temp;
}


template <typename T>
T reduce_all(long N, T initial_val${args_extra}){
    T ans = initial_val;
    if (N > 0){
        %if openmp:
        int ntiles = omp_get_max_threads();
        %else:
        int ntiles = 1;
        %endif
        T* stage1_res = new T[ntiles];

        #pragma omp parallel
        {
            // Step 1 - reducing each tile
            %if openmp:
            int itile = omp_get_thread_num();
            %else:
            int itile = 0;
            %endif
            int last_tile = ntiles - 1;
            int tile_size = (N / ntiles);
            int last_tile_size = N - tile_size * last_tile;
            int cur_tile_size = itile == ntiles - 1 ? last_tile_size : tile_size;
            int cur_start_idx = itile * tile_size;

            stage1_res[itile] = reduce<T>(cur_start_idx, cur_tile_size, initial_val${call_extra});
            #pragma omp barrier

            #pragma omp single
            ans = reduce_one_ar<T>(0, ntiles, initial_val, stage1_res);
        }
        delete[] stage1_res;
    }
    return ans;
}

PYBIND11_MODULE(${name}, m) {
    m.def("${name}", [](long n, ${type} initial${pyb_args}){
        return reduce_all(n, initial${pyb_call});
    });
}

'''

scan_c_template = '''

template<typename T>
T combine(T a, T b){
    return ${scan_expr};
}


template<typename T>
T reduce( T* ary, int offset, int n, T initial_val${args_in_extra}){
    T a, b, temp;
    temp = initial_val;

    for (int i = offset; i < (n + offset); i++){
        a = temp;
        b = ${scan_input_expr_call};

        temp = combine<T>(a, b);
    }
    return temp;
}

template <typename T>
void excl_scan_wo_ip_exp( T* ary, T* out, int N, T initial_val){
    if (N > 0){
        T a, b, temp;
        temp = initial_val;
        
        for (int i = 0; i < N - 1; i++){
            a = temp;
            b = ary[i];
            out[i] = temp;
            temp = combine<T>(a, b);
        }
        out[N - 1] = temp;
    }
}


template <typename T>
void incl_scan( T* ary, int offset, int cur_buf_size, int N, T initial_val${args_extra}){
    if (N > 0){
        T a, b, carry, prev_item, item;
        carry = initial_val;

        for (int i = offset; i < (cur_buf_size + offset); i++){
            a = carry;
            b = ${scan_input_expr_call};
            prev_item = carry;
            carry = combine<T>(a, b);
            item = carry;
            
            ${scan_output_expr_call};
        }
    }
}


template <typename T>
void scan( T* ary, long N, T initial_val${args_extra}){
    if (N > 0){
        %if openmp:
        int ntiles = omp_get_max_threads();
        %else:
        int ntiles = 1;
        %endif
        T* stage1_res = new T[ntiles];
        T* stage2_res = new T[ntiles];

        #pragma omp parallel
        {
            // Step 1 - reducing each tile
            %if openmp:
            int itile = omp_get_thread_num();
            %else:
            int itile = 0;
            %endif
            int last_tile = ntiles - 1;
            int tile_size = (N / ntiles);
            int last_tile_size = N - tile_size * last_tile;
            int cur_tile_size = itile == ntiles - 1 ? last_tile_size : tile_size;
            int cur_start_idx = itile * tile_size;

            stage1_res[itile] = reduce<T>(ary, cur_start_idx, cur_tile_size, initial_val${call_in_extra});
            #pragma omp barrier

            #pragma omp single
            excl_scan_wo_ip_exp<T>(stage1_res, stage2_res, ntiles, initial_val);

            incl_scan<T>(ary, cur_start_idx, cur_tile_size, N, stage2_res[itile]${call_extra});
        }
        delete[] stage1_res;
        delete[] stage2_res;
        py::print(ary);
    }
}



PYBIND11_MODULE(${name}, m) {
    m.def("${name}", [](py::array_t<${type}> x, long n, ${type} initial${pyb_args}){
        return scan((${type}*) x.request().ptr, n, initial${pyb_call});
    });
}
'''