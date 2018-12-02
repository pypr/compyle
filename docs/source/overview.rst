An overview
==============

ComPyle allows users to execute a restricted subset of Python (almost similar
to C) on a variety of HPC platforms. Currently we support multi-core execution
using Cython, and OpenCL and CUDA for GPU devices.

Users start with code implemented in a very restricted Python syntax, this
code is then automatically transpiled, compiled and executed to run on either
one CPU core, or multiple CPU cores or on a GPU. ComPyle offers source-to-source
transpilation, making it a very convenient tool for writing HPC libraries.

ComPyle is not a magic bullet,

- Do not expect that you may get a tremendous speedup.
- Performance optimization can be hard and is platform specific. What works on
  the CPU may not work on the GPU and vice-versa. ComPyle does not do anything
  to make this aspect easier. All the issues with memory bandwidth, cache, false
  sharing etc. still remain. Differences between memory architectures of CPUs
  and GPUs are not avoided at all -- you still have to deal with it. But you can
  do so from the comfort of one simple programming language, Python.
- ComPyle makes it easy to write everything in pure Python and generate the
  platform specific code from Python. It provides a low-level tool to make it
  easy for you to generate whatever appropriate code.
- The restrictions ComPyle imposes make it easy for you to think about your
  algorithms in that context and thereby allow you to build functionality that
  exploits the hardware as you see fit.
- ComPyle hides the details of the backend to the extent possible. You can write
  your code in Python, you can reuse your functions and decompose your problem
  to maximize reuse. Traditionally you would end up implementing some code in C,
  some in Python, some in OpenCL/CUDA, some in string fragments that you put
  together. Then you'd have to manage each of the runtimes yourself, worry about
  compilation etc. ComPyle minimizes that pain.
- By being written in Python, we make it easy to assemble these building blocks
  together to do fairly sophisticated things relatively easily from the same
  language.
- ComPyle is fairly simple and does source translation making it generally
  easier to understand and debug. The core code-base is less than 7k lines of
  code.
- ComPyle has relatively simple dependencies, for CPU support it requires
  Cython_ and a C-compiler which supports OpenMP_. On the GPU you need either
  PyOpenCL_ or PyCUDA_. In addition it depends on NumPy_ and Mako_.


.. _Cython: http://www.cython.org
.. _OpenMP: http://openmp.org/
.. _PyOpenCL: https://documen.tician.de/pyopencl/
.. _PyCUDA: https://documen.tician.de/pycuda/
.. _OpenCL: https://www.khronos.org/opencl/
.. _NumPy: http://numpy.scipy.org
.. _Mako: https://pypi.python.org/pypi/Mako

While ComPyle is simple and modest, it is quite powerful and convenient. In
fact, ComPyle has its origins in PySPH_ which is a powerful Python package
supporting SPH, molecular dynamics, and other particle-based algorithms. The
basic elements of ComPyle are used in PySPH_ to automatically generate HPC code
from code written in pure Python and execute it on multiple cores, and on GPUs
without the user having to change any of their code. ComPyle generalizes this
code generation to make it available as a general tool.

.. _PySPH: http://pysph.readthedocs.io


These are the restrictions on the Python language that ComPyle poses:

- Functions with a C-syntax.
- Function arguments must be declared using either type annotation or with a
  decorator or with default arguments.
- No Python data structures, i.e. no lists, tuples, or dictionaries.
- Contiguous Numpy arrays are supported but must be one dimensional.
- No memory allocation is allowed inside these functions.
- On OpenCL no recursion is supported.
- All function calls must not use dotted names, i.e. don't use ``math.sin``,
  instead just use ``sin``. This is because we do not perform any kind of name
  mangling of the generated code to make it easier to read.

Basically think of it as good old FORTRAN.

Technically we do support structs internally (we use it heavily in PySPH_) but
this is not yet exposed at the high-level and is very likely to be supported
in the future.


Simple example
--------------

Enough talk, lets look at some code.  Here is a very simple example::

   from compyle.api import Elementwise, annotate, wrap, get_config
   import numpy as np

   @annotate(i='int', x='doublep', y='doublep', double='a,b')
   def axpb(i, x, y, a, b):
       y[i] = a*sin(x[i]) + b

   x = np.linspace(0, 1, 10000)
   y = np.zeros_like(x)
   a = 2.0
   b = 3.0

   backend = 'cython'
   get_config().use_openmp = True
   x, y = wrap(x, y, backend=backend)
   e = Elementwise(axpb, backend=backend)
   e(x, y, a, b)

This will execute the elementwise operation in parallel using OpenMP with
Cython. The code is auto-generated, compiled and called for you transparently.
The first time this runs, it will take a bit of time to compile everything but
the next time, this is cached and will run much faster.

If you just change the ``backend = 'opencl'``, the same exact code will be
executed using PyOpenCL_ and if you change the backend to ``'cuda'``, it will
execute via CUDA without any other changes to your code. This is obviously a
very trivial example, there are more complex examples available as well.

More examples
--------------

More complex examples (but still fairly simple) are available in the `examples
<https://github.com/pypr/compyle/tree/master/examples>`_ directory.

- `axpb.py <https://github.com/pypr/compyle/tree/master/examples/axpb.py>`_: the
  above example but for openmp and opencl compared with serial showing that in
  some cases serial is actually faster than parallel!

- `vm_elementwise.py
  <https://github.com/pypr/compyle/tree/master/examples/vm_elementwise.py>`_:
  shows a simple N-body code with two-dimensional point vortices. The code uses
  a simple elementwise operation and works with OpenMP and OpenCL.

- `vm_numba.py
  <https://github.com/pypr/compyle/tree/master/examples/vm_numba.py>`_: shows
  the same code written in numba for comparison. In our benchmarks, ComPyle is
  actually faster even in serial and in parallel it can be much faster when you
  use all cores.

- `vm_kernel.py
  <https://github.com/pypr/compyle/tree/master/examples/vm_kernel.py>`_: shows
  how one can write a low-level OpenCL kernel in pure Python and use that. This
  also shows how you can allocate and use local (or shared) memory which is
  often very important for performance on GPGPUs. This code will only run via
  PyOpenCL.

- `bench_vm.py
  <https://github.com/pypr/compyle/tree/master/examples/bench_vm.py>`_:
  Benchmarks the various vortex method results above for a comparison with
  numba.


Read on for more details about ComPyle.
