ComPyle: execute a subset of Python on HPC platforms
======================================================

|Travis Status| |Appveyor Status| |Documentation Status|


.. |Travis Status| image:: https://travis-ci.org/pypr/compyle.svg?branch=master
    :target: https://travis-ci.org/pypr/compyle
.. |Appveyor Status| image:: https://ci.appveyor.com/api/projects/status/isg727d5ofn69rrm?svg=true
    :target: https://ci.appveyor.com/project/prabhuramachandran/compyle
.. |Documentation Status| image:: https://readthedocs.org/projects/compyle/badge/?version=latest
    :target: https://compyle.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

ComPyle allows users to execute a restricted subset of Python (almost similar
to C) on a variety of HPC platforms. Currently we support multi-core execution
using Cython, and OpenCL and CUDA for GPU devices.

Users start with code implemented in a very restricted Python syntax, this code
is then automatically transpiled, compiled and executed to run on either one CPU
core, or multiple CPU cores (via OpenMP_) or on a GPU. CPy offers
source-to-source transpilation, making it a very convenient tool for writing HPC
libraries.

Some simple yet powerful parallel utilities are provided which can allow you to
solve a remarkably large number of interesting HPC problems.

ComPyle also features JIT transpilation if you wish making it easy to use.

Documentation is available at: https://compyle.readthedocs.io

While ComPyle seems simple it is not a toy and is used heavily by the PySPH_
project where ComPyle has its origins.

.. _PySPH: https://github.com/pypr/pysph


Installation
-------------

ComPyle is itself pure Python but depends on numpy_ and requires either Cython_
or PyOpenCL_ or PyCUDA_ along with the respective backends of a C/C++ compiler,
OpenCL and CUDA. If you are only going to execute code on a CPU then all you
need is Cython.

You should be able to install ComPyle by doing::

  $ pip install compyle


.. _PyOpenCL: https://documen.tician.de/pyopencl/
.. _OpenCL: https://www.khronos.org/opencl/
.. _Cython: http://www.cython.org
.. _numpy: http://www.numpy.org
.. _OpenMP: http://openmp.org/
.. _PyCUDA: https://documen.tician.de/pycuda/

A simple example
----------------

Here is a very simple example::

   from compyle.api import Elementwise, annotate, wrap, get_config
   import numpy as np

   @annotate
   def axpb(i, x, y, a, b):
       y[i] = a*sin(x[i]) + b

   x = np.linspace(0, 1, 10000)
   y = np.zeros_like(x)
   a, b = 2.0, 3.0

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


Examples
---------

Some simple examples and benchmarks are available in the `examples
<https://github.com/pypr/compyle/tree/master/examples>`_ directory.
