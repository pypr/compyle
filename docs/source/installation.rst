Installation
==============

ComPyle is itself pure Python but depends on numpy_ and requires either Cython_
or PyOpenCL_ or PyCUDA_ along with the respective backends of a C/C++ compiler,
OpenCL and CUDA. If you are only going to execute code on a CPU then all you
need is Cython_. The full list of requirements is shown in the
``requirements.txt`` file on the repository.

You should be able to install ComPyle by doing::

  $ pip install compyle


Note that when executing code on a CPU, you will need to have a C/C++ compiler
that is compatible with your Python installation. In addition, if you need to
use OpenMP you will need to make sure your compiler is compatible with that.
Some additional information on this is included below.

.. _PyOpenCL: https://documen.tician.de/pyopencl/
.. _OpenCL: https://www.khronos.org/opencl/
.. _Cython: http://www.cython.org
.. _numpy: http://www.numpy.org
.. _PyCUDA: https://documen.tician.de/pycuda
.. _OpenMP: http://openmp.org/


Setting up on GNU/Linux
-------------------------

This is usually very simple, just installing the standard gcc/g++ packages ought
to work. OpenMP_ is typically available but if it is not, it can be installed
with (on apt-compatible systems)::

    $ sudo apt-get install libgomp1


Setting up on MacOS
---------------------

Ensure that you have gcc or clang installed by installing XCode. See `this
<http://stackoverflow.com/questions/12228382/after-install-xcode-where-is-clang>`_
if you installed XCode but can't find clang or gcc.


OpenMP on MacOS
~~~~~~~~~~~~~~~~

The default "gcc" available on OSX uses an LLVM backend and does not support
OpenMP_. To use OpenMP_ on OSX, you can install the GCC available on brew_ using
::

    $ brew install gcc

Once this is done, you need to use this as your default compiler. The ``gcc``
formula on brew currently ships with gcc version 8. Therefore, you can
tell Python to use the GCC installed by brew by setting::

    $ export CC=gcc-8
    $ export CXX=g++-8

Note that you still do need to have the command-line-tools for XCode installed,
otherwise the important header files are not available.

.. _brew: http://brew.sh/


Setting up on Windows
----------------------

Windows will work but you need to make sure you have the right compiler
installed. See this page for the details of what you need installed.

https://wiki.python.org/moin/WindowsCompilers

OpenMP will work if you have this installed.


Setting up OpenCL/CUDA
-----------------------

This is too involved a topic for this instead look at the appropriate
documentation for PyOpenCL_ and PyCUDA_. Once those packages work correctly, you
should be all set. Note that if you are only using OpenCL/CUDA you do not need
to have Cython or a C/C++ compiler.
