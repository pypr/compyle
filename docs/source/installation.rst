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

Installing the bleeding edge
----------------------------

Note that if you want the latest bleeding edge of compyle, clone the
repository and install compyle like so::

  $ git clone https://github.com/pypr/compyle
  $ cd compyle
  $ python setup.py develop
  # Or
  $ pip install -e .

If you just want the latest version and do not want to clone the repository,
you can also do::

  $ pip install https://github.com/pypr/compyle/zipball/master


.. _PyOpenCL: https://documen.tician.de/pyopencl/
.. _OpenCL: https://www.khronos.org/opencl/
.. _Cython: http://www.cython.org
.. _numpy: http://www.numpy.org
.. _PyCUDA: https://documen.tician.de/pycuda
.. _OpenMP: http://openmp.org/
.. _CuPy: https://cupy.chainer.org/


Setting up on GNU/Linux
-------------------------

This is usually very simple, just installing the standard gcc/g++ packages ought
to work. OpenMP_ is typically available but if it is not, it can be installed
with (on apt-compatible systems)::

    $ sudo apt-get install libgomp1


Installation with conda on MacOS
---------------------------------

Recent conda_ packages make the process of setup very easy on MacOS assuming
that you have the `XCode command line utilities`_ installed. Please make sure
you install this.

For example with conda-forge_ the following creates a new Python 3.8
environment with compyle installed and working with both OpenMP and OpenCL::

  $ conda create -c conda-forge -n py38 python=3.8 numpy pyopencl
  $ conda activate py38 # or a suitable such invocation
  $ pip install compyle

Note that the above implicitly installs the ``llvm-openmp`` package in the
environment which works out of the box with clang and provides OpenMP support.

.. _conda: https://docs.conda.io/
.. _conda-forge: https://conda-forge.org/
.. _XCode command line utilities: http://stackoverflow.com/questions/12228382/after-install-xcode-where-is-clang


Possible issues on MacOS
--------------------------

Ensure that you have gcc or clang installed by installing XCode. See
installing `XCode command line utilities`_ if you installed XCode but can't
find clang or gcc.

If you are getting strange errors of the form::

  lang: warning: libstdc++ is deprecated; move to libc++ with a minimum deployment target of OS X 10.9 [-Wdeprecated]
  ld: library not found for -lstdc++
  clang: error: linker command failed with exit code 1 (use -v to see invocation)

Then try this (on a bash shell)::

  $ export MACOSX_DEPLOYMENT_TARGET=10.9

And run your command again (replace the above with a suitable line on other
shells). This is necessary because your Python was compiled with an older
deployment target and the current version of XCode that you have installed is
not compatible with that. By setting the environment variable you allow
compyle to use a newer version. If this works, it is a good idea to set this
in your default environment (``.bashrc`` for bash shells) so you do not have
to do this every time. You may also do this in the compyle configuration file,
see :ref:`config`.



OpenMP on MacOS
~~~~~~~~~~~~~~~~

These instructions are a bit old and only if you are not using conda as
discussed above.

The default clang compiler available on MacOS uses an LLVM backend and does
not support OpenMP_ out of the box. There are two ways to support OpenMP. The
first involves installing the OpenMP support for clang. This can be done with
brew_ using::

  $ brew install libomp

Once that is done, it should "just work". If you get strange errors, try
setting the ``MACOSX_DEPLOYMENT_TARGET`` as shown in the previous section.

Another option is to install GCC for MacOS available on brew_ using ::

    $ brew install gcc

Once this is done, you need to use this as your default compiler. The ``gcc``
formula on brew currently ships with gcc version 9. Therefore, you can
tell Python to use the GCC installed by brew by setting::

    $ export CC=gcc-9
    $ export CXX=g++-9

Note that you still do need to have the command-line-tools for XCode
installed, otherwise the important header files are not available. See
`how-to-install-xcode-command-line-tools
<https://stackoverflow.com/questions/9329243/how-to-install-xcode-command-line-tools>`_
for more details. You may also want to set these environment variables in your
``.bashrc`` so you don't have to do this every time. You may also do this in
the compyle configuration file, see :ref:`config`.

Once you do this, compyle will automatically use this version of GCC and will
also work with OpenMP. Note that on some preliminary benchmarks, GCC's OpenMP
implementation seems about 10% or so faster than the LLVM version. Your
mileage may vary.

.. _brew: http://brew.sh/


Setting up on Windows
----------------------

Windows will work but you need to make sure you have the right compiler
installed. See this page for the details of what you need installed.

https://wiki.python.org/moin/WindowsCompilers

OpenMP will work if you have this installed. For recent Python versions
(>=3.5), install the `Microsoft Build Tools for Visual Studio 2019
<https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2019>`_


Setting up OpenCL/CUDA
-----------------------

This is too involved a topic to discuss here, instead look at the appropriate
documentation for PyOpenCL_ and PyCUDA_. Once those packages work correctly,
you should be all set. Note that if you are only using OpenCL/CUDA you do not
need to have Cython or a C/C++ compiler. Some features on CUDA require the use
of the CuPy_ library.

If you want to use OpenCL support, you will need to install the ``pyopencl``
package (``conda install -c conda-forge pyopencl`` or ``pip install
pyopencl``). For CUDA Support, you will need to install ``pycuda`` and
``cupy``. Of course this assumes you have the required hardware for this.


.. _config:

Using the configuration file
-----------------------------

Instead of setting environment variables and build options on the shell you
can have them setup using a simple configuration file.

The file is located in ``~/.compyle/config.py``. Here ``~`` is your home
directory which on Linux is ``/home/username``, on MacOS ``/Users/username``
and on Windows the location is likely ``\Users\username``. This file is
executed and certain options may be set there.

For example if you wish to set the environment variables ``CC`` and ``CXX``
you could do this in the ``config.py``::

  import os

  os.environ['CC'] = 'gcc-9'
  os.environ['CXX'] = 'g++-9'

If you are using an atypical compiler like icc, Cray, or PGI, you can set
these up here too. You may also setup custom OpenMP related flags. For
example, on a Cray system you may do the following::

  OMP_CFLAGS = ['-homp']
  OMP_LINK = ['-homp']

The ``OMP_CFLAGS`` and ``OMP_LINK`` parameters should be lists. Other packages
like pyzoltan or pysph may also use this file for customizations.
