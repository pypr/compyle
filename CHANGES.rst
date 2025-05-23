0.9
~~~~

* Release date: 23 May, 2025.
* Allow user to pass a cython include directory when using ``ExtModule``.
* Fix error with Cython compilation error messages not being shown.
* Fix issue with the root log level being set when building an extension module.
* Use cdivision always as that is usually the intent when using compyle.
* Add a ``COMPYLE_DEBUG`` environment variable to print debug information.
* Explicitly type any float literals for single precision to improve GPU performance.
* Fix bug with the directory where the sources were saved.
* Support for NumPy 2 and Cython 3.x.
* Drop Python 2 support.
* Do late binding so the backend can be changed.
* Fix NumPy deprecation errors.


0.8.1
~~~~~~

* Release date: 7th November, 2021.
* Fix issue with accidental file in sdist.


0.8
~~~~

* Release date: 7th November, 2021.
* Improve array module to support more numpy like functionality.
* Improve profile output so it works in a distributed setting.
* Add support for a configuration file in ~/.compyle/config.py
* Added `atomic_dec` support.
* Fix output capturing on jupyter notebooks.
* Fix issues due to ast changes in Python 3.9.x.
* Fix tests on 32bit architectures.
* Fix several bugs and issues.


0.7
~~~~

* Release date: 1st October, 2020.
* Add convenient option to profile execution of code.
* Add a convenient argument parser for scripts.
* Add easy way to see generated sources.
* Fix bug with installation of previous version.
* Fix several bugs and issues.
* Update the documentation.

0.6
~~~~

* Release date: 15th June, 2020.
* Add some non-trivial examples showcasing the package.
* Document how one can use clang + OpenMP.
* Add sorting, align, and other functions to array module.
* Support for mapping structs on a GPU with CUDA.
* Add address, cast, and address low-level functions.
* Support for mako-templates for reducing repetitive code.
* Bitwise operator support.
* Attempt to auto-declare variables when possible.
* Fix several bugs and issues.



0.5
~~~~

* Release date: 3rd, December 2018
* First public release.
* Support for elementwise, scan, and reduction operations on CPU and GPU using
  Cython, OpenCL and CUDA.
