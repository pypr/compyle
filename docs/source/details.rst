Using Compyle
==============

In this section we provide more details on the compyle package and how it can be
used.

An overview of functionality
-----------------------------

The functionality provided falls into two broad categories,

- Common parallel algorithms that will work across backends. This includes,
  elementwise operations, reductions, and prefix-sums/scans.

- Specific support to run code on a particular backend. This is for code that
  will only work on one backend by definition. This is necessary in order to
  best use different hardware and also use differences in the particular
  backend implementations. For example, the notion of local (or shared) memory
  only has meaning on a GPGPU. In this category we provide support to compile
  and execute Cython code, and also create and execute a GPU kernel.

In addition there is common functionality to perform type annotations. At a
lower level, there are code translators (transpilers) that handle generation
of Cython and C code from annotated Python code. Technically these transpilers
can be reused by users to do other things but we only go over the higher level
tools in this documentation. All the code is fairly extensively tested and
developed using a test-driven approach. In fact, a good place to see examples
are the tests.

We now go into the details of each of these so as to provide a high-level
overview of what can be done with Compyle.

Annotating functions
---------------------

The first step in getting started using Compyle is to annotate your functions and
also declare variables in code.

Annotation is provided by a simple decorator, called ``annotate``. One can
declare local variables inside these functions using ``declare``. A simple
example serves to illustrate these::


  @annotate(i='int', x='floatp', return_='float')
  def f(i, x):
      return x[i]*2.0

  @annotate(i='int', floatp='x, y', return_='float')
  def g(i, x, y):
      return f(i, x)*y[i]


Note that for convenience ``annotate``, accepts types and variable names in
two different ways, which you can use interchangeably.

1. You can simply use ``var_name=type_str``, or ``var_name=type`` where the
   type is from the ``compyle.types`` module.

2. You can instead use ``type_name='x, y, z'``, which is often very
   convenient. The order of the variables is not important and need not match
   with the order in which they are declared.

You can use ``return_=type``, where ``type`` is an appropriate type or
standard string representing one of the types. If the return type is not
specified it assumes a ``void`` return.


The definitions of the various standard types is in ``compyle.types.TYPES``. Some
are listed below:

- ``'float', 'double', 'int', 'long', 'uint', 'ulong'``: etc. are exactly as
  you would expect.

- ``'doublep'`` would refer to a double pointer, i.e. ``double*`` and
  similarly for anything with a ``p`` at the end.

- ``gdoublep`` would be a ``global doublep``, which makes sense with OpenCL
  where you would have ``__global double* xx``. The global address space
  specification is ignored when Cython code is generated, so this is safe to
  use with Cython code too.

- ``ldoublep`` would be equivalent to ``__local double*`` in OpenCL, for local
  memory. Again this address space qualifier is ignored in Cython.

All these types are available in the ``compyle.types`` module namespace also for
your convenience. The ``int, float, long`` types are accessible as ``int_,
float_, long_`` so as not to override the default Python types. For example
the function ``f`` in the above could also have been declared like so::

  from compyle.types import floatp, float_, int_

  @annotate(i=int_, x=floatp, return_=float_)
  def f(i, x):
      return x[i]*2.0


One can also use custom types (albeit with care) by using the
``compyle.typs.KnownType`` class. This is convenient in other scenarios where you
could potentially pass instances/structs to a function. We will discuss this
later but all of the basic types discussed above are all instances of
``KnownType``.

Compyle actually supports Python3 style annotations but only for the function
arguments and NOT for the local variables. The only caveat is you must use the
types in ``compyle.types``, i.e. you must use ``KnownType`` instances as the
types for things to work.

JIT transpilation
-----------------

Compyle also support just-in-time transpilation when annotations of a function
are not provided. These functions are annotated at runtime when the
call arguments are passed. The generated kernel and annotated functions
are then cached with the types of the call arguments as key. Thus,
the function ``f`` defined in the previous section can also be defined
as follows::

    @annotate
    def f(i, x):
        return x[i]*2.0

While using in-built functions such as ``sin``, ``cos``, ``abs`` etc. it is
recommended that you store the value in a variable or appropriate type
before returning it. If not the return type will default to ``double``.
For example,::

    @annotate
    def f(i, x):
        return abs(x[i])

This will set the return type of function ``f`` to the default
type, ``double`` even when ``x`` is an array of integers.
To avoid this problem, one could define ``f`` instead as,::

    @annotate
    def f(i, x):
        y = declare('int')
        y = abs(x[i])
        return y

Currently JIT support is only limited to the common parallel algorithms explained
in a later section.

Declaring variables
-------------------

In addition to annotating the function arguments and return types, it is
important to be able to declare the local variables. We provide a simple
``declare`` function that lets us do this. One again, a few examples serve to
illustrate this::

  i = declare('int')
  x = declare('float')
  u, v = declare('double', 2)

Notice the last one where we passed an additional argument of the number of
types we want. This is really done to keep this functional in pure Python so
that your code executes on Python also.  In Cython these would produce::

  cdef int i
  cdef float x
  cdef double u, v

On OpenCL this would produce the equivalent::

  int i;
  float x;
  double u, v;

Technically one could also write::

  f = declare('float4')

but clearly this would only work on OpenCL, however, you can definitely
declare other variables too!

Note that in OpenCL/Cython code if you do not declare a variable, it is
automatically declared as a ``double`` to prevent compilation errors.

We often also require arrays, ``declare`` also supports this, for example
consider these examples::

  r = declare('matrix(3)')
  a = declare('matrix((2, 2))')
  u, v = declare('matrix(2)', 2)

This reduces to the following on OpenCL::

  double r[3];
  double a[3][3];
  double u[2], v[2];

Note that this will only work with fixed sizes, and not with dynamic sizes. As
we pointed out earlier, dynamic memory allocation is not allowed. Of course
you could easily do this with Cython code but the declare syntax does not
allow this.

If you want non-double matrices, you can simply pass a type as in::

  a = declare('matrix((2, 2), "int")')

Which would result in::

  int a[2][2];

As you can imagine, just being able to do this opens up quite a few
possibilities.  You could also do things like this::

  xloc = declare('LOCAL_MEM matrix(128)')

which will become in OpenCL::

  LOCAL_MEM double xloc[128];

The ``LOCAL_MEM`` is a special variable that expands to the appropriate flag
on OpenCL or CUDA to allow you to write kernels once and have them run on
either OpenCL or CUDA. These special variables are discussed later below.

Writing the functions
----------------------

All of basic Python is supported. As you may have seen in the examples, you
can write code that uses the following:

- Indexing (only positive indices please).
- Conditionals with if/elif/else.
- While loops.
- For loops with the ``for var in range(...)`` construction.
- Nested fors.
- Ternary operators.

This allows us to write most numerical code. Fancy slicing etc. are not
supported, numpy based slicing and striding are not supported. You are
supposed to write these out elementwise. The idea is to keep things simple.
Yes, this may make things verbose but it does keep our life simple and
eventually yours too.

Do not create any Python data structures in the code unless you do not want to
run the code on a GPU. No numpy arrays can be created, also avoid calling any
numpy functions as these will NOT translate to any GPU code. You have to write
what you need by hand. Having said that, all the basic math functions and
symbols are automatically available. Essentially all of ``math`` is available.
All of the ``math.h`` constants are also available for use.

If you declare a global constant it will be automatically defined in the
generated code.  For example::

  MY_CONST = 42

  @annotate(x='double', return_='double')
  def f(x):
     return x + MY_CONST


The ``MY_CONST`` will be automatically injected in your generated code.

Now you may wonder about how you can call an external library that is not in
``math.h``. Lets say you have an external CUDA library, how do you call that?
We have a simple approach for this which we discuss later. We call this an
``Extern`` and discuss it later.


.. _PyOpenCL: https://documen.tician.de/pyopencl/
.. _Cython: http://www.cython.org
.. _PySPH: http://pysph.readthedocs.io

Common parallel algorithms
---------------------------

Compyle provides a few very powerful parallel algorithms. These are all directly
motivated by Andreas Kloeckner's PyOpenCL_ package. On the GPU they are
wrappers on top of the functionality provided there. These algorithms make it
possible to implement scalable algorithms for a variety of common numerical
problems. In PySPH_ for example all of the GPU based nearest neighbor finding
algorithms are written with these fundamental primitives and scale very well.

All of the following parallel algorithms allow choice of a suitable backend
and take a keyword argument to specify this backend. If no backend is provided
a default is chosen from the ``compyle.config`` module. You can get the global
config using::

  from compyle.config import get_config

  cfg = get_config()
  cfg.use_openmp = True
  cfg.use_opencl = True

etc. The following are the parallel algorithms available from the
``compyle.parallel`` module.

``Elementwise``
~~~~~~~~~~~~~~~

This is also available as a decorator ``elementwise``. One can pass it an
annotated function and an optional backend. The elementwise processes every
element in the second argument to the function. The elementwise basically
passes the function an index of the element it is processing and parallelizes
the calls to this automatically. If you are familiar with writing GPU kernels,
this is the same thing except the index is passed along to you.

Here is a very simple example that shows how this works for a case where we
compute ``y = a*sin(x) + b`` where ``y, a, x, b`` are all numpy arrays but let
us say we want to do this in parallel::

  import numpy as np
  from compyle.api import annotate, Elementwise, get_config

  @annotate(i='int', doublep='x, y, a, b')
  def axpb(i, x, y, a, b):
      y[i] = a[i]*sin(x[i]) + b[i]

  # Setup the input data
  n = 1000000
  x = np.linspace(0, 1, n)
  y = np.zeros_like(x)
  a = np.random.random(n)
  b = np.random.random(n)

  # Use OpenMP
  get_config().use_openmp = True

  # Now run this in parallel with Cython.
  backend = 'cython'
  e = Elementwise(axpb, backend=backend)
  e(x, y, a, b)

This will call the ``axpb`` function in parallel and if your problem is large
enough will effectively scale on all your cores.  Its as simple as that.

Now let us say we want to run this with OpenCL. The only issue with OpenCL is
that the data needs to be sent to the GPU. This is transparently handled by a
simple ``Array`` wrapper that handles this for us automatically. Here is a
simple example building on the above::

  from compyle.api import wrap

  backend = 'opencl'
  x, y, a, b = wrap(x, y, a, b, backend=backend)

What this does is to wrap each of the arrays and also sends the data to the
device. ``x`` is now an instance of ``compyle.array.Array``, this simple
class has two attributes, ``data`` and ``dev``. The first is the original data
and the second is a suitable device array from PyOpenCL/PyCUDA depending on
the backend. To get data from the device to the host you can call ``x.pull()``
to push data to the device you can call ``x.push()``.

Now that we have this wrapped we can simply do::

  e = Elementwise(axpb, backend=backend)
  e(x, y, a, b)

We do not need to change any of our other code.  As you can see this is very convenient.

Here is all the code put together::

  import numpy as np
  from compyle.api import annotate, Elementwise, get_config, wrap

  @annotate(i='int', doublep='x, y, a, b')
  def axpb(i, x, y, a, b):
      y[i] = a[i]*sin(x[i]) + b[i]

  # Setup the input data
  n = 1000000
  x = np.linspace(0, 1, n)
  y = np.zeros_like(x)
  a = np.random.random(n)
  b = np.random.random(n)

  # Turn on OpenMP for Cython.
  get_config().use_openmp = True

  for backend in ('cython', 'opencl'):
      xa, ya, aa, ba = wrap(x, y, a, b, backend=backend)
      e = Elementwise(axpb, backend=backend)
      e(xa, ya, aa, ba)

This will run the code on both backends! We use the for loop just to show that
this will run on all backends! The ``axpb.py`` example shows this for a
variety of array sizes and plots the performance.


``Reduction``
~~~~~~~~~~~~~~~

The ``compyle.parallel`` module also provides a ``Reduction`` class which can be
used fairly easily. Using it is a bit complex, a good starting point for this
is the documentation of PyOpenCL_, here
https://documen.tician.de/pyopencl/algorithm.html#module-pyopencl.reduction

The difference from the PyOpenCL implementation is that the ``map_expr`` is a
function rather than a string.

We provide a couple of simple examples to illustrate the above. The first
example is to find the sum of all elements of an array::

  x = np.linspace(0, 1, 1000)/1000
  x = wrap(x, backend=backend)

  r = Reduction('a+b', backend=backend)
  result = r(x)

Here is an example of a function to find the minimum of an array::

  x = np.linspace(0, 1, 1000)/1000
  x = wrap(x, backend=backend)

  r = Reduction('min(a, b)', neutral='INFINITY', backend=backend)
  result = r(x)

Here is a final one with a map expression thrown in::

  from math import cos, sin
  x = np.linspace(0, 1, 1000)/1000
  y = x.copy()
  x, y = wrap(x, y, backend=backend)

  @annotate(i='int', doublep='x, y')
  def map(i=0, x=[0.0], y=[0.0]):
      return cos(x[i])*sin(y[i])

  r = Reduction('a+b', map_func=map, backend=backend)
  result = r(x, y)

As you can see this is faithful to the PyOpenCL implementation with the only
difference that the ``map_expr`` is actually a nice function. Further, this
works on all backends, even on Cython.


``Scan``
~~~~~~~~~~

Scans are generalizations of prefix sums / cumulative sums and can be used as
building blocks to construct a number of parallel algorithms. These include
but not are limited to sorting, polynomial evaluation, and tree operations.
Blelloch's literature on prefix sums (`Prefix Sums and Their Applications
<https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf>`_) has many more examples and
is a recommended read before using scans. The ``compyle.parallel`` module
provides a ``Scan`` class which can be used to develop and execute such scans.
The scans can be run on GPUs using the OpenCL or CUDA backend or on CPUs using
either the OpenCL or Cython backend.

The scan semantics in compyle are similar to those of the GenericScanKernel in
PyOpenCL
(https://documen.tician.de/pyopencl/algorithm.html#pyopencl.scan.GenericScanKernel). Similar
to the case for reduction, the main differences from the PyOpenCL implementation
are that the expressions (`input_expr`, `segment_expr`, `output_expr`) are all
functions rather than strings.

The following examples demonstrate how scans can be used in compyle. The first
example is to find the cumulative sum of all elements of an array::

  ary = np.arange(10000, dtype=np.int32)
  ary = wrap(ary, backend=backend)

  @annotate(i='int', ary='intp', return_='int')
  def input_expr(i, ary):
      return ary[i]

  @annotate(int='i, item', ary='intp')
  def output_expr(i, item, ary):
      ary[i] = item

  scan = Scan(input_expr, output_expr, 'a+b', dtype=np.int32,
              backend=backend)
  scan(ary=ary)
  ary.pull()

  # Result = ary.data

Here is a more complex example of a function that finds the unique elements in
an array::

  ary = np.random.randint(0, 100, 1000, dtype=np.int32)
  unique_ary = np.zeros(len(ary.data), dtype=np.int32)
  unique_ary = wrap(unique_ary, backend=backend)
  unique_count = np.zeros(1, dtype=np.int32)
  unique_count = wrap(unique_count, backend=backend)
  ary = wrap(ary, backend=backend)

  @annotate(i='int', ary='intp', return_='int')
  def input_expr(i, ary):
      if i == 0 or ary[i] != ary[i - 1]:
          return 1
      else:
          return 0

  @annotate(int='i, prev_item, item, N', ary='intp',
            unique='intp', unique_count='intp')
  def output_expr(i, prev_item, item, N, ary, unique, unique_count):
      if item != prev_item:
          unique[item - 1] = ary[i]
      if i == N - 1:
          unique_count[0] = item

  scan = Scan(input_expr, output_expr, 'a+b', dtype=np.int32, backend=backend)
  scan(ary=ary, unique=unique_ary, unique_count=unique_count)
  unique_ary.pull()
  unique_count.pull()
  unique_count = unique_count.data[0]
  unique_ary = unique_ary.data[:unique_count]

  # Result = unique_ary

The following points highlight some important details and quirks about using
scans in compyle:

1. The scan call does not return anything. All output must be handled manually.
   Usually this involves writing the results available in ``output_expr``
   (``prev_item``, ``item`` and ``last_item``) to an array.
2. ``input_expr`` might be evaluated multiple times. However, it can be assumed
   that ``input_expr`` for an element or index ``i`` is not evaluated again
   after the output expression ``output_expr`` for that element is
   evaluated. Therefore, it is safe to write the output of a scan back to an
   array also used for the input like in the first example.
3. (For PyOpenCL users) If a segmented scan is used, unlike PyOpenCL where the
   ``across_seg_boundary`` is used to handle the segment logic in the scan
   expression, in compyle the logic is handled automatically. More specifically,
   using ``a + b`` as the scan expression in compyle is equivalent to using
   ``(across_seg_boundary ? b : a + b)`` in PyOpenCL.


Debugging
----------

Debugging can be a bit difficult with multiple different architectures and
backends. One convenience that compyle provides is that the generated sources
can be inspected. All the parallel algorithms (``Elementwise, Reduction,
Scan``) provide a ``.source`` or ``.all_source`` attribute that contains the
source.  For example say you have the following::

  e = Elementwise(axpb, backend=backend)
  e(x, y, a, b)

You can examine the source generated for your functions using::

  e.source

This is probably most useful for end users. For those more curious, all of the
source generated and used for the complete elementwise (or other) parallel
algorithm can be seen using::

  e.all_source

This code can be rather long and difficult to read so use this only if you
really need to see the underlying code from PyOpenCL or PyCUDA. On the GPU
this will often include multiple kernels as well. Note that on CUDA the
``all_source`` does not show all of the sources as PyCUDA currently does not
make it easy to inspect the code.


Abstracting out arrays
-----------------------

As discussed in the section on Elementwise operations, different backends need
to do different things with arrays. With OpenCL/CUDA one needs to send the
array to the device. This is transparently managed by the
``compyle.array.Array`` class. It is easiest to use this transparently with
the ``wrap`` convenience function as below::

  x = np.linspace(0, 1, 1000)/1000
  y = x.copy()
  x, y = wrap(x, y, backend=backend)

Thus these, new arrays can be passed to any operation and is handled transparently.


Choice of backend and configuration
------------------------------------

The ``compyle.config`` module provides a simple ``Configuration`` class that
is used internally in Compyle to set things like the backend (Cython,
OpenCL/CUDA), and some common options like profiling, turning on OpenMP, using
double on the GPU etc.  Here is an example of the various options::

  from compyle.config import get_config
  cfg = get_config()
  cfg.use_double
  cfg.profile
  cfg.use_opencl
  cfg.use_openmp

If one wants to temporarily set an option and perform an action, one can do::

  from compyle.config import use_config

  with use_config(use_openmp=False):
     ...

Here everything within the ``with`` clause will be executed using the
specified option and once the clause is exited, the previous settings will be
restored.  This can be convenient.

Templates
----------

When creating libraries, it is useful to be able to write a function as a
"template" where the code can be generated depending on various user options.
Compyle facilitates this by using Mako_ templates. We provide a convenient
``compyle.template.Template`` class which can be used for this purpose. A
trivial and contrived example demonstrates its use below. The example sets any
number of given arrays to a constant value::


    from compyle.types import annotate
    from compyle.template import Template

    class SetConstant(Template):
        def __init__(self, name, arrays):
            super(SetConstant, self).__init__(name=name)
            self.arrays = arrays

        def my_func(self, value):
            '''The contents of this function are directly injected.
            '''
            tmp = sin(value)

        def extra_args(self):
            return self.arrays, {'doublep': ','.join(self.arrays)}

        @annotate(i='int', value='double')
        def template(self, i, value):
            '''Set the arrays to a constant value.'''
            '''
            ${obj.inject(obj.my_func)}
            % for arr in obj.arrays:
            ${arr}[i] = tmp
            % endfor
            '''

    set_const = SetConstant('set_const', ['x', 'y', 'z']).function
    print(set_const.source)

This will print out this::

  def set_const(i, value, x, y, z):
       """Set arrays to constant.
       """
       tmp = sin(value)

       x[i] = tmp
       y[i] = tmp
       z[i] = tmp


This is obviously a trivial example but the idea is that one can create fairly
complex templated functions that can be then transpiled and used in different
cases. The key point here is the ``template`` method which should simply
create a string which is rendered using Mako_ and then put into a function.
The ``extra_args`` method allows us to configure the arguments used by the
function. The mako template can use the name ``obj`` which is ``self``. The
``obj.inject`` method allows one to literally inject any function into the
body of the code with a suitable level of indentation. Of course normal mako
functionality is available to do a variety of things.


.. _Mako: https://www.makotemplates.org/

Low level functionality
-----------------------

In addition to the above, there are also powerful low-level functionality that
is provided in ``compyle.low_level``.


``Kernel``
~~~~~~~~~~~

The ``Kernel`` class allows one to execute a pure GPU kernel. Unlike the
Elementwise functionality above, this is specific to OpenCL/CUDA and will not
execute via Cython. What this class lets one do is write low-level kernels
which are often required to extract the best performance from your hardware.

Most of the functionality is exactly the same, one declares functions and
annotates them and then passes a function to the ``Kernel`` which calls this
just as we would a normal OpenCL kernel for example. The major advantage is
that all your code is pure Python. Here is a simple example::

   from compyle.api import annotate, wrap, get_config
   from compyle.low_level import Kernel, LID_0, LDIM_0, GID_0
   import numpy as np

   @annotate(x='doublep', y='doublep', double='a,b')
   def axpb(x, y, a, b):
       i = declare('int')
       i = LDIM_0*GID_0 + LID_0
       y[i] = a*sin(x[i]) + b

   x = np.linspace(0, 1, 10000)
   y = np.zeros_like(x)
   a = 2.0
   b = 3.0

   get_config().use_opencl = True
   x, y = wrap(x, y)

   k = Kernel(axpb)
   k(x, y, a, b)

This is the same Elementwise kernel equivalent from the first example at the
top but written as a raw kernel. Notice that ``i`` is not passed but computed
using ``LDIM_0, GID_0 and LID_0`` which are automatically made available on
OpenCL/CUDA. In addition to these the function ``local_barrier`` is also
available. Internally these are ``#defines`` that are like so on OpenCL::

   #define LID_0 get_local_id(0)
   #define LID_1 get_local_id(1)
   #define LID_2 get_local_id(2)

   #define GID_0 get_group_id(0)
   #define GID_1 get_group_id(1)
   #define GID_2 get_group_id(2)

   #define LDIM_0 get_local_size(0)
   #define LDIM_1 get_local_size(1)
   #define LDIM_2 get_local_size(2)

   #define GDIM_0 get_num_groups(0)
   #define GDIM_1 get_num_groups(1)
   #define GDIM_2 get_num_groups(2)

   #define local_barrier() barrier(CLK_LOCAL_MEM_FENCE);

On CUDA, these are mapped to the equivalent ::

   #define LID_0 threadIdx.x
   #define LID_1 threadIdx.y
   #define LID_2 threadIdx.z

   #define GID_0 blockIdx.x
   #define GID_1 blockIdx.y
   #define GID_2 blockIdx.z

   #define LDIM_0 blockDim.x
   #define LDIM_1 blockDim.y
   #define LDIM_2 blockDim.z

   #define GDIM_0 gridDim.x
   #define GDIM_1 gridDim.y
   #define GDIM_2 gridDim.z

   #define local_barrier() __syncthreads();


In fact these are all provided by the ``_cluda.py`` in PyOpenCL and PyCUDA.
These allow us to write CUDA/OpenCL agnostic code from Python.

One may also pass local memory to such a kernel, this trivial example
demonstrates this::


   from compyle.api import annotate
   from compyle.low_level import (
       Kernel, LID_0, LDIM_0, GID_0, LocalMem, local_barrier
   )
   import numpy as np

   @annotate(gdoublep='x',  ldoublep='xl')
   def f(x, xl):
       i, thread_id = declare('int', 2)
       thread_id = LID_0
       i = GID_0*LDIM_0 + thread_id

       xl[thread_id] = x[i]
       local_barrier()


   x = np.linspace(0, 1, 10000)

   get_config().use_opencl = True
   x = wrap(x)
   xl = LocalMem(1)

   k = Kernel(f)
   k(x, xl)

This kernel does nothing useful and is just meant to demonstrate how one can
allocate and use local memory. Note that here we "allocated" the local memory
on the host and are passing it in to the Kernel. The local memory is allocated
as ``LocalMem(1)``, this implicitly means allocate the required size in
multiples of the size of the type and the work group size. Thus the allocated
memory is ``work_group_size * sizeof(double) * 1``. This is convenient as very
often the exact work group size is not known.

A more complex and meaningful example is the ``vm_kernel.py`` example that is
included with Compyle.


``Cython``
~~~~~~~~~~~

Just like the ``Kernel`` we also have a ``Cython`` class to run pure Cython
code. Here is an example of its usage::

  from compyle.config import use_config
  from compyle.types import annotate
  from compyle.low_level import Cython, nogil, parallel, prange

  import numpy as np

  @annotate(n='int', doublep='x, y', a='double')
  def cy_ex(x, y, a, n):
      i = declare('int')
      with nogil, parallel():
          for i in prange(n):
              y[i] = x[i]*a

  n = 1000
  x = np.linspace(0, 1, n)
  y = np.zeros_like(x)
  a = 2.0

  with use_config(use_openmp=True):
      cy = Cython(cy_ex)

   cy(x, y, a, n)

If you look at the above code, we are effectively writing Cython code but
compiling it and calling it in the last two lines. Note the use of the
``nogil, parallel`` and ``prange`` functions which are also provided in the
``low_level`` module. As you can see it is just as easy to write Cython code
and have it execute in parallel.


Externs
~~~~~~~

The ``nogil, parallel`` and ``prange`` functions we see in the previous
section are examples of external functionality. Note that these have no
straight-forward Python analog or implementation. They are implemented as
Externs. This functionality allows us to link to external code opening up many
interesting possibilities.

Note that as far as Compyle is concerned, we need to know if a function needs to
be wrapped or somehow injected. Externs offer us a way to cleanly inject
external function definitions and use them. This is useful for example when
you need to include an external CUDA library.

Let us see how the ``prange`` extern is internally defined::

  from compyle.extern import Extern

  class _prange(Extern):
      def link(self, backend):
          # We don't need to link to anything to get prange working.
          return []

      def code(self, backend):
          if backend != 'cython':
              raise NotImplementedError('prange only available with Cython')
          return 'from cython.parallel import prange'

      def __call__(self, *args, **kw):
          # Ignore the kwargs.
          return range(*args)

  prange = _prange()


The Extern object has two important methods, ``link`` and ``code``. The
``__call__`` interface is provided just so this can be executed with pure
Python. The link returns a list of link args, these are currently ignored
until we figure out a good test/example for this. The ``code`` method returns
a suitable line of code inserted into the generated code. Note that in this
case it just performs a suitable import.

Thus, with this feature we are able to connect Compyle with other libraries.
This functionality will probably evolve a little more as we gain more experience
linking with other libraries. However, we have a clean mechanism for doing so
already in-place.
