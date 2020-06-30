Molecular Dynamics Example
--------------------------

We have 3 implementations of a simple molecular dynamics simulation
of an N body problem in Lennard Jones potential. The first implementation
is a simple :math:`O(N^2)` implementation that can be found in
:code:`md_simple.py`. The second implementation is using nearest neighbor
searching to reduce the complexity to :math:`O(N)` and can be
found in :code:`md_nnps.py`.

We also have two different implementations of nearest neighbor search
algorithms, one using a radix sort on GPU and numpy sort on CPU
and the other using a native counting sort implementation. The counting
sort version is about 30% faster. Both these implementations can be
found in :code:`nnps.py`.

This example has been discussed at length in 
`this <http://procbuild.scipy.org/download/prabhuramachandran-compyle>`_ 
SciPy 2020 paper.
Following commands can be used to reproduce the performance results
shown in the paper.

+------------------+---------------------------------------------------------------+
| Figure 2         | `python performance_comparison.py -c omp_comp --nnps simple`  |
+------------------+---------------------------------------------------------------+
| Figure 3         | `python performance_comparison.py -c gpu_comp --nnps simple`  |
+------------------+---------------------------------------------------------------+
| Figure 4 & 5     | `python performance_comparison.py -c gpu_comp`                |
+------------------+---------------------------------------------------------------+
| Figure 6 & 7     | `python performance_comparison.py -c comp_algo`               |
+------------------+---------------------------------------------------------------+
| Figure 8         | `cd 3D && python performance_comparison.py --use-count-sort`  |
+------------------+---------------------------------------------------------------+

To generate energy plots for HooMD and Compyle implementations, run the script
:code:`3D/compare_results.py`

Users can use the google colab notebook 
`here <https://colab.research.google.com/drive/1SGRiArYXV1LEkZtUeg9j0qQ21MDqQR2U?usp=sharing>`_
to play around with the example.
