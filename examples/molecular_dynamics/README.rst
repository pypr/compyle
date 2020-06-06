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

We have also provided a script to run performance comparison between
these implementations in `performance_comparison.py`. Users can use
the google colab notebook 
`here <https://colab.research.google.com/drive/1SGRiArYXV1LEkZtUeg9j0qQ21MDqQR2U?usp=sharing>`_.

