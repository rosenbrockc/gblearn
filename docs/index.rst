.. gblearn documentation master file, created by
   sphinx-quickstart on Tue Oct 10 12:11:44 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

`gblearn`: Machine Learning for Grain Boundaries
================================================

Recently, we proposed a universal descriptor for grain boundaries that
has desirable mathematical properties, and which can be applied to
arbitrary grain boundaries. Using this descriptor, we were able to
create a feature matrix for machine learning based on the local atomic
environments present at the grain boundary. In addition to being
useful for predicting grain boundary energy and mobility, the method
also allows important atomic environments to be discovered for each of
the properties.

If you use this package, please cite the paper:

::
   
    @article{Rosenbrock:2017vd,
    author = {Rosenbrock, Conrad W and Homer, Eric R and Csanyi, G{\'a}bor and Hart, Gus L W},
    title = {{Discovering the building blocks of atomic systems using machine learning: application to grain boundaries}},
    journal = {npj Computational Materials},
    year = {2017},
    volume = {3},
    number = {1},
    pages = {29}
    }

To get started quickly, take a look at the :doc:`examples`, which show
how to generate the plots and model from the paper.

Workflow and Examples
---------------------

.. toctree::
   :maxdepth: 1

   examples.rst

Modules in the Package
----------------------
   
.. autosummary::
   :toctree: Modules

   gblearn.elements
   gblearn.selection
   gblearn.gb
   gblearn.soap
   gblearn.decomposition
   gblearn.io
   gblearn.lammps
   gblearn.reduce
   gblearn.utility

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
