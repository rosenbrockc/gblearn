Examples for using `gblearn`
============================

We begin with the construction of the LER matrix. It represents each
grain boundary as a feature vector whose components are the relative
fractions of each *unique* Local Atomic Environment (LAE) in the
entire grain boundary system.

We begin by creating a :class:`~gblearn.gb.GrainBoundaryCollection`
that holds a representation of each :class:`~gblearn.gb.GrainBoundary`
in the collection and manages the calculation and storage of the
various representations that can be derived using
:mod:`~gblearn.soap`. For this code, we assume that all the Olmsted
[1]_ dump files from LAMMPS are in `/dbs/olmsted`. We tell the
framework to store all representations in the `/gbs/olmsted`
folder. Notice that we give a regular expression that matches the file
names of the LAMMPS dump files. We use a named capture group to grab
out the publication integer id from the file name. Each grain boundary
will be referred to by that id for the rest of the analysis.

.. code-block:: python

   from gblearn.gb import GrainBoundaryCollection as GBC
   olmsted = GBC("olmsted", "/dbs/olmsted", "/gbs/olmsted",
		 r"ni.p(?P<gbid>\d+).out",
                 rcut=3.25, lmax=12, nmax=12, sigma=0.5)

You will also notice that we specify the soap parameters as part of
this constructor. Now that we have the collection, we can calculate
the SOAP matrices for each grain boundary.

.. code-block:: python

   olmsted.soap()
   with olmsted.P["1"] as P:
       print(P)

Because grain boundary databases can get quite large, and SOAP
matrices can *also* get quite large, `gblearn` implements
memory-sensitive storage for the SOAP matrices. It does this using
context managers so that a SOAP matrix is read from disk and then
cleared from memory once it falls out of context. The `with` construct
shown here will load the file from disk, print it, and then remove it
from memory.

.. note:: If you are only after the Local Environment Representation,
   you won't have to worry about accessing memory-safe SOAP matrices.

Once the SOAP matrices have been calculated, we can grab the Averaged
SOAP Representation (ASR) via a property:

.. code-block:: python

   olmsted.ASR

Whenever any of the representations is accessed, it is calculated in
the background and then cached to disk automatically. To avoid
caching, specify `None` for the path to storage folder in the
:class:`~gblearn.gb.GrainBoundaryCollection` constructor. Subsequent
requests for the same representation will be served from memory/disk
cache for optimization.

Constructing the LER requires a similarity parameter `eps` that is the
cutoff for deciding when two atomic environments are similar. It is
related to the :func:`~gblearn.soap.S` similarity metric between SOAP
vectors.

.. code-block:: python

   eps = 0.0025
   LER = olmsted.LER(eps)

When you run this code, will see several progress bars as the code
runs over the grain boundary collection in the background.

1. All grain boundaries are iterated to determine a set of unique
   environments for the entire collection.
2. The collection is iterated over *again* so that every atom in each
   grain boundary can be classified with the unique LAE that it is
   *most* similar to.
3. The fraction of each type of unique LAE is computed for each grain
   boundary to form the LER vectors.

Summary: LER Construction
-------------------------

In summary, you can generate the LER for new collection of grain
boundaries using:

.. code-block:: python
		
   from gblearn.gb import GrainBoundaryCollection as GBC
   olmsted = GBC("olmsted", "/dbs/olmsted", "/gbs/olmsted",
		 r"ni.p(?P<gbid>\d+).out",
                 rcut=3.25, lmax=12, nmax=12, sigma=0.5)
   olmsted.soap()
   olmsted.LER(0.0025)

