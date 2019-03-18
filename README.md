# `gblearn`: Machine Learning for Grain Boundaries

[API Documentation](https://rosenbrockc.github.io/gblearn/index.html)
[![Build Status](https://travis-ci.org/rosenbrockc/gblearn.svg?branch=master)](https://travis-ci.org/rosenbrockc/gblearn)

Recently, we proposed a universal descriptor for grain boundaries that
has desirable mathematical properties, and which can be applied to
arbitrary grain boundaries. Using this descriptor, we were able to
create a feature matrix for machine learning based on the local atomic
environments present at the grain boundary. In addition to being
useful for predicting grain boundary energy and mobility, the method
also allows important atomic environments to be discovered for each of
the properties.

If you use this package, please cite the paper:

```   
    @article{Rosenbrock:2017vd,
    author = {Rosenbrock, Conrad W and Homer, Eric R and Csanyi, G{\'a}bor and Hart, Gus L W},
    title = {{Discovering the building blocks of atomic systems using machine learning: application to grain boundaries}},
    journal = {npj Computational Materials},
    year = {2017},
    volume = {3},
    number = {1},
    pages = {29}
    }
```

You can generate the Local Environment Representation for the Olmsted
dataset using the following code. It assumes that all the Olmsted
[1] dump files from LAMMPS are in `/dbs/olmsted`. We tell the
framework to store all representations in the `/gbs/olmsted`
folder.

```python
   # Load the perfect FCC as a seed so the LER can be constructed.
   # It assumes the the seed file is found at /seeds/"Ni.pissnnl_seed.txt"
   seed = np.loadtxt("/seeds/Ni.pissnnl_seed.txt")

   from gblearn.gb import GrainBoundaryCollection as GBC
   olmsted = GBC("olmsted", "/dbs/olmsted", "/gbs/olmsted",
		 r"ni.p(?P<gbid>\d+).out", seed=seed, padding=6.50)

   # We explicitly call :meth:`load` to parse the GB files. Then, construct
   # the SOAP representation for each GB.
   # As part of the load function, we call it with Z=28 for the nickel database,
   # and also give it a method and pattern to use
   olmsted.load(Z=28, method="cna", pattr="c_cna")

   # Calculate the SOAP representation.
   # The SOAP representation includes padding around the boundary atoms, so
   # that each atom in the GB has a full `rcut` of atoms around it.
   # The "meth: 'soap' auto trims those atoms that don't have full environments.
   olmsted.soap(rcut=3.25, lmax=12, nmax=12, sigma=0.5)

   #Now, we can finally construct the LER.
   olmsted.LER(0.0025)
```

## References

[1]: Olmsted, D. L., Foiles, S. M. & Holm, E. A. Survey of computed grain boundary properties in face-centered cubic metals: I. Grain boundary energy. Acta Mater. 57, 3694–3703 (2009).				   
