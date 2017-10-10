# `gblearn`: Machine Learning for Grain Boundaries

[API Documentation](https://rosenbrockc.github.io/gblearn/index.html)

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
   from gblearn.gb import GrainBoundaryCollection as GBC
   olmsted = GBC("olmsted", "/dbs/olmsted", "/gbs/olmsted",
		 r"ni.p(?P<gbid>\d+).out",
                 rcut=3.25, lmax=12, nmax=12, sigma=0.5)
   olmsted.soap()
   olmsted.LER(0.0025)
```

## References

[1]: Olmsted, D. L., Foiles, S. M. & Holm, E. A. Survey of computed grain boundary properties in face-centered cubic metals: I. Grain boundary energy. Acta Mater. 57, 3694–3703 (2009).				   