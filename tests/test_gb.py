"""Tests the grain boundary instance methods as well as the GB
collection methods.
"""
import pytest
from os import path
import numpy as np
from gblearn.utility import reporoot

@pytest.fixture
def GBCol(tmpdir):
    from gblearn.gb import GrainBoundaryCollection as GBC
    gbpath = path.join(reporoot, "tests", "homer")
    root = str(tmpdir.join("homer"))
    result = GBC("homer", gbpath, root, r"ni.p(?P<gbid>\d+).out", padding=6.50)

    from gblearn.gb import GrainBoundary
    result.load(Z=28, method="cna_z", pattr="c_cna")
    for gbid, gb in result.items():
        assert isinstance(gb, GrainBoundary)

    return result

@pytest.fixture(scope="module")
def GB9(request):
    """Returns the grain boundary atoms from the 9th sample in the
    Olmsted set.
    """
    from gblearn.lammps import Timestep
    p9 = Timestep("tests/selection/ni.p9.out")
    return p9.gb(28)

def test_properties(GBCol):
    """Tests the loading and reading of properties for a GB collection.
    """
    pdict = {str(i): i + 10.5 for i in range(453, 460)}
    model = np.array([i + 10.5 for i in range(453, 460)])
    GBCol.add_property("fromdict", values=pdict)
    assert np.allclose(model, GBCol.get_property("fromdict"))

    valfile = path.join(reporoot, "tests", "homer", "energy.txt")
    GBCol.add_property("fromfile", valfile)
    assert np.allclose(model, GBCol.get_property("fromfile"))

def test_K(GB9):
    """Tests generation of the kernel matrix for the GB.
    """
    K = GB9.K

def test_gb(GB9, tmpdir):
    """Tests the basic grain boundary instance attributes and methods
    (i.e., those that don't interact with other modules).
    """
    assert len(GB9) == 644
    a = GB9.atoms
    assert a.n == 644

    fxyz = str(tmpdir.join("s9.xyz"))
    GB9.save_xyz(fxyz, "Ni")

    from quippy import Atoms
    A = Atoms(fxyz)
    B = Atoms("tests/gb/s9.xyz")
    assert A.equivalent(B)

def _preload_soap(GBCol):
    """Preloads all the SOAP matrices into the GB collection to speed up
    computations.

    .. note:: Since the SOAP calculation includes the trimming of the atoms to
      those that fall within the padding constraints, this function also calls
      :meth:`~gblearn.gb.GrainBoundary.trim` on each grain boundary.

    Returns:
        int: dimensions of SOAP vectors that are loaded.
    """
    GBCol.restricted = False
    GBCol.repargs["soap"] = {"rcut":3.25, "lmax":12, "nmax":12, "sigma":0.5}
    GBCol.store.configure("soap", rcut=3.25., lmax=12, nmax=12, sigma=0.5)
    GBCol.store.restricted = False
    GBCol.store.P.restricted = False

    for gbid, gb in GBCol.items():
        Pfile = "pissnnl.{}.npy".format(gbid)
        model = np.load(path.join(GBCol.root, Pfile))
        GBCol.store.P[gbid] = model
        N = model.shape[1]
        gb.trim()

    return N

def test_gbids(GBCol):
    assert list(GBCol.gbfiles.keys()) == list(map(str, range(453, 460)))

    #We also need to test the case where there is *no* regex specified, so we
    #just get the file names as GB ids.
    from gblearn.gb import GrainBoundaryCollection as GBC
    gbpath = path.join(reporoot, "tests", "homer")
    col = GBC("homer", gbpath, padding=6.50)
    model = (["ni.p{}.out".format(i) for i in range(453, 460)] +
             ["pissnnl.{}.npy".format(i) for i in range(453, 460)] +
             ["README.md", "energy.txt"])
    assert list(sorted(col.gbfiles.keys())) == sorted(model)

def test_gbsoap(GBCol):
    """Tests construction of grain boundary objects for each of dump files found
    in the testing directory.
    """
    assert len(GBCol.P) == 0

    GBCol.soap(rcut=3.25, lmax=12, nmax=12, sigma=0.5)
    for gbid in GBCol:
        with GBCol.P[gbid] as stored:
            Pfile = "pissnnl.{}.npy".format(gbid)
            model = np.load(path.join(GBCol.root, Pfile))
            assert np.allclose(stored, model)

    #Make sure it doesn't recompute if they're all there.
    assert GBCol.soap(rcut=3.25, lmax=12, nmax=12, sigma=0.5) is GBCol.P

def test_gbscatter(GBCol):
    """Tests calculation of Scatter vectors.
    """
    assert len(GBCol.Scatter) == 0

    GBCol.scatter()
    for gbid in GBCol:
        with GBCol.Scatter[gbid] as stored:
            model = np.arange(10)
            assert np.allclose(stored, model)

    #Make sure it doesn't recompute if they're all there.
    assert GBCol.scatter() is GBCol.Scatter

def test_gbscattercache(GB9):
    """Tests caching of Scatter vector for a single GB.
    """
    assert GB9.Scatter == None

    GB9.scatter()
    model = np.arange(10)
    assert np.allclose(GB9.Scatter, model)

    #Make sure it doesen't recompute if the value is already cached
    assert GB9.scatter() is GB9.Scatter

def test_ASR(GBCol):
    """Tests construction of ASR.
    """
    #Speed up the test by pre-loading the SOAP matrices. Their construction is
    #tested separately.
    N = _preload_soap(GBCol)
    ASR = GBCol.ASR
    assert ASR.shape == (len(GBCol), N)

def test_uniquify(GBCol):
    """Tests the unique LAE extraction and GB classification to create the LER.
    """
    #Speed up the test by pre-loading the SOAP matrices. Their construction is
    #tested separately.
    N = _preload_soap(GBCol)
    eps = 0.002500
    with pytest.raises(ValueError):
        U = GBCol.U(eps)

    #Now, assign the seed for the perfect FCC lattice
    seed = np.loadtxt(path.join(reporoot, "tests", "elements", "Ni.pissnnl_seed.txt"))
    GBCol.seed = seed
    U = GBCol.U(eps)

    #Make sure we found the same unique LAEs as Jonathan verified.
    ukeyfile = path.join(reporoot, "tests", "unique", "soap-keys.txt")
    modelkeys = np.asarray(np.loadtxt(ukeyfile), dtype=int)
    for mkey in modelkeys:
        assert (str(mkey[0]), mkey[1]) in U["U"]

    #Make sure that we didn't mess up the indices or identifiers. The soap
    #vectors for each index that we found should match the model ones.
    uvecfile = path.join(reporoot, "tests", "unique", "soap-vecs.txt")
    modelvecs = np.loadtxt(uvecfile)
    for i, mkey in enumerate(modelkeys):
        skey = (str(mkey[0]), mkey[1])
        assert np.allclose(modelvecs[i,:], U["U"][skey])

    #Next, check that we are correctly assigning LAEs to the atoms in the GB.
    for gbid, gb in GBCol.items():
        laefile = path.join(reporoot, "tests", "unique", "LAE-{}.txt".format(gbid))
        model = np.loadtxt(laefile)
        ids = np.arange(1, len(gb) + 1).reshape((len(gb), 1))
        ours = np.hstack((ids, gb.xyz, np.array(gb.LAEs, dtype=int)))
        assert np.allclose(ours, model)

def _preload_U(GBCol, eps):
    """Preloads the set of unique vectors and the assignment of specific atoms
    to LAEs in the GB objects.
    """
    from cPickle import load
    upkl = path.join(reporoot, "tests", "unique", "U.pkl")
    with open(upkl, 'rb') as f:
        U = load(f)
    GBCol.store.U = {eps: U}
    assert isinstance(GBCol.U(eps), dict)

def test_LER(GBCol):
    """Tests construction of the LER.
    """
    #Speed up the test by pre-loading the SOAP matrices. Their construction is
    #tested separately.
    N = _preload_soap(GBCol)
    eps = 0.002500
    #We also want to pre-load the unique vectors.
    _preload_U(GBCol, eps)

    LER = GBCol.LER(eps)
    U = GBCol.U(eps)
    assert LER.shape == (len(GBCol), len(U["U"]))

    model = np.load(path.join(reporoot, "tests", "unique", "LER.pkl"))
    assert np.allclose(LER, model)

    #Make sure it doesen't recompute if they are already there
    assert GBCol.LER(eps) is GBCol.store.LER[eps]

def test_LAE(GBCol):
    """Tests the LAE property
    """
    #Speed up the test by pre-loading the SOAP matrices and the unique vectors
    N = _preload_soap(GBCol)
    eps = 0.002500
    _preload_U(GBCol, eps)

    #Check that the LAE property gives the correct SOAP vector related to it
    U = GBCol.U(eps)
    for lae, soap in enumerate(U["U"].values()):
        assert np.allclose(GBCol.LAE[lae].soap, soap)

def test_others(GBCol):
    """Tests parsing and analyzing of Grain Boundaries outside the
        original collection
    """

    N = _preload_soap(GBCol)
    eps = 0.002500
    #We also want to pre-load the unique vectors.
    _preload_U(GBCol, eps)
    import pdb; pdb.set_trace()
    seed = np.loadtxt(path.join(reporoot, "tests", "elements", "Ni.pissnnl_seed.txt"))
    GBCol.seed = seed
    GBCol.load(name="other", fname='ni.p453.out', Z=28, method="cna_z", pattr="c_cna")
    LER = GBCol.analyze_other("other", eps=eps)
    model = np.load(path.join(reporoot, "tests", "unique", "LER.pkl"))
    assert np.allclose(LER, model[0])
