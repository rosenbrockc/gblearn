"""Tests the grain boundary instance methods as well as the GB
collection methods.
"""
import pytest
from os import path
import numpy as np

@pytest.fixture
def GBCol(tmpdir):
    from gblearn.gb import GrainBoundaryCollection as GBC
    from gblearn.utility import reporoot
    gbpath = path.join(reporoot, "tests", "homer")
    root = str(tmpdir.join("homer"))
    result = GBC("homer", gbpath, root, r"ni.p(?P<gbid>\d+).out",
                 rcut=3.25, lmax=12, nmax=12, sigma=0.5)

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
    GBCol.store.restricted = False
    GBCol.store.P.restricted = False
    
    for gbid, gb in GBCol.items():
        Pfile = "pissnnl.{}.npy".format(gbid)
        model = np.load(path.join(GBCol.root, Pfile))
        GBCol.store.P[gbid] = model
        N = model.shape[1]
        gb.trim()

    return N

@pytest.mark.skip()
def test_gbids(GBCol):
    assert list(GBCol.gbfiles.keys()) == list(map(str, range(453, 460)))

    #We also need to test the case where there is *no* regex specified, so we
    #just get the file names as GB ids.
    from gblearn.gb import GrainBoundaryCollection as GBC
    from gblearn.utility import reporoot
    gbpath = path.join(reporoot, "tests", "homer")
    col = GBC("homer", gbpath,
              rcut=3.25, lmax=12, nmax=12, sigma=0.5)
    model = (["ni.p{}.out".format(i) for i in range(453, 460)] +
             ["pissnnl.{}.npy".format(i) for i in range(453, 460)] +
             ["README.md"])
    assert list(sorted(col.gbfiles.keys())) == sorted(model)

@pytest.mark.skip()
def test_gbsoap(GBCol):
    """Tests construction of grain boundary objects for each of dump files found
    in the testing directory.
    """
    assert len(GBCol.P) == 0    
        
    GBCol.soap()
    for gbid in GBCol:
        with GBCol.P[gbid] as stored:
            Pfile = "pissnnl.{}.npy".format(gbid)
            model = np.load(path.join(GBCol.root, Pfile))
            assert np.allclose(stored, model)

    #Make sure it doesn't recompute if they're all there.
    assert GBCol.soap() is GBCol.P

@pytest.mark.skip()
def test_ASR(GBCol):
    """Tests construction of ASR.
    """
    #Speed up the test by pre-loading the SOAP matrices. Their construction is
    #tested separately.
    N = _preload_soap(GBCol)
    ASR = GBCol.ASR               
    assert ASR.shape == (len(GBCol), N)

def test_LER(GBCol):
    """Tests the unique LAE extraction and GB classification to create the LER.
    """
    #Speed up the test by pre-loading the SOAP matrices. Their construction is
    #tested separately.
    N = _preload_soap(GBCol)
    eps = 0.002500
    with pytest.raises(ValueError):
        U = GBCol.U(eps)

    #Now, assign the seed for the perfect FCC lattice
    from gblearn.utility import reporoot
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

    
    
@pytest.mark.skip()
def test_gb(GB9, tmpdir):
    """Tests the basic grain boundary instance attributes and methods
    (i.e., those that don't interact with other modules).
    """
    assert len(GB9) == 644
    a = GB9.atoms
    assert a.n == 644

    fxyz = str(tmpdir.join("s9.xyz"))
    GB9.save_xyz(fxyz, "Ni")
    with open(fxyz) as f:
        tfile = f.read()
    with open("tests/gb/s9.xyz") as f:
        mfile = f.read()
    assert tfile == mfile

@pytest.mark.skip()
def test_K(GB9):
    """Tests generation of the kernel matrix for the GB.
    """
    K = GB9.K
