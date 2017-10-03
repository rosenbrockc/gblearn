"""Tests the grain boundary instance methods as well as the GB
collection methods.
"""
import pytest
from os import path

@pytest.fixture
def GBCol(tmpdir):
    from gblearn.gb import GrainBoundaryCollection as GBC
    from gblearn.utility import reporoot
    gbpath = path.join(reporoot, "tests", "homer")
    root = str(tmpdir.join("homer"))
    return GBC("homer", gbpath, root, r"ni.p(?P<gbid>\d+).out",
               rcut=3.25, lmax=12, nmax=12, sigma=0.5)

@pytest.fixture(scope="module")
def GB9(request):
    """Returns the grain boundary atoms from the 9th sample in the
    Olmsted set.
    """
    from gblearn.lammps import Timestep
    p9 = Timestep("tests/selection/ni.p9.out")
    return p9.gb(28)

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

def test_gbs(GBCol):
    """Tests construction of grain boundary objects for each of dump files found
    in the testing directory.
    """
    from gblearn.gb import GrainBoundary
    GBCol.load(Z=28, method="cna_z", pattr="c_cna")
    for gbid, gb in GBCol.items():
        assert isinstance(gb, GrainBoundary)
    
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
def test_soap(GB9):
    """Tests generation of the SOAP vectors for this grain boundary.
    """
    P = GB9.soap

@pytest.mark.skip()
def test_K(GB9):
    """Tests generation of the kernel matrix for the GB.
    """
    K = GB9.K
