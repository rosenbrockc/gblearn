"""Tests the grain boundary instance methods as well as the GB
collection methods.
"""
import pytest
@pytest.fixture
def GBCol(tmpdir):
    from gblearn.gb import GrainBoundaryCollection
    root = str(tmpdir.join("olmsted"))

@pytest.fixture(scope="module")
def GB9(request):
    """Returns the grain boundary atoms from the 9th sample in the
    Olmsted set.
    """
    from gblearn.lammps import Timestep
    p9 = Timestep("tests/selection/ni.p9.out")
    return p9.gb(28)

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

def test_soap(GB9):
    """Tests generation of the SOAP vectors for this grain boundary.
    """
    P = GB9.soap

def test_K(GB9):
    """Tests generation of the kernel matrix for the GB.
    """
    K = GB9.K
