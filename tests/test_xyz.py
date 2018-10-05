import pytest
from os import getcwd, path
from gblearn.utility import reporoot
import numpy as np

@pytest.fixture
def XYZD(scope="module"):
    """Returns the 9th grain boundary stored in extended XYZ format
    """
    from gblearn.xyz import XYZParser as XYZP
    result = XYZP("tests/xyz/00110391110_Dud.xyz")
    return result

@pytest.fixture
def GBColXYZ(tmpdir):
    """Returns a GB Collection made from xyz files
    """
    from gblearn.gb import GrainBoundaryCollection as GBC
    gbpath = path.join(reporoot, "tests", "xyz")
    root = str(tmpdir.join("xyz"))
    result = GBC("imeall", gbpath, root, r"^(?P<gbid>[_Dud\d]+).xyz$", rcut=3.25, lmax=12, nmax=12, sigma=0.5)

    from gblearn.gb import GrainBoundary
    from gblearn.xyz import XYZParser
    result.load(parser=XYZParser, Z=26, method="cna_z", pattr="cna", cna_val=3)
    for gbid, gb in result.items():
        assert isinstance(gb, GrainBoundary)

    return result

def test_instattr(XYZD):
    """Tests XYZParser instance attributes for the 00110391110_Dud grain boundary
    """
    assert len(XYZD) == 11520
    assert XYZD.atoms.n == 11520
    assert XYZD.xyz.shape == (11520, 3)

def test_gbids(XYZD):
    """Tests gb ids in a z-oriented grain boundary are chosen correctly using the cna_z method.
    """
    import numpy as np
    xyzd = XYZD.gbids(method="cna_z", pattr="cna", cna_val=3)
    loadd = np.load("tests/xyz/00110391110_Dud.npy")
    assert np.allclose(np.sort(xyzd), np.sort(loadd))


def test_gb(XYZD):
    """Tests that the GrainBoundary instance made through xyz.gb is made as expected.
    """
    GBXYZ = XYZD.gb(Z=26, pattr="cna", method="cna_z", cna_val=3)

    assert GBXYZ.xyz.shape == (7080, 3)
    assert GBXYZ.extras == ['force', 'map_shift', 'pos', 'csp', 'cna', 'n_neighb', 'Z', 'species']
    assert len(GBXYZ.cna) == 7080

def test_xyzproperty(GBColXYZ):
    """Tests getting properties of GB collections made from xyz files
    """
    assert GBColXYZ.get_property("energy").shape == (6,)
    GBColXYZ.get_property("species")

def test_xyzLER(GBColXYZ):
    """Tests SOAP calculation and LER collections for GB collection made from xyz files
    """
    GBColXYZ.soap()
    for gbid, gb in GBColXYZ.items():
        with GBColXYZ.P[gbid] as stored:
            assert stored.shape == (len(gb.cna), 1015)

    seed = np.load(path.join(reporoot, "tests", "xyz", "alpha_fe.npy"))
    GBColXYZ.seed = seed
    eps = 0.002500
    LER = GBColXYZ.LER(eps)
    U = GBColXYZ.U(eps)
    assert LER.shape == (len(GBColXYZ), len(U["U"]))
