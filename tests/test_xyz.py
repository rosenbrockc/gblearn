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
    result = GBC("imeall", gbpath, root, r"^(?P<gbid>[_Dud\d]+).xyz$", padding=6.5)
    result.store.configure("soap", rcut=3.25, lmax=12, nmax=12, sigma=0.5)

    from gblearn.gb import GrainBoundary
    from gblearn.xyz import XYZParser
    result.load(parser=XYZParser, Z=26, method="cna", pattr="cna", cna_val=3)
    for gbid, gb in result.items():
        assert isinstance(gb, GrainBoundary)

    return result

def test_instattr(XYZD):
    """Tests XYZParser instance attributes for the 00110391110_Dud grain boundary
    """
    assert len(XYZD) == 11520
    assert len(XYZD.atoms) == 11520
    assert XYZD.xyz.shape == (11520, 3)

def test_gbxyzids(XYZD):
    """Tests gb ids in a z-oriented grain boundary are chosen correctly using the cna_z method.
    """
    import numpy as np
    xyzd = XYZD.gbids(method="cna", pattr="cna", cna_val=3, coord=2, padding=5.)
    loadd = np.load("tests/xyz/00110391110_Dud.npy")
    assert np.allclose(np.sort(xyzd), np.sort(loadd))


def test_gbxyz(XYZD):
    """Tests that the GrainBoundary instance made through xyz.gb is made as expected.
    """
    GBXYZ = XYZD.gb(Z=26, pattr="cna", method="cna", cna_val=3, coord=2, padding=5.)

    assert GBXYZ.xyz.shape == (7080, 3)
    assert GBXYZ.extras == ['cutoff', 'adsorbate_info', 'force', 'nneightol',
    'energy', 'positions', 'map_shift', 'numbers', 'cna', 'n_neighb', 'virial', 'csp']
    assert len(GBXYZ.cna) == 7080

def test_xyzLER(GBColXYZ):
    """Tests SOAP calculation and LER collections for GB collection made from xyz files
    """
    GBColXYZ.soap(rcut=3.25, lmax=12, nmax=12, sigma=0.5)
    for gbid, gb in GBColXYZ.items():
        with GBColXYZ.P[gbid] as stored:
            assert stored.shape == (len(gb.cna), 1015)

    seed = np.load(path.join(reporoot, "tests", "xyz", "alpha_fe.npy"))
    GBColXYZ.seed = seed
    eps = 0.002500
    LER = GBColXYZ.LER(eps)
    U = GBColXYZ.U(eps)
    assert LER.shape == (len(GBColXYZ), len(U["U"]))
