import pytest
from os import getcwd

@pytest.fixture
def XYZD(scope="module"):
    """Returns the 9th grain boundary stored in extended XYZ format
    """
    from gblearn.xyz import XYZParser as XYZP
    result = XYZP("tests/xyz/00110391110_Dud.xyz")
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
    """Tests that a GrainBoundary instance made through xyz.gb is made as expected.
    """

    GBXYZ = XYZD.gb(Z=26, pattr="cna", method="cna_z", cna_val=3)

    assert GBXYZ.xyz.shape == (7080, 3)
    assert GBXYZ.extras == ['force', 'map_shift', 'pos', 'csp', 'cna', 'n_neighb', 'Z', 'species']
    assert len(GBXYZ.cna) == 7080
