"""Tests the SOAP vector generation and decomposition for the pure
element cases.
"""
import pytest
import numpy as np

@pytest.fixture(scope="module", autouse=True)
def elements(request):
    """Returns a list of elements that represent FCC, BCC and HCP lattices.

    Returns:
        (list): of `str` element names.
    """
    return ["Ni", "Cr", "Mg"]

@pytest.fixture(scope="module", autouse=True)
def models(request):
    """Returns a function that formats a file name to have the correct
    path for this modules model outputs.
    """
    from os import path
    return lambda f: path.join("tests/elements", f)

def test_atoms(elements, models):
    """Tests the initialization of pure elements Ni, Mg and Cr.
    """
    from gblearn.elements import atoms
    for e in elements:
        a = atoms(e)
        modelfile = models("{}.positions.npy".format(e))
        assert np.allclose(a.positions, np.load(modelfile))

    #Finally test the dummy case.
    assert atoms("Dummy") is None

def test_pissnnl(elements, models):
    """Tests the SOAP vector for each of the elements.
    """
    from gblearn.elements import pissnnl
    elements = ["Ni", "Cr", "Mg"]
    for e in elements:
        for i, dZ in enumerate(pissnnl(e)):
            modelfile = models("{}.pissnnl_{}.npy".format(e, i))
            assert np.allclose(dZ, np.load(modelfile))
