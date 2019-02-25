"""Tests the result store object on the small GB collection in the
unit tests.

.. note:: These tests don't actually compute any real SOAP matrices or
  other representations; they use tiny, "dummy" random matrices to
  ensure that the storage is operating correctly.
"""
import pytest
import numpy as np
from os import path
from gblearn.io import ResultStore

@pytest.fixture
def store(tmpdir):
    root = tmpdir.join("rndstore")
    res = ResultStore(range(1,8), str(root))
    res.configure("soap", lmax=8, nmax=8, rcut=4.3)
    res.configure("scatter", density=0.5, Layers=2, SPH_L=6, n_trans=8, n_angle1=8, n_angle2=8)
    return res

@pytest.fixture
def memstore():
    """Returns a memory-only result store.
    """
    res = ResultStore(range(1,8))
    res.configure("soap", lmax=8, nmax=8, rcut=4.3)
    res.configure("scatter", density=0.5, Layers=2, SPH_L=6, n_trans=8, n_angle1=8, n_angle2=8)
    return res

def test_errors():
    """Tests raising of exceptions for faulty values.
    """
    xstore = ResultStore(range(1, 8))
    with pytest.raises(KeyError):
        xstore.configure("soap", lmax=8, nmax=None, rcut=4.)

def test_soapmem(memstore):
    """Tests the memory-only store for SOAP matrices.
    """
    assert memstore.P == {}

    Ps = {gbid: np.random.random((2,2)) for gbid in memstore.gbids}
    memstore.P = Ps

    #If we have the wrong number of matrices, then nothing should happen.
    oPs = {gbid: np.random.random((2,2)) for gbid in memstore.gbids}
    oPs[9] = np.random.random((2,2))
    memstore.P = oPs

    for gbid in memstore.gbids:
        assert np.allclose(memstore.P[gbid], Ps[gbid])

    #Now, set the SOAP args and make sure that the rep was clobbered.
    memstore.configure("soap", lmax=8, nmax=8, rcut=4.3)
    assert memstore.P == {}
    assert getattr(memstore,"soapstr") == "8_8_4.30"

def test_soap(store):
    """Tests the saving and restoration of random 2x2 SOAP matrices
    for each of the GBs.
    """
    #There shouldn't be anything before it is set.
    with store.P[store.gbids[0]] as stored:
        assert stored is None

    #First, we generate the random matrices, then we set P and get P
    #and make sure they match.
    Ps = {gbid: np.random.random((2,2)) for gbid in store.gbids}
    store.P = Ps

    #Check that the directory has the relevant files and that they
    #don't have zero size.
    target = path.join(store.root, "soap", "P", "8_8_4.30")
    assert path.isdir(target)
    for gbid in store.gbids:
        gbpath = path.join(target, "{}.npy".format(gbid))
        assert path.isfile(gbpath)

    #Ask for a new store so that we can load the arrays from disk and
    #check their equality.
    nstore = ResultStore(range(1,8), store.root)
    nstore.configure("soap", lmax=8, nmax=8, rcut=4.3)
    assert len(nstore.gbids) == len(nstore.P)
    for gbid in nstore.gbids:
        with nstore.P[gbid] as stored:
            assert np.allclose(stored, Ps[gbid])

def test_scattermem(memstore):
    """Tests the memory-only store for Scatter matrices.
    """
    assert memstore.Scatter == {}

    Scatters = {gbid: np.random.random((2,2)) for gbid in memstore.gbids}
    memstore.Scatter = Scatters

    #If we have the wrong number of matrices, then nothing should happen.
    oScatters = {gbid: np.random.random((2,2)) for gbid in memstore.gbids}
    oScatters[9] = np.random.random((2,2))
    memstore.Scatter = oScatters

    for gbid in memstore.gbids:
        assert np.allclose(memstore.Scatter[gbid], Scatters[gbid])

    #Now, set the Scatter args and make sure that the rep was clobbered.
    memstore.configure("scatter",density=0.5, Layers=2, SPH_L=10, n_trans=8, n_angle1=8, n_angle2=8)
    assert memstore.Scatter == {}
    assert getattr(memstore,"scatterstr") == "0.50_2_10_8_8_8"

def test_scatter(store):
    """Tests the saving and restoration of random 2x2 Scatter matrices
    for each of the GBs.
    """
    #There shouldn't be anything before it is set.
    with store.Scatter[store.gbids[0]] as stored:
        assert stored is None

    #First, we generate the random matrices, then we set Scatter and get Scatter
    #and make sure they match.
    Scatters = {gbid: np.random.random((2,2)) for gbid in store.gbids}
    store.Scatter = Scatters

    #Check that the directory has the relevant files and that they
    #don't have zero size.
    target = path.join(store.root, "scatter", "Scatter", "0.50_2_6_8_8_8")
    assert path.isdir(target)
    for gbid in store.gbids:
        gbpath = path.join(target, "{}.npy".format(gbid))
        assert path.isfile(gbpath)

    #Ask for a new store so that we can load the arrays from disk and
    #check their equality.
    nstore = ResultStore(range(1, 8), store.root)
    nstore.configure("scatter", density=0.5, Layers=2, SPH_L=6, n_trans=8, n_angle1=8, n_angle2=8)
    assert len(nstore.gbids) == len(nstore.Scatter)
    for gbid in nstore.gbids:
        with nstore.Scatter[gbid] as stored:
            assert np.allclose(stored, Scatters[gbid])

def test_LER(store):
    """Tests the parameterized, aggregated storage for U and LER.
    """
    eps = [1.1, 2.2, 3.3]
    LER = {e: np.random.random((7, 5)) for e in eps}
    store.LER = LER

    #Ask for a new store so that we can load the arrays from disk and
    #check their equality.
    nstore = ResultStore(range(1, 8), store.root)
    nstore.configure("soap", lmax=8, nmax=8, rcut=4.3)
    for e in eps:
        assert np.allclose(nstore.LER[e], LER[e])

def test_U_mem(memstore):
    """Tests the parameterized, aggregated storage for U and LER.
    """
    assert memstore.U == {}

    eps = [1.1, 2.2, 3.3]
    U = {e: np.random.random((7, 5)) for e in eps}
    memstore.U = U

    for e in eps:
        assert np.allclose(memstore.U[e], U[e])

def test_ASR(store):
    """Tests the aggregated, single matrix storage.
    """
    ASR = np.random.random((7, 6))
    store.ASR = ASR

    target = path.join(store.root, "soap", "ASR", "8_8_4.30.npy")
    assert path.isfile(target)

    nstore = ResultStore(range(1, 8), store.root)
    nstore.configure("soap", lmax=8, nmax=8, rcut=4.3)
    assert np.allclose(nstore.ASR, ASR)
    assert np.allclose(store.ASR, ASR)

def test_ASR_mem(memstore):
    """Tests the aggregated, single matrix storage.
    """
    assert memstore.ASR is None
    ASR = np.random.random((7, 6))
    memstore.ASR = ASR
    assert np.allclose(memstore.ASR, ASR)
