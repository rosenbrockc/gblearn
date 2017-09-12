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
    return ResultStore(range(1, 8), str(root), lmax=8, nmax=8, rcut=4.3)

@pytest.fixture
def memstore():
    """Returns a memory-only result store.
    """
    return ResultStore(range(1, 8), lmax=8, nmax=8, rcut=4.3)

def test_errors():
    """Tests raising of exceptions for faulty values.
    """
    with pytest.raises(ValueError):
        ResultStore(range(1, 8), lmax=8)

    xstore = ResultStore(range(1, 8), lmax=8, nmax=None, rcut=4.)
    with pytest.raises(ValueError):
        xstore.SOAP_str

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
    memstore.SOAP = {"lmax": 8, "nmax": 8, "rcut": 4.3}
    assert memstore.SOAP == (8, 8, 4.3)
    assert memstore.P == {}
    assert memstore.SOAP_str == "8_8_4.30"
        
def test_soap(store):
    """Tests the saving and restoration of random 2x2 SOAP matrices
    for each of the GBs.
    """
    #There shouldn't be anything before it is set.
    assert store.P is None

    #First, we generate the random matrices, then we set P and get P
    #and make sure they match.
    Ps = {gbid: np.random.random((2,2)) for gbid in store.gbids}
    store.P = Ps

    #Check that the directory has the relevant files and that they
    #don't have zero size.
    target = path.join(store.P_, store.SOAP_str)
    assert path.isdir(target)
    for gbid in store.gbids:
        gbpath = path.join(target, "{}.npy".format(gbid))
        assert path.isfile(gbpath)

    #Ask for a new store so that we can load the arrays from disk and
    #check their equality.
    nstore = ResultStore(range(1, 8), store.root, lmax=8, nmax=8, rcut=4.3)
    for gbid in nstore.gbids:
        assert np.allclose(nstore.P[gbid], Ps[gbid])

def test_LER(store):
    """Tests the parameterized, aggregated storage for U and LER.
    """
    eps = [1.1, 2.2, 3.3]
    LER = {e: np.random.random((7, 5)) for e in eps}
    store.LER = LER
    
    #Ask for a new store so that we can load the arrays from disk and
    #check their equality.
    nstore = ResultStore(range(1, 8), store.root, lmax=8, nmax=8, rcut=4.3)
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

    target = path.join(store.ASR_, "{}.npy".format(store.SOAP_str))
    assert path.isfile(target)

    nstore = ResultStore(range(1, 8), store.root, lmax=8, nmax=8, rcut=4.3)
    assert np.allclose(nstore.ASR, ASR)
    assert np.allclose(store.ASR, ASR)

def test_ASR_mem(memstore):
    """Tests the aggregated, single matrix storage.
    """
    assert memstore.ASR is None
    ASR = np.random.random((7, 6))
    memstore.ASR = ASR
    assert np.allclose(memstore.ASR, ASR)
