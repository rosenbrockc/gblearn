"""Tests the SOAP vector decomposition routines.
"""
import pytest
import numpy as np
@pytest.fixture(scope="module")
def FCC(request):
    """Returns the pure Ni FCC pissnnl.
    """
    from gblearn.elements import pissnnl
    return pissnnl("Ni")
@pytest.fixture(scope="module")
def BCC(request):
    """Returns the pure Cr BCC pissnnl.
    """
    from gblearn.elements import pissnnl
    return pissnnl("Cr")
@pytest.fixture(scope="module")
def HCP(request):
    """Returns the pure Mg HCP pissnnl.
    """
    from gblearn.elements import pissnnl
    return pissnnl("Mg")
@pytest.fixture(scope="module")
def rx(request):
    return np.load("tests/decomp/r.npy")
@pytest.fixture(scope="module")
def GBs(request):
    """Returns the full, averaged GB system matrix for the 387 grain
    boundaries with non-trivial vectors.
    """
    pissnnl=np.load("data/asr.npy")
    pself = np.array([np.dot(p, p) for p in pissnnl])
    return np.array([pissnnl[i,:]/np.sqrt(pself[i])
                     for i in range(len(pissnnl))
                     if pself[i] > 0])

def test_cnum(rx):
    """Tests the numeric computation of the radial distribution
    functions.
    """
    #Note: when I plot these, they look different than the
    #analytically derived ones, though they are much faster (because
    #they use linear algebra with matrix calculations). There is also
    #a problem with certain values of nmax and lmax that make it
    #unstable (which is why I switched to the analytic calculation,
    #which works fine). It is tested here so that if we decide to fix
    #it later, the tests are in place.
    from gblearn.decomposition import SOAPDecomposer
    d = SOAPDecomposer()
    for n in range(1,4):
        model = np.load("tests/decomp/cnum.{}.npy".format(n))
        code = np.array([d.cnl(n,0,ri,cnum=(d.nmax, d.lmax)) for ri in rx])
        assert np.allclose(model, code)

def test_collection(GBs, tmpdir):
    """Tests the collection of SOAP Vectors.
    """
    from gblearn.decomposition import SOAPVectorCollection
    SVC = SOAPVectorCollection(GBs, rcut=5.0, lmax=18, nmax=18)

    #Test all the overloaded operators.
    GB10 = SVC[10]
    assert len(SVC) == 387
    assert GB10 == SVC.vectors[10]
    SVC[10] = GB10
    assert GB10 in SVC
    with pytest.raises(TypeError):
        SVC[10] = "some string"
    
    for i, GB in enumerate(SVC):
        if i > 3:
            break
        
    #Get a subset of the vectors into a new collection.
    sub = SVC[7:11]
    assert isinstance(sub, SOAPVectorCollection)
    assert len(sub) == 4
    assert GB10 in sub

    #Now construct the unique RDFs for this collection. This one uses
    #the multi-processing functionality.
    from gblearn.base import set_nprocs
    set_nprocs(4)
    RDFs = sub.RDFs()
    ADFs = sub.ADFs()
    set_nprocs(None)
    with pytest.raises(TypeError):
        cDFs = RDFs + ADFs
    with pytest.raises(TypeError):
        RDFs[0] + ADFs[0]
    with pytest.raises(TypeError):
        RDFs.add(ADFs[0])
        
    #Make sure that the slices also get the DFs right.
    DF6 = SVC[6:12]
    DF6.RDFs()
    DF6.ADFs(catom=True)
    DF4 = DF6[0:4]
    
    #Test the saving and re-loading of the collection.
    svcfile = str(tmpdir.join("SVC.pkl"))
    sub.save(svcfile)
    resub = SOAPVectorCollection.from_file(svcfile)
    assert resub == sub
    assert resub.equal(sub)

    esvcfile = str(tmpdir.join("emptysvc.pkl"))
    emptySVC = SOAPVectorCollection()
    emptySVC.save(esvcfile)
    memptySVC = SOAPVectorCollection.from_file(esvcfile)
    assert emptySVC == memptySVC
    assert emptySVC.equal(memptySVC)
    
    #Also test construction serially.
    bRDFs = SVC[25:27].RDFs()
    bADFs = SVC[25:27].ADFs()

    #Next test the saving and restoration of a DFCollection.
    from gblearn.decomposition import DFCollection
    RDFfile = str(tmpdir.join("bRDFs.pkl"))
    bRDFs.save(RDFfile)
    mRDFs = DFCollection.from_file(RDFfile)
    assert mRDFs == bRDFs

    ADFfile = str(tmpdir.join("bADFs.pkl"))
    bADFs.save(ADFfile)
    mADFs = DFCollection.from_file(ADFfile)
    assert mADFs == bADFs
    
    #Next, try and save/restore a single RDF.
    from gblearn.decomposition import DF
    single = RDFs[0]
    sfile = str(tmpdir.join("singleRDF.pkl"))
    single.save(sfile)
    msingle = DF.from_file(sfile)
    assert msingle == single
    assert msingle in RDFs
    RDFs[0] = msingle

    single = ADFs[0]
    sfile = str(tmpdir.join("singleADF.pkl"))
    single.save(sfile)
    msingle = DF.from_file(sfile)
    assert msingle == single
    assert msingle in ADFs
    ADFs[0] = msingle
    
    with pytest.raises(TypeError):
        RDFs[0] = True
    with pytest.raises(TypeError):
        ADFs[0] = False
    
    #Also test bogus equalities
    assert not (mRDFs == "dummy")
    assert not (single == 1e10)

    #Now, try working with adding and removing.
    cRDFs = bRDFs + RDFs
    assert len(cRDFs) == len(bRDFs) + len(RDFs)
    blength = len(bRDFs)
    #We also do some label assignments to make sure they are updating
    #correctly.
    RDFs.label = "GB4"
    bRDFs += RDFs
    bRDFs.label = "GBb + GBr"
    assert len(bRDFs) == blength + len(RDFs)
    bRDFs += RDFs
    with pytest.raises(TypeError):
        bRDFs += 10
        
    #Test the string representation of the collection and individual
    #RDF.
    str(bRDFs)
    repr(bRDFs)

    #Make sure the correct errors are raised if the user is confused.
    with pytest.raises(ValueError):
        DFCollection.dfs_from_soap(None, "bogus")
    
    #Testing creating a RDF from scratch.
    custom = DFCollection()
    custom.add(RDFs[1])
    custom.add(RDFs[3])
    custom.add(RDFs[1])
    assert len(custom) == 2
    assert RDFs[1] in custom
    custom.remove(RDFs[1])
    assert RDFs[1] not in custom
    assert RDFs[2] not in custom
    custom.remove(RDFs[2])

    with pytest.raises(TypeError):
        custom.add(False)
    uncust = custom.unique()
    uncust.add(RDFs[1])
    uncust.add(RDFs[1])
    uncust.add(RDFs[3])
    uncust.remove(True)
    uncust.histogram()

    refined = custom.refine(RDFs)
    histfile = str(tmpdir.join("RDFcolHist.pdf"))
    RDFs.histogram(savefile=histfile)
    
    from gblearn.elements import shells
    refined.plot(withavg=True, shells=shells("Ni"))
    with pytest.raises(TypeError):
        custom.refine("garbage")

    plotfile = str(tmpdir.join("RDFcolPlot.pdf"))
    ax = refined.plot(title="Unit test plot", xlabel="Radial",
                      ylabel="Distribution", savefile=plotfile)
    custom.plot(ax)
    RDFs.plot()

    #Test what happens when we call methods on an empty class.
    from gblearn.decomposition import RDFCollection
    empty = RDFCollection()
    unempty = empty.unique()
    assert len(unempty) == 0
    assert type(unempty) == RDFCollection
    
def test_fcut(rx):
    """Tests the cutoff function values.
    """
    model = np.load("tests/decomp/fcut.npy")
    from gblearn.decomposition import fcut
    assert np.allclose(model, fcut(rx, 6., 0.5))

def test_decompose(FCC, BCC, HCP):
    """Tests decomposition of the pure elements P vectors.
    """
    #For the elements, we use all the standard, default parameters in
    #the SOAP vector.
    from gblearn.decomposition import SOAPDecomposer
    from cPickle import load

    with open("tests/decomp/dFCC.pkl", 'rb') as f:
        mFCC = load(f)
    with open("tests/decomp/dBCC.pkl", 'rb') as f:
        mBCC = load(f)
    with open("tests/decomp/dHCP.pkl", 'rb') as f:
        mHCP = load(f)

    d = SOAPDecomposer()    
    #assert mFCC == d.decompose(FCC[0])
    assert mBCC == d.decompose(BCC[1])
    assert mHCP == d.decompose(HCP[1])

def test_partition():
    from gblearn.decomposition import SOAPDecomposer    
    d = SOAPDecomposer()
    l0 = np.load("tests/decomp/partition_0.npy")
    lrest = np.load("tests/decomp/partition_rest.npy")

    assert np.allclose(d.partition([0]), l0)
    assert np.allclose(d.partition([0], True), lrest)

def test_RDF(FCC, BCC, HCP, rx):
    """Tests the RDF construction for the pure elements.
    """
    
    NiRDF = np.load("tests/decomp/NiRDF.npy")
    CrRDF = np.load("tests/decomp/CrRDF.npy")
    MgRDF = np.load("tests/decomp/MgRDF.npy")
    
    from gblearn.decomposition import SOAPDecomposer, SOAPVector
    d = SOAPDecomposer()
    
    vFCC = SOAPVector(FCC[0], d)
    #We call this assertion twice because the second time it is
    #supposed to use a cached version, since constructing the RDF is
    #quite expensive.
    assert np.allclose(vFCC.RDF(rx).df, NiRDF)
    assert np.allclose(vFCC.RDF(rx).df, NiRDF)
    vBCC = SOAPVector(BCC[1], d)
    assert np.allclose(vBCC.RDF(rx).df, CrRDF)
    vHCP = SOAPVector(HCP[1], d)
    assert np.allclose(vHCP.RDF(rx).df, MgRDF)

    NiRDF0 = np.load("tests/decomp/NiRDF_0.npy")
    CrRDF0 = np.load("tests/decomp/CrRDF_0.npy")
    MgRDF0 = np.load("tests/decomp/MgRDF_0.npy")

    vFCC.rx = None
    assert np.allclose(vFCC.RDF(rx, True).df, NiRDF0)
    assert np.allclose(vFCC.RDF(rx, True).df, NiRDF0)
    assert np.allclose(vBCC.RDF(rx, True).df, CrRDF0)
    assert np.allclose(vHCP.RDF(rx, True).df, MgRDF0)

def test_vector_saveload(FCC, rx, tmpdir):
    """Tests saving and loading a SOAP vector.
    """
    from gblearn.decomposition import SOAPDecomposer, SOAPVector
    NiSV = tmpdir.join("Ni.pij.pkl")
    d = SOAPDecomposer()
    vFCC = SOAPVector(FCC[0], d)
    eFCC = SOAPVector.from_element("Ni")

    assert vFCC == eFCC
    assert vFCC.equal(eFCC)

    vFCC.RDF(rx)
    vFCC.ADF(np.linspace(0, np.pi, 100))
    vFCC.ADF(np.linspace(0, np.pi, 100), catom=True)
    vFCC.save(str(NiSV))
    rFCC = SOAPVector.from_file(str(NiSV))
    assert rFCC == vFCC
    assert rFCC.equal(vFCC)

    #Since the default comparison only looks at P vectors, we do some
    #more rigorous testing as well.
    assert rFCC.equal(vFCC)

    #Finally, set a specific domain in the radial space and then save
    #and restore and check again.
    NiSVr = tmpdir.join("Nir.pij.pkl")
    rx = np.linspace(0, 6, 100)
    vFCC.RDF(rx)
    vFCC.RDF(rx, True)
    
    vFCC.save(str(NiSVr))
    nFCC = SOAPVector.from_file(str(NiSVr))
    assert nFCC.equal(vFCC)

def test_df(FCC, rx, tmpdir):
    """Tests plotting and distance metric of the radial/angular
    distribution function.
    """
    from gblearn.decomposition import SOAPDecomposer, SOAPVector
    outfile = tmpdir.join("FCC_RDF.pdf")
    d = SOAPDecomposer()
    vFCC = SOAPVector(FCC[0], d)
    ax = vFCC.RDF(rx).plot(title="Unit test plot", xlabel="Radial",
                      ylabel="Distribution", savefile=str(outfile))
    vFCC.RDF(rx).plot(ax)
    vFCC.RDF(rx).plot()

    assert vFCC.RDF(rx).same(vFCC.RDF(rx))
    with pytest.raises(TypeError):
        vFCC.RDF(rx) - 200
    assert vFCC.RDF(rx).same(vFCC.RDF(rx), 1e-10)
    
