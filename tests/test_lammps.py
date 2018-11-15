"""Tests the lammps.py module for extracting and writing dump files.
"""
import pytest
import numpy as np
def test_lattice():
    """Tests construction of a lattice from LAMMPS box.
    """
    from gblearn.lammps import make_lattice
    with pytest.raises(ValueError):
        make_lattice(np.array([[1,0]]))

def test_cnasel():
    """Tests selection of GB by CNA deviation from perfect crystal.
    """
    from gblearn.lammps import Timestep
    coord=0
    t = Timestep("tests/lammps/dump.in")
    gbids = t.gbids("cna", "c_cna", padding=10., coord=coord)
    gb = t.gb(28, "cna", "c_cna", padding=10., coord=coord)

    t3 = Timestep("tests/lammps/dump-3.in")
    with pytest.raises(ValueError):
        gbids = t3.gbids("cna", "c_cna", padding=10., coord=coord)        

def test_corners():
    """Tests some corner cases for 100% code coverage with time steps.
    """
    from gblearn.lammps import Timestep
    t = Timestep("tests/lammps/dump-corners.in")
    with open("tests/lammps/dump-corners.in") as f:
        t2 = Timestep("tests/lammps/dump-corners.in", openf=f)

    t3 = Timestep("tests/lammps/dump-corners.in", 4)
    t3 = Timestep("tests/lammps/dump-corners.in", None, stepfilter=[1,2])

    assert len(t) == 0
    assert len(t2) == 0
    assert len(t3) == 0

    t4 = Timestep("tests/lammps/dump-corners.in", 3)
    assert t4.periodic == (False, False, False)
    assert len(t4) == 20

def test_timestep(tmpdir):
    """Tests reading a lammps dump file.
    """
    from gblearn.lammps import Timestep
    t = Timestep("tests/lammps/dump.in")
    assert len(t) == len(t.types)
    assert len(t) == len(t.ids)
    assert np.allclose(t.box, np.array([[-62.4863, 117.625],
                                        [0., 30.1777],
                                        [0., 38.5597]]))
    assert len(t) == 20
    assert t.types[0] == 4
    assert t.ids[0] == 78
    assert np.allclose(t.xyz[0,:], [-61.6991, 0.629647, 1.08782])
    assert np.allclose(t.xyz[-1,:], [-24.705, 1.34849, 0.766101])
    assert t.c_cna[0] == 5
    assert abs(t.c_csd[0]-12.3904) < 1e-7

    #Next, test that the dumping works correctly.
    tsfile = str(tmpdir.join("tsdump.out"))
    t.dump(tsfile)
    r = Timestep(tsfile)
    assert r==t
    assert not (r == 10)

    #Test the reboxing on save:
    bsfile = str(tmpdir.join("boxdump.out"))
    t.dump(bsfile, rebox=True)
    bs = Timestep(bsfile)
    assert np.allclose(bs.box, np.array([[-61.6991, -24.705 ],
                                         [  0.6296,   2.067 ],
                                         [  0.4448,   2.3731]]))

def test_dump(tmpdir):
    """Tests reading a dump file with multiple timesteps.
    """
    from gblearn.lammps import Dump
    d = Dump("tests/lammps/dump-2.in")
    assert len(d) == 2
    assert 0 in d
    assert 1 in d

    t = d[0]
    assert np.allclose(t.box, np.array([[-62.4863, 117.625],
                                        [0., 30.1777],
                                        [0., 38.5597]]))
    assert len(t) == 20
    assert t.types[0] == 4
    assert t.ids[0] == 78
    assert np.allclose(t.xyz[0,:], [-61.6991, 0.629647, 1.08782])
    assert np.allclose(t.xyz[-1,:], [-24.705, 1.34849, 0.766101])
    assert t.c_cna[0] == 5
    assert abs(t.c_csd[0]-12.3904) < 1e-7

    t1 = d[1]
    assert np.allclose(t1.box, np.array([[-66.4863, 120.625],
                                        [0., 30.1777],
                                        [0., 38.5597]]))
    assert len(t1) == 20

    #Next, test that the dumping works correctly.
    dsfile = str(tmpdir.join("fulldump.out"))
    d.dump(dsfile)
    rd = Dump(dsfile)
    assert d == rd
    assert not (d == True)
