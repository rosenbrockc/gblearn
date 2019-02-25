"""Tests the selection tools for finding atoms that are close to a
grain boundary.
"""
import pytest
import numpy as np
def test_median():
    """Tests selection of atoms at the grain boundary based on
    relative difference between the median and maximum values of the
    centro-symmetry parameter.
    """
    from gblearn.lammps import Timestep
    p9 = Timestep("tests/selection/ni.p9.out")
    s9 = Timestep("tests/selection/ni.s9.out")

    sind = p9.gbids(pattr="c_csd")
    sids = p9.ids[sind]
    #Unfortunately, the order is not necessarily preserved between the
    #input and model output. Sort by the atom ids, which *should* stay
    #the same.
    assert len(sind) == len(s9)
    assert np.allclose(np.sort(sids), np.sort(s9.ids))

    #Try the same thing with passing keywords args to gb()
    sind = p9.gbids(limit_extent=(0, 10.), pattr="c_csd")
    sids = p9.ids[sind]
    #Unfortunately, the order is not necessarily preserved between the
    #input and model output. Sort by the atom ids, which *should* stay
    #the same.
    assert len(sind) == len(s9)
    assert np.allclose(np.sort(sids), np.sort(s9.ids))
