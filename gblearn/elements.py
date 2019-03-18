"""Crystal definitions and SOAP vector calculations for simple
elements.
"""
from gblearn import msg
import numpy as np
from ase import Atoms
_shells = {}
"""dict: keys are element names, values are lists of shells (in Ang.).
"""

elements = {
    "Ni": ("FaceCenteredCubic", 3.52, 28, [0]),
    "Cr": ("BodyCenteredCubic", 2.91, 24, [0, 1]),
    "Mg": ("HexagonalClosedPacked", {'a':3.21, 'c/a':1.633}, 12, [0, 1])
}
"""dict: keys are element names, values are a tuple of (`str` lattice,
`float` lattice parameter, `int` element number, `list` basis indices).
"""

def atoms(element):
    """Returns a :class:`quippy.Atoms` structure for the given
    element, using the tabulated lattice parameters.

    Args:
        element (str): name of the element.
    """
    lattice = "unknown"
    if element in elements:
        lattice, latpar, Z, basis = elements[element]
        if lattice == "HexagonalClosedPacked":
            import ase.lattice.hexagonal as structures
        else:
            import ase.lattice.cubic as structures
        if hasattr(structures, lattice):
            lat = getattr(structures, lattice)(element, latticeconstant=latpar)
            a = Atoms(positions=lat.positions, numbers=lat.numbers)
            a.set_cell(lat.cell)
            return a

    emsg = "Element {} with structure {} is not auto-configurable."
    msg.err(emsg.format(element, lattice))

def pissnnl(element, lmax=12, nmax=12, rcut=6.0, sigma=0.5, trans_width=0.5):
    """Computes the :math:`P` matrix for the given element.

    Args:
        element (str): name of the element.
        nmax (int): bandwidth limits for the SOAP descriptor radial basis
          functions.
        lmax (int): bandwidth limits for the SOAP descriptor spherical
          harmonics.
        rcut (float): local environment finite cutoff parameter.
        sigma (float): width parameter for the Gaussians on each atom.
        trans_width (float): distance over which the coefficients in the
            radial functions are smoothly transitioned to zero.
    """
    lattice, latpar, Z, basis = elements[element]
    from pycsoap.soaplite import SOAP
    soap_desc = SOAP(atomic_numbers=[Z], lmax=lmax, nmax=nmax, rcut=rcut)
    a = atoms(element)
    return soap_desc.create(a)
