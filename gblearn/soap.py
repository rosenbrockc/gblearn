"""Functions for generating the SOAP representation of a grain
boundary.
"""
import numpy as np
def S(a, b):
    """Computes the SOAP similarity kernel between two SOAP vectors,
    :math:`d(a,b) = \sqrt{K(a,a)+K(b,b)-2*K(a,b)}`.
    """
    return np.sqrt(np.dot(a, a) + np.dot(b, b) - 2*np.dot(a, b))

class SOAPCalculator(object):
    """Represents a set of unique SOAP parameters for which SOAP
    vectors can be calculated.

    Args:
        rcut (float): local environment finite cutoff parameter.
        nmax (int): bandwidth limits for the SOAP descriptor radial basis
          functions.
        lmax (int): bandwidth limits for the SOAP descriptor spherical
          harmonics.
        sigma (float): width parameter for the Gaussians on each atom.
        trans_width (float): distance over which the coefficients in the
            radial functions are smoothly transitioned to zero.

    Attributes:
        rcut (float): local environment finite cutoff parameter.
        nmax (int): bandwidth limits for the SOAP descriptor radial basis
          functions.
        lmax (int): bandwidth limits for the SOAP descriptor spherical
          harmonics.
        sigma (float): width parameter for the Gaussians on each atom.
        trans_width (float): distance over which the coefficients in the
            radial functions are smoothly transitioned to zero.
    """
    def __init__(self, rcut=5., nmax=12, lmax=12, sigma=0.5, trans_width=0.5):
        self.rcut = rcut
        self.nmax = nmax
        self.lmax = lmax
        self.sigma = sigma
        self.trans_width = trans_width

    def calc(self, atoms, central, basis=None):
        """Calculates a SOAP vector for the specified species and atomic
        positions.

        Args:
            atoms (quippy.Atoms): list of atoms to calculate the vector for.
            central (int): integer element number to set as the central atom type
              for the SOAP calculation.
            basis (list): of `int` defining which of the atoms in the *conventional*
              unit cell should be retained as a unique part of the basis.
        """
        import quippy
        import quippy.descriptors as descriptor
        descstr = ("soap cutoff={0:.1f} n_max={1:d} l_max={2:d} "
                   "atom_sigma={3:.2f} n_species=1 species_Z={{{4:d}}} "
                   "Z={4:d} trans_width={5:.2f} normalise=F")
        Z = np.unique(atoms.get_atomic_numbers())[0]
        D = descriptor.Descriptor
        descZ = D(descstr.format(self.rcut, self.nmax, self.lmax, self.sigma,
                                 Z, self.trans_width))
        atoms.set_cutoff(descZ.cutoff())
        atoms.calc_connect()
        PZ = descZ.calc(atoms)
        if basis is not None:
            dZ = [PZ["descriptor"][b,:] for b in basis]
            return dZ
        else:
            return PZ
