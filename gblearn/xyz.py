"""Module to create :class:`GrainBoundary` objects from extended XYZ files.
"""
from ase.io import read

class XYZParser(object):
    """Represents a grain boundary stored in extended XYZ format.

    Args:
        filepath (str): path to the grain boundary XYZ file.

    Attributes:
        atoms (ase.Atoms): parsed atoms object from which the
          :class:`GrainBoundary` will be created.
        xyz (numpy.ndarray): xyz positions of the atoms in the XYZ file.
        extras (list): of `str` parameter names with additional, global GB
          properties.
        types (numpy.ndarray): integer lattice types of the atoms in the list.

    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.atoms = read(filepath)
        self.xyz = self.atoms.get_positions()
        self.arrays = list(self.atoms.arrays.keys())
        self.info = list(self.atoms.info.keys())
        for k in self.arrays:
            setattr(self, k, self.atoms.arrays[k])
        for k in self.info:
            setattr(self, k, self.atoms.info[k])
        self.types = None
        self.box = self.atoms.cell

    def __eq__(self, other):# pragma: no cover
        return self.atoms == other.atoms
    def __len__(self):
        return len(self.atoms)

    def gb(self, Z=None, method="cna", pattr="c_cna", extras=True, padding=10.0,
           **selectargs):
        """Returns the grain boundary for this XYZ file.

        Args:
            Z (int or list): element code(s) for the atomic species.
            method (str): one of ['cna'].
            pattr (str): name of an attribute in :attr:`extras` to pass as the
              selection parameter of the routine.
            extras (bool): when True, include extra attributes in the new GB
              structure.
            selectargs (dict): additional arguments passed to the atom selection
              function. For `cna*` see :func:`gblearn.selection.cna_max`.

        Returns:
            gblearn.gb.GrainBoundary: instance with only those atoms that appear
              to be at the boundary.
        """
        if Z is None:# pragma: no cover
            raise ValueError("`Z` is a required parameter for constructing a "
                             ":class:`GrainBoundary` instance.")

        from gblearn.gb import GrainBoundary
        selargs = {
            "method": method,
            "pattr": pattr
        }
        selargs.update(selectargs)

        ids = self.gbids(padding=padding, **selargs)

        if extras:
            x = {k: getattr(self, k)[ids] for k in self.arrays}
            x.update({k: getattr(self, k) for k in self.info})
        else:# pragma: no cover
            x = None
        if self.types is not None:# pragma: no cover
            types = self.types[ids]
        else:
            types = None

        result = GrainBoundary(self.xyz[ids,:], types,
                               self.box, Z, extras=x, makelat=False,
                               selectargs=selargs, padding=padding)
        return result

    def gbids(self, method="cna", pattr=None, padding=10.0, **kwargs):
        """Returns the *indices* of the atoms that lie at the grain
        boundary.

        Args:
            method (str): one of [cna', 'cna_z'].
            pattr (str): name of an attribute in :attr:`extras` to pass as the
              selection parameter of the routine.
            cna_val (int): type id of the *perfect crystal*.
            padding (float): amount of perfect bulk to include as padding around
              the grain boundary before the representation is made.
            kwargs (dict): additional arguments passed to the atom selection
              function. For `cna*` see :func:`gblearn.selection.cna_max`.

        Returns:
            numpy.ndarray: of integer indices of atoms in this timestep that are
              considered to lie on the boundary.

        Examples:

            >>> from gblearn.xyz import XYZParser
            >>> gb0 = XYZParser("gb10.xyz")
            >>> ids = gb0.gbids()
            >>> xyz = gb0.xyz[ids,:]
        """
        import gblearn.selection as sel
        from functools import partial
        methmap = {
            "cna": partial(sel.cna_max, coord=kwargs['coord'])
            }
        if method in methmap:
            extra = getattr(self, pattr) if pattr is not None else None
            return methmap[method](self.xyz, extra, padding=padding, types=self.types, **kwargs)
