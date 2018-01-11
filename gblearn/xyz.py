"""Module to create :class:`GrainBoundary` objects from extended XYZ files.
"""
import quippy

class XYZParser(object):
    """Represents a grain boundary stored in extended XYZ format.

    Args:
        filepath (str): path to the grain boundary XYZ file.

    Attributes:
        atoms (quippy.Atoms): parsed atoms object from which the
          :class:`GrainBoundary` will be created.
        xyz (numpy.ndarray): xyz positions of the atoms in the XYZ file.
        extras (list): of `str` parameter names with additional, global GB
          properties.
        types (numpy.ndarray): integer lattice types of the atoms in the list.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.atoms = quippy.Atoms(filepath)
        self.xyz = self.atoms.positions
        self.extras = list(self.atoms.properties.keys())
        for k in self.extras:
            setattr(self, k, self.atoms.properties[k])
        self.types = None

    def __eq__(self, other):
        return self.atoms == other.atoms
    def __len__(self):
        return self.atoms.n
    
    def gb(self, Z=None, method="median", pattr="c_csd", extras=True, soapargs={},
           **kwargs):
        """Returns the grain boundary for this XYZ file.

        Args:
            Z (int or list): element code(s) for the atomic species.
            method (str): one of ['median'].
            pattr (str): name of an attribute in :attr:`extras` to pass as the
              selection parameter of the routine.
            extras (bool): when True, include extra attributes in the new GB
              structure.
            soapargs (dict): initialization parameters for the
              :class:`gblearn.soap.SOAPCalculator` instance for the GB.
            kwargs (dict): additional arguments passed to the atom selection
              function. For `median`, see :func:`gblearn.selection.median` for the
              arguments. For `cna*` see :func:`gblearn.selection.cna_max`.
        
        Returns:
            gblearn.gb.GrainBoundary: instance with only those atoms that appear
              to be at the boundary.
        """
        if Z is None:
            raise ValueError("`Z` is a required parameter for constructing a "
                             ":class:`GrainBoundary` instance.")
        
        from gblearn.gb import GrainBoundary
        selectargs = {
            "method": method,
            "pattr": pattr
        }
        selectargs.update(kwargs)
        
        ids = self.gbids(**selectargs)
        
        if extras:
            x = {k: getattr(self, k)[ids] for k in self.extras}
        else:
            x = None
        result = GrainBoundary(self.xyz[ids,:], self.types[ids],
                               self.box, Z, extras=x,
                               selectargs=selectargs, **soapargs)
        return result

    def gbids(self, method="median", pattr=None, cna_val=3, **kwargs):
        """Returns the *indices* of the atoms that lie at the grain
        boundary.

        Args:
            method (str): one of ['median', 'cna', 'cna_z'].
            pattr (str): name of an attribute in :attr:`extras` to pass as the
              selection parameter of the routine.
            cna_val (int): type id of the *perfect crystal*.
            kwargs (dict): additional arguments passed to the atom selection
              function. For `median`, see :func:`gblearn.selection.median` for the
              arguments.

        Returns:
            numpy.ndarray: of integer indices of atoms in this timestep that are
              considered to lie on the boundary.

        Examples:
            Retrieve the positions of the atoms that lie at the boundary using the
            median centro-symmetry parameter values.

            >>> from gblearn.xyz import XYZParser
            >>> gb0 = XYZParser("gb10.xyz")
            >>> ids = gb0.gbids()
            >>> xyz = gb0.xyz[ids,:]
        """
        import gblearn.selection as sel
        from functools import partial
        methmap = {
            "median": sel.median,
            "cna": partial(sel.cna_max, coord=0, cna_val=cna_val),
	    "cna_z": partial(sel.cna_max, coord=2, cna_val=cna_val)
            }
        if method in methmap:
            extra = getattr(self, pattr) if pattr is not None else None
            return methmap[method](self.xyz, extra, types=self.types, **kwargs)
