"""Functions for selecting the atoms that allegedly contribute to the
grain boundary properties.
"""
import numpy as np
from gblearn.base import deprecated

def extent(struct, axis=2):
    """Returns the maximum extent of the structure along the given
    axis (default `x`, on axis 2).

    Args:
        struct (numpy.ndarray): each row is an atom entry. Usually columns are
          ['id', 'type', 'x', 'y', 'z', ...], though as long as `axis` is
          consistent, it doesn't matter.
        axis (int): axis in `struct` along which to find the `min` and
          `max` extent.

    Returns:
        tuple: of (`float` min, `float` max) structure extent along
          `axis`.
    """
    return np.min(struct[:,axis]), np.max(struct[:,axis])

def cna_max(xyz, cna, types=None, cna_val=1, padding=10.0, coord=None, **kwargs):
    """Returns the atoms in the crystal whose CNA value deviates from
    the given type; a buffer of rcut is added for padding to both
    sides of the grain boundary.

    .. note:: the boundary is isolated by looking for the `min(x)` and
      `max(x)` where `cna != cna_val` and `type not in [4, 5]`, where
      types 4 and 5 identify the far sides of the entire crystal,
      which are far from the GB.

    Args:
        xyz (numpy.ndarray): each row is an atom. Columns are cartesian atomic
          positions
        cna (numpy.ndarray): of `c_cna` values being considered for the
          selection.
        types (numpy.ndarray): of crystal types for each atom in the crystal.
        cna_val (int): type id of the *perfect crystal*.
        padding (float): how much padding to add to each side of the
          isolated grain boundary.
        coord (int): integer coordinate `(x:0, y:1, z:2)` to select with respect
          to.
        kwargs (dict): dummy parameter so that selection routines can all accept
          the same dictionary.

    Returns:
        numpy.ndarray: of integer indices in `xyz` that match the filtering
          conditions.
    """
    if types is not None:
        type_mask = np.logical_and(types != 4, types != 5)
        cna_mask = np.logical_and(cna != cna_val, type_mask)
    else:
        cna_mask = cna != cna_val
    xvals = xyz[cna_mask,coord]

    if len(xvals) == 0:
        raise ValueError("No atoms selected at the grain boundary.")

    #Now that we have all atoms that deviate from perfect crystal and
    #are away from the edges, we find the minimum and maximum values
    #and add the desired padding to each.
    minx, maxx = np.min(xvals) - padding, np.max(xvals) + padding
    result = np.where(np.logical_and(xyz[:,coord] >= minx, xyz[:,coord] <= maxx))[0]
    return result

@deprecated
def median(xyz, param, limit_extent=None, tolerance=0.5, width=8., types=None, **kwargs): # pragma: no cover
    """Returns those atoms that deviate from the median appreciably,
    along a given axis.

    Args:
        xyz (numpy.ndarray): each row is an atom. Columns are cartesian atomic
          positions
        param (numpy.ndarray): of parameter values being considered for the
          selection.
        limit_extent (tuple): of (`int` axis, `float` extent) when not
          `None`, only atoms within `extent` units of the min, max extent
          along the `axis` are included. I.e., it provides further spatial
          filtering in addition to the median filtering of `param`. Default is
          x-axis `axis=0` with `extent=10.`.
        tolerance (float): atoms are selected when their parameter value
          along `axis` exceeds some threshold. The threshold is calculated
          relative to the difference between *median* and *maximum*
          `param` values for all atoms in the structure. `tolerance`
          scales the selection width (on a :math:`log_10` scale) between
          median and maximum values for selection.
        width (float): width of the rectangular slab that will be selected
          around the group of atoms with highest centro-symmetry.
        types (numpy.ndarray): of crystal types for each atom in the
          crystal. It is included here so that we can use a common
          interface for all selection routines.
        kwargs (dict): dummy parameter so that selection routines can all accept
          the same dictionary.

    Returns:
        numpy.ndarray: of integer indices in `xyz` that match the filtering
          conditions.
    """
    if limit_extent is not None:
        xaxis, ext = limit_extent
    else:
        xaxis, ext = (0, 10.)
    minx, maxx = extent(xyz, xaxis)
    space = xyz[:,xaxis]

    #We determine the median and maximum values of the parameter on
    #the given axis.
    mcsd = np.median(param)
    mxcsd = np.max(param)
    dcsd= 10**(np.log10(mcsd) + (np.log10(mxcsd)-np.log10(mcsd))*tolerance)

    pind = np.where(param > dcsd)
    sind = np.where(np.logical_and(space > minx + ext,
                                   space < maxx - ext))
    iind = np.intersect1d(pind[0], sind[0])
    interm = list(zip(iind, param[iind]))

    #It is possible that an atom right at the GB has perfect centrosymmetry;
    #we need to include those. So, we look at the average x-value and then
    #grab all atoms within \pm 10 of that.
    from operator import itemgetter
    mxi = sorted(interm, key=itemgetter(1), reverse=True)[0:15]
    mx = np.mean(space[np.array([m[0] for m in mxi])])
    rind = np.where(np.logical_and(space > mx-width/2., space < mx+width/2.))[0]
    return rind
