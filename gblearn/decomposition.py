"""Methods for decomposing the SOAP vectors into radial and angular
components for analysis.
"""
from gblearn import msg
import numpy as np
epsilon = 5e-5
"""float: finite precision comparison value for RDF norms.
"""
def pissnnl(pissnnl, nspecies, vcutoff, nmax=12, lmax=12):
    """Returns the geometric information for any entries of a pissnnl
    vector higher than the specified cutoff. 

    Args:
        pissnnl (numpy.ndarray): vector to examine.
        nspecies (int): number of species for the SOAP descriptor.
        vcutoff (float): values higher than this will be returned with
          geometry information.
        nmax (int): bandwidth limits for the SOAP descriptor radial basis
          functions.
        lmax (int): bandwidth limits for the SOAP descriptor spherical harmonics.

    Returns:
        list: of tuples with (index, value, (n_i, s_i), (n_j, s_j), l), where `n_i`
          is the `n` value of the radial basis function for species `i`, and `s_i`
          is the value of the species index.
    """
    rs_index = np.zeros((2, nspecies*nmax), int)
    count = 0
    for i in range(1, nspecies+1):
        for a in range(1, nmax+1):
            rs_index[:,count] = [a, i]
            count += 1
            
    ipow = 0
    skip = 0
    result = []
    for ia in range(1, nspecies*nmax+1):
        for jb in range(1, ia+1):
            for l in range(lmax+1):
                if ipow > len(pissnnl):# pragma: no cover
                    skip += 1
                    ipow += 1
                    continue

                if np.abs(pissnnl[ipow]) >= vcutoff:
                    ra = rs_index[0,ia-1]
                    sa = rs_index[1,ia-1]
                    rb = rs_index[0,jb-1]
                    sb = rs_index[1,jb-1]
                    result.append((ipow, pissnnl[ipow], (ra, sa), (rb, sb), l))
                ipow += 1

    if skip > 0:# pragma: no cover
        msg.warn("skipped {} entries in the vector.".format(skip))

    dmsg = "Considered {} entries in P vector (size={})."
    msg.info(dmsg.format(ipow, len(pissnnl)), 2)
    
    from operator import itemgetter
    return sorted(result, key=itemgetter(1), reverse=True)

def fcut(r, rcut, trans_width):
    """Applies the cutoff function to the coefficients so that the
    function will go smoothly to zero at the cutoff.

    Args:
        r (float): radial distance parameter.
        rcut (float): values greater than this are identically zero.
        trans_width(float): the transition to zero at the cutoff happens
          smoothly across this width.

    Returns:
        float: value in [0., 1.] to multiply the coefficient by so
          that all values fall within the specified cutoff.
    """
    if (isinstance(r, list) or isinstance(r, np.ndarray)) and len(r) > 1:
        return np.array([fcut(ri, rcut, trans_width) for ri in r])
    else:
        if r > rcut:
            return 0.
        elif r > rcut-trans_width:
            return 0.5*(np.cos(np.pi*(r-rcut+trans_width)/trans_width) + 1)
        else:
            return 1.0
        
class SOAPDecomposer(object):
    """Implements decompsition of vectors in the SOAP space allowing
    for differing `lmax` and `nmax` values in a single session. Uses
    caching to optimize the decomposition.

    Args:
        nspecies (int): number of species for the SOAP descriptor.
        nmax (int): bandwidth limits for the SOAP descriptor radial basis
          functions.
        lmax (int): bandwidth limits for the SOAP descriptor spherical
          harmonics.
        rcut (float): local environment finite cutoff parameter.
        sigma (float): width parameter for the Gaussians on each atom.
        trans_width (float): distance over which the coefficients in the
            radial functions are smoothly transitioned to zero.

    Attributes:
        alpha (float): constant affecting the width of the Gaussians. Inversely
          proportional to :attr:`sigma`.
        rb (numpy.ndarray): positions of radial basis functions in the
          radial space.
        partitions (dict): keys are `tuple` of (`l` values, inverse); values are the
          corresponding indices that form the partition.
        fcrs (dict): keys are `float` values for radius `r`; values are :func:`fcut`
          evaluated at `r`.
        rbexp (list): of `float` values; the `n`-dependent exponential
          damping factor for radial basis function values.
        rbsph (dict): keys are `tuple` (n,l,r) where `n` and `l` are the integer
          radial and spherical basis function indices and `r` is the `float` value
          at which the spherical bessel function was evaluated.
        cRs (dict): keys are `float` values of radius `r`; values are the
          corresponding coefficient matrices returned by
          :meth:`_c_numeric`.
        aRs (dict): keys are `tuple` (n,l,r) where `n` and `l` are the integer
          radial and spherical basis function indices and `r` is the `float`
          value that the radial functions were evaluated at; values are the
          corresponding coefficients of the radial basis functions.
        transformbasis (numpy.ndarray): transformation matrix needed to account for
          radial basis function positions when calculating the radial function
          coefficients using linear algebra.
    """
    def __init__(self, nspecies=1, lmax=12, nmax=12, rcut=6.0, sigma=0.5,
                 trans_width=0.5):
        self.nspecies = nspecies
        self.lmax = lmax
        self.nmax = nmax
        self.rcut = rcut
        self.sigma = sigma
        self.trans_width = trans_width

        self.alpha = 0.5/self.sigma**2
        self.rb = self._radial_basis()
        self.partitions = {}
        
        #All of these next dictionaries are used only when caching is enabled to
        #speed up the calculation. It speeds things up by about 4 or 5 times.
        self.fcrs = {}
        self.rbexp = [np.exp(-self.alpha*(self.rb[n-1]**2))
                      for n in range(1, nmax+1)]
        self.rbsph = {}        
        self.cRs = {}
        self.aRs = {}
        self.transformbasis = None

    def get_params(self):
        """Returns a dictionary of the constructor parameters needed to
        re-initialize this decomposer instance.
        """
        return {
            "nspecies": self.nspecies,
            "lmax": self.lmax,
            "nmax": self.nmax,
            "rcut": self.rcut,
            "sigma": self.sigma,
            "trans_width": self.trans_width
            }
        
    def _init_numeric(self):
        """Initializes the linear algebra matrices for the radial basis so that
        the coefficients can be solved using :meth:`cnl`.
        """
        from scipy.special import erf
        covbasis = np.zeros((self.nmax, self.nmax))
        overbasis = np.zeros((self.nmax, self.nmax))
        #Get local references to these variables so that we don't need `self`
        #all over in the overbasis calculation below.
        alpha = self.alpha
        rb = self.rb
        
        for i in range(self.nmax):
            for j in range(self.nmax):
                covbasis[j,i] = np.exp(-alpha * (rb[i] - rb[j])**2)
                overbasis[j,i] = (np.exp(-alpha*(rb[i]**2+rb[j]**2))*np.sqrt(2.)* 
                                  alpha**1.5*(rb[i] + rb[j]) + 
                                  alpha*np.exp(-0.5*alpha*(rb[i] - rb[j])**2)*
                                  np.sqrt(np.pi)*
                                  (1. + alpha*(rb[i] + rb[j])**2)*
                                  (1.0 + erf(np.sqrt(alpha/2.0)*(rb[i]+rb[j]))))
                
        overbasis /= np.sqrt(128. * alpha**5)

        from numpy.linalg import cholesky
        choloverlap = cholesky(overbasis)

        for i in range(self.nmax):
            for j in range(i):
                choloverlap[j,i] = 0.0

        from numpy.linalg import solve
        self.transformbasis = solve(covbasis, choloverlap)
        
    def _radial_basis(self):
        """Calculates the radial basis using the initialization
        parameters passed to the class.
        """
        errexp = 10
        cutbasis = self.rcut + self.sigma*np.sqrt(2.*errexp*np.log(10.))
        spacebasis = cutbasis/self.nmax
        rbasis = np.zeros(self.nmax)
        rbasis[0] = 1.
        for i in range(1, self.nmax):
            rbasis[i] = rbasis[i-1] + spacebasis
        return rbasis

    def _c_numeric(self, rij):
        """Calculates the matrix of radial basis function coefficients for all
        `l` and `n` indices up to `self.nmax` and `self.lmax` at once using
        linear algebra for the specified radial distance `r`.

        Args:
            rij (float): distance between atoms `i` and `j` in the structure.
        """
        radial_fun = np.zeros((self.lmax+1, self.nmax))
        radial_fun[0,1] = 1.0

        #Get local references to these variables so that we don't need `self`
        #all over in the overbasis calculation below.
        alpha = self.alpha
        rb = self.rb        
        for n in range(1, self.nmax+1):
            argbess = 2*alpha*rb[n-1]*rij
            ep = np.exp(-alpha*(rij + rb[n-1])**2)
            em = np.exp(-alpha*(rij - rb[n-1])**2)
            #In the loops below, msb prefix refers to modified spherical bessel.
            for l in range(self.lmax+1):
                if l == 0:
                    if argbess == 0.0:
                        msb_fi_ki_l = np.exp(-alpha*(rb[n-1]**2 + rij**2))
                    else:
                        #msb_fi_ki_lm = cosh(arg_bess)/arg_bess
                        #msb_fi_ki_l = sinh(arg_bess)/arg_bess
                        msb_fi_ki_lm = 0.5 * (em + ep) / argbess
                        msb_fi_ki_l  = 0.5 * (em - ep) / argbess
                else:
                    if argbess == 0.0:
                        msb_fi_ki_l = 0.0
                    else:
                        msb_fi_ki_lmm = msb_fi_ki_lm
                        msb_fi_ki_lm = msb_fi_ki_l
                        msb_fi_ki_l = msb_fi_ki_lmm-(2*l-1)*msb_fi_ki_lm/argbess

                radial_fun[l,n-1] = msb_fi_ki_l #* rb[n-1]
        fc = fcut(rij, self.rcut, self.trans_width)
        return np.dot(radial_fun, self.transformbasis)*fc
    
    def cnl(self, n, l, r, cnum=False, fast=True):
        """Returns the coefficient of the radial basis function
        :math:`g_n(r)`.

        Args:
            n (int): index of the radial basis function.
            l (int): index of the spherical harmonic associated with the
              radial function.
            r (float): value at which to evaluate the basis function
              :math:`g_n(r)`.
            cnum (bool): when True, the function is evaluated numerically
              using linear algebra instead of the built-in `numpy`
              functions; can be faster, but the results can have numerical
              noise issues.
            fast (bool): when True, caching is used to speed up evaluation
              of the basis functions between multiple `pissnnl` vectors.
        """
        if cnum:
            if self.transformbasis is None:
                self._init_numeric()
                
            if r not in self.cRs:
                self.cRs[r] = self._c_numeric(r)
            return self.cRs[r][l, n-1]
        else:
            from scipy.special import sph_in
            if fast:
                if r not in self.fcrs:
                    self.fcrs[r] = 4*np.pi*fcut(r, self.rcut, self.trans_width)
                if (n, l, r) not in self.rbsph:
                    besseli = sph_in(l, 2*self.alpha*self.rb[n-1]*r)[0][-1]
                    self.rbsph[(n, l, r)] = besseli*self.rb[n-1]
                if (n, l, r) not in self.aRs:
                    self.aRs[(n, l, r)] = (self.fcrs[r] *
                                           self.rbexp[n-1] *
                                           np.exp(-self.alpha*(r**2)) *
                                           self.rbsph[(n, l, r)])
            
                return self.aRs[(n, l, r)]
            else: #pragma: no cover
                #There is no real sense in not using the cached method since the
                #results are identical. However, we keep it here for reference.
                return (4*np.pi*fcut(r) *
                        np.exp(-alpha*(self.rb[n-1]**2 + r**2)) *
                        sph_in(l, 2*alpha*self.rb[n-1]*r)[0][-1]*self.rb[n-1])

    def apnl(self, dp, rx, cnum=False, fast=True):
        """Returns the analytically/numerically derived coefficient for the
        specified, decomposed pissnnl.
        
        Args:
            dp (tuple): (`int` index in `P`, `float` cnlm, 
              (`int` ni, `int` si), (`int` nj, `int` sj), `int` l). These tuples
              are calculated in bulk for a `pissnnl` vector using
              :func:`gblearn.decomposition.pissnnl`.
            rx (numpy.ndarray): linear space of values to evaluate at.
            cnum (bool): when True, the function is evaluated numerically
              using linear algebra instead of the built-in `numpy`
              functions. See also :meth:`cnl`.
            fast (bool): when True, caching is used to speed up evaluation
              of the basis functions between multiple `pissnnl` vectors.
        """
        ni, si = dp[2]
        nj, sj = dp[3]
        l = dp[4]
        return np.array([(self.cnl(ni, l, r, cnum, fast) *
                          self.cnl(nj, l, r, cnum, fast)) for r in rx])

    def RDF(self, dP, rx, fast=True):
        """Computes the summed radial distribution function for the specified
        decomposed `P` vector.

        Args:
            dp (tuple): (`int` index in `P`, `float` cnlm, 
              (`int` ni, `int` si), (`int` nj, `int` sj), `int` l). These tuples
              are calculated in bulk for a `P` vector using
              :func:`gblearn.decomposition.pissnnl`.
            rx (numpy.ndarray): linear space of values to evaluate at.
        """
        parts = np.zeros((len(dP), len(rx)))
        for i, dPi in enumerate(dP):
            w = np.sign(dPi[1])*np.sqrt(np.sqrt(np.abs(dPi[1])))
            parts[i,:] = w*self.apnl(dPi, rx, fast=fast)
        return np.sum(parts, axis=0)

    def _renorm_p(self, p):
        """Returns the re-normalized coefficient of expansion for use in
        rebuilding the radial and angular distribution functions.
        """
        return np.sign(p)*np.sqrt(np.sqrt(np.abs(p)))
    
    def _ang_part(self, dP):
        """Returns the angular grouping by `l` for the specified P decomposition.
        """
        import pandas as pd
        dsP = pd.DataFrame(dP, columns=["i", "Pij", "nisi", "njsj", "l"])
        dsP["Pij"] = dsP["Pij"].apply(self._renorm_p)
        return dsP.groupby("l").sum()["Pij"].to_dict()

    def ADF(self, dP, ax):
        """Calculates the angular distribution function for the given SOAP
        vector decomposition.

        Args:
            dp (tuple): (`int` index in `P`, `float` cnlm, 
              (`int` ni, `int` si), (`int` nj, `int` sj), `int` l). These tuples
              are calculated in bulk for a `P` vector using
              :func:`gblearn.decomposition.pissnnl`.
            ax (numpy.ndarray): linear space of polar angle values to evaluate
              at.
        """
        from scipy.special import sph_harm
        ang = self._ang_part(dP)
        #scipy defines their harmonics to have `theta` be azimuthal, which is
        #opposite from physics.
        #we set $m = 0$ so that the azimuthal part doesn't contribute at all.
        result = np.zeros(len(ax))
        for l, p in ang.items():
            Ylm = sph_harm(0, l, 0, ax)*np.sqrt(2*l+1)
            #We are interested in the c* c of this value, which is multiplied
            #together to get pissnnl.
            result += p*np.sqrt(np.absolute(Ylm*Ylm.conjugate()))
        return result
    
    def decompose(self, P, vcutoff=1e-5):
        """Decomposes the specified SOAP vector into its radial and angular
        contributions using :func:`gblearn.decomposition.pissnnl`.

        Args:
            P (numpy.ndarray): vector representing the projection of a local
              atomic environment into the SOAP space.
            vcutoff (float): values higher than this will be included in the
              RDF.
        """
        return pissnnl(P, self.nspecies, vcutoff, self.nmax, self.lmax)

    def partition(self, lfilter, inverse=False):
        """Returns a list of indices for a `P` vector that correspond to the
        specified `l` values.
        
        Args:
            lfilter (list): of `l` values to include in the resulting indices.
            inverse (bool): when True, then any `l` values *not* in `lfilter` are
              included instead of the inverse.
        
        Returns:
            (numpy.ndarray): of `int` indices that have the specified `l` values.
        """
        key = (tuple(lfilter), inverse)
        if key not in self.partitions:
            rs_index = np.zeros((2, self.nspecies*self.nmax), int)
            count = 0
            for i in range(1, self.nspecies+1):
                for a in range(1, self.nmax+1):
                    rs_index[:,count] = [a, i]
                    count += 1
                    
            ipow = 0
            result = []
            for ia in range(1, self.nspecies*self.nmax+1):
                for jb in range(1, ia+1):
                    for l in range(self.lmax+1):
                        if ((inverse and l not in lfilter) or
                            (not inverse and l in lfilter)):
                            result.append(ipow)
                        ipow += 1

            self.partitions[key] = np.array(result)

        return self.partitions[key]

def _plot_shells(ax, shells, maxy=1.):
    """Adds the specified shells to axes using a linear colorspace.

    Args:
        ax (matplotlib.axes.Axes): axes to plot on. If not specified, a new
          figure and new axes will be created.
        shells (list): of `float` values for neighbor shells to be added as
          vertical lines to the plot.
    """
    if shells is not None:
        from gblearn.utility import colorspace
        cycols = colorspace(len(shells))
        for shell in shells:
            ax.plot([shell,shell], [0.0,1.1*maxy], color=next(cycols), lw=2)

class DF(object):
    """Represents the Radial or Angluar Distribution Function of a SOAP vector.

    Args:
        dP (list): of tuples; result of calling
          :func:`gblearn.decomposition.pissnnl` on the SOAP vector with
          :math:`l=0` or :math:`l>0` components set to zero.
        catom (bool): when True, this DF is for the *central* atom only (i.e.,
          :math:`l=0` components only).
        x (numpy.ndarray): domain in radial/angular space on which to sample
          the function.    
        decomposer (SOAPDecomposer): instance used to decompose the SOAP vector; has
          configuration information for SOAP parameters and caches for rapid
          evaluation of the basis functions.
        calculate (bool): when True, the distribution function will be calculated
          upon initialization.
    Attributes:
        df (numpy.ndarray): distribution function values corresponding to
          :attr:`x`.
    """
    def __init__(self, dP, catom, x, decomposer, radial=True, calculate=True):
        self.dP = dP
        self.catom = catom
        self.x = x
        self.decomposer = decomposer
        #We don't necessarily always want to re-calculate on construction. In
        #that case, the instance is just a useful container for its basic
        #parameters.
        if calculate:
            if radial:
                self.df = decomposer.RDF(dP, x)
            else:
                self.df = decomposer.ADF(dP, x)
        else:
            self.df = None

        self._norm = None
        """float: :func:`numpy.linalg.norm` of the DF vector.
        """
        self.dtype = "R" if radial else "A"
        """str: since this could be a radial *or* and angular DF, the type of
        the distribution function.
        """

    def __eq__(self, other):
        if isinstance(other, DF):
            return np.allclose(self.df, other.df)
        else:
            return False
        
    @property
    def norm(self):
        """Calculates the L2 norm of the distribution function values as if they
        were a vector in some space.
        """
        if self._norm is None:
            from numpy.linalg import norm as lnorm
            self._norm = lnorm(self.df)
        return self._norm
        
    def __sub__(self, other):
        """Determines the distance between two DFs using the dot product
        metric.
        """
        if not hasattr(other, "dtype") or self.dtype != other.dtype:
            raise TypeError("Can only calculate distance between two DFs. of "
                            "the same type.")
        return abs(np.dot(self.df, other.df)/(self.norm*other.norm)-1.)

    @staticmethod
    def from_file(filename, decomposer=None, x=None):
        """Restores a DF from file.
        
        Args:
            filename (str): path to the file that was created by
              :meth:`DF.save`.
            decomposer (SOAPDecomposer): instance used to decompose the SOAP vector; has
              configuration information for SOAP parameters and caches for rapid
              evaluation of the basis functions.
            x (numpy.ndarray): common domain values shared by a collection; used if
              the file doesn't explicitly specify the domain values.
        """
        from six.moves.cPickle import load
        with open(filename, 'rb') as f:
            data = load(f)
        return DF.from_dict(data, decomposer, x)

    @staticmethod
    def from_dict(data, decomposer=None, x=None):
        """Restores a DF from a serialized dict (i.e., one returned by
        :meth:`serialize`).
        
        Args:
            data (dict): result of calling :meth:`serialize`.
            decomposer (SOAPDecomposer): instance used to decompose the SOAP vector; has
              configuration information for SOAP parameters and caches for rapid
              evaluation of the basis functions.
            x (numpy.ndarray): if this DF is constructed from a parent collection,
              the common domain sample vector to use.
        """
        if x is not None and data["x"] is None:
            data["x"] = x
        result = DF(data["dP"], data["catom"], data["x"], decomposer,
                    data["dtype"]=="R", calculate=False)
        result.df = data["df"]
        return result
    
    def serialize(self, withdecomp=True, commonx=None, withdP=True):
        """Returns a serializable dictionary that represents this DF.

        Args:
            withdecomp (bool): when True, include the parameters of the SOAP
              decomposer in the dict.
            commonx (numpy.ndarray): xf this DF is part of a collection, the domain
              values will be the same for every DF; in that case we don't need to
              include the domain in the serialization. This is the parent
              collections common radial vector. If the DFs is the same, it won't
              be serialized. If unspecified, the domain values are serialized.
            withdP (bool): when True, the large, decomposed SOAP vector values are
              serialized. Necessary to completely reconstruct the representation,
              but not necessary for most analysis.
        """
        result = {
            "dP": self.dP if withdP else None,
            "catom": self.catom,
            "x": (None if commonx is not None
                  and np.allclose(commonx, self.x) else self.x),
            "dtype": self.dtype,
            "df": self.df
        }
        if withdecomp:
            result["decomposer"] = self.decomposer.get_params()
        return result
    
    def save(self, target):
        """Saves the DF to disk.

        Args:
            target (str): path to save the vector to.
        """
        from six.moves.cPickle import dump
        data = self.serialize()
        with open(target, 'wb') as f:
            dump(data, f)
    
    def same(self, other, epsilon_=None):
        """Tests whether the specifed DF is similar to this one.

        Args:
            other (DF): DF to compare similarity to.
            epsilon_ (float): override the global (default) value
              :data:`~gblearn.decomposition.epsilon` for determining if the DFs
              are similar.

        Returns:
            bool: True when both of the DFs are identical within tolerance.
        """
        if epsilon_ is None:
            return self-other < epsilon
        else:
            return self-other < epsilon_
    
    def plot(self, ax=None, savefile=None, shells=None, opacity=1., color='b',
             title=None, xlabel=None, ylabel=None):
        """Plots the DF.

        Args:
            ax (matplotlib.axes.Axes): axes to plot on. If not specified, a new
              figure and new axes will be created. If axes are specifed, *no* `x`
              or `y` labels or plot titles will be added.
            savefile (str): path to a file if the figure should be saved.
            shells (list): of `float` values for neighbor shells to be added as
              vertical lines to the plot.
            opacity (float): transparency to use for the DF plot.
            color: any color option received by :func:`matplotlib.pyplot.plot`.
            title (str): override the default title of the plot.
            xlabel (str): override the default (generic, not very helpful) label for
              the x-axis.
            ylabel (str): override the default label for the y-axis.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            plt.figure()
            axset = plt
        else:
            axset = ax

        axset.plot(self.x, self.df)
        if ax is None:
            if title is None:
                plt.title("Distribution Function of SOAP Vector")
            else:
                plt.title(title)
            if xlabel is None:
                plt.xlabel("Distance (unknown units)")
            else:
                plt.xlabel(xlabel)
            if ylabel is None:
                plt.ylabel("Accumulated Density")
            else:
                plt.ylabel(ylabel)

        _plot_shells(axset, shells, np.max(self.df))
        
        if savefile is not None:
            plt.savefig(savefile)
            
        from gblearn.base import testmode
        if not testmode:# pragma: no cover
            plt.show()
            
        return axset

class DFCollection(object):
    """Represents a collection of Distrubition Functions. A DF can only
    be added once to a collection (object instance comparison). Two identical
    DFs (in the physical sense) can belong to the same collection.

    Args:
        dfs (list): of :class:`DF` to intialize the collection with.
        counts (list): of `int` specifying how many DFs of each type are in the
          collection. For example, `self.counts[3]` says that we have three DFs
          in the collection that are identical to the DF in `self.dfs[3]`,
          though we only keep one of them.

    Attributes:
        dfs (list): of :class:`DF` to intialize the collection with.
        counts (list): of `int` specifying how many DFs of each type are in the
          collection. For example, `self.counts[3]` says that we have three DFs
          in the collection that are identical to the DF in `self.dfs[3]`,
          though we only keep one of them.
        label (str): user-defined label for the collection.
        tags (dict): user-defined tags to identify this collection in a larger
          analysis.
    """
    def __init__(self, dfs=None, counts=None):
        if dfs is None:
            self.dfs = []
        else:
            self.dfs = dfs

        if counts is not None:
            self.counts = counts
        else:
            self.counts = [1 for i in range(len(self.dfs))]
            
        self._unique = False
        """bool: when True, this collection has already been tested for
        uniqueness. Any subsequent additions will be tested for uniqueness
        before adding them in.
        """
        self._original = len(self.dfs)
        """int: number of DFs in the original list (before non-unique ones were
        removed). Useful for statistics.
        """
        self.label = None
        self.tags = {}
        self._average = None
        """numpy.ndarray: averaged distribution function for the entire
        collection.
        """

    def __eq__(self, other):
        """Tests the equality of this DFCollection with another.
        """
        if isinstance(other, DFCollection):
            return all([sdf == odf for sdf, odf in zip(self, other)])
        else:
            return False    
        
    def __str__(self):
        title = "{0}DFCollection {1} with {2:d} items"
        label = "" if self.label is None else "({})".format(self.label)
        if len(self) > 0:
            dtype = self[0].dtype[0].upper()
        else:# pragma: no cover
            #This is just a sanity check. It should never happen unless someone
            #is hacking around in the code.
            dtype = ""
        return title.format(dtype, label, len(self))
    def __repr__(self):
        return str(self)
    
    def __add__(self, other):
        """Concatenate two collections together. This doesn't enforce any
        uniqueness constraints. Tags and labels are *not* updated.
        """
        if not isinstance(other, type(self)):
            raise TypeError("Only DFs of the same type can be combined.")

        dfs = self.dfs + other.dfs
        counts = self.counts + other.counts
        return self.__class__(dfs, counts)

    def __iadd__(self, other):
        """Extend this collection with the other one.
        """
        if not isinstance(other, type(self)):
            raise TypeError("Only DFs of the same type can be combined.")
        self.dfs.extend(other.dfs)
        self.counts.extend(other.counts)
        self._unique = False
        self._original += other._original
        if self.label is None:
            if other.label is not None:
                self.label = other.label
        else:
            if other.label is not None:
                self.label += "+" + other.label
        self.tags.update(other.tags)
        self._average = None
        return self
    
    def __iter__(self):
        return iter(self.dfs)
    def __contains__(self, df):
         return df in self.dfs
    def __len__(self):
        return len(self.dfs)
    def __getitem__(self, index):
        if isinstance(index, slice):
            dfs = self.dfs[index]
            counts = self.counts[index]
            return self.__class__(dfs, counts)
        else:
            return self.dfs[index]
    def __setitem__(self, index, value):
        if len(self) > 0:
            if not isinstance(value, type(self.dfs[0])):
                emsg = "Only DFs of type {} can be placed in this collection."
                raise TypeError(emsg.format(type(self)))
        self.dfs[index] = value
        self.counts[index] = 1
        self._average = None

    @staticmethod
    def dfs_from_soap(collection, ctype, resolution=75, catom=False):
        """Creates a DF collection from the specified SOAPVector collection
        using the :math:`l>0` components.

        Args:
            collection (SOAPVectorCollection): collection of SOAP vectors to
              initialize from.
            ctype (str): one of ['RDF', 'ADF'], specifies the kind of collection
              to initialize.
            resolution (int): number of points to sample in the radial domain.
            catom (bool): when True, the DFs are constructed using the density of
              the central atom.
        """
        if ctype not in ["RDF", "ADF"]:
            raise ValueError("'ctype' must be either 'RDF' or 'ADF'.")
        
        from gblearn.base import nprocs
        if ctype == "RDF":
            x = np.linspace(0., collection.decomposer.rcut, resolution)
        else:
            x = np.linspace(0., np.pi, resolution)
            
        if nprocs is not None:
            from multiprocessing import Pool
            mpool = Pool(nprocs)
            compute = [mpool.apply_async(_multiproc_execute, (v, ctype, (x, catom)))
                       for v in collection]
            dfs = []
            for j, df in enumerate(compute):
                dfs.append(df.get())
        else:
            dfs = [getattr(v, ctype)(x) for v in collection]
        return dfs
        
    @staticmethod
    def from_file(filename):
        """Restores a DFCollection from file.
        
        Args:
            filename (str): path to the file that was created by
              :meth:`DF.save`.
        """
        from six.moves.cPickle import load
        with open(filename, 'rb') as f:
            data = load(f)
        return DFCollection.from_dict(data)

    @staticmethod
    def from_dict(data):
        """Restores a DFCollection from a serialized dict (i.e., one returned by
        :meth:`serialize`).
        
        Args:
            data (dict): result of calling :meth:`serialize`.
        """
        dfs = []
        decomposer = SOAPDecomposer(**data["decomposer"])
        for dfdata in data["dfs"]:
            dfs.append(DF.from_dict(dfdata, decomposer, data["x"]))

        if len(dfs) > 0:
            dtype = dfs[0].dtype
        else:# pragma: no cover
            #We default to radial distributions.
            dtype == "R"

        if dtype == "R":
            result = RDFCollection(dfs, data["counts"])
        else:
            result = ADFCollection(dfs, data["counts"])
        result.label = data["label"]
        result.tags = data["tags"]
        return result
    
    def serialize(self, withdP=False):
        """Returns a serializable dictionary that represents this DFCollection.

        Args:
            withdP (bool): when True, the large, SOAP decomposition from which the
              DF was constructed is also included. Wasteful, but necessary to
              completely represent the DF.
        """
        decomposer = None
        if len(self) > 0:
            decomposer = self.dfs[0].decomposer
            x = self.dfs[0].x
            
        dfs = []
        for df in self:
            dfs.append(df.serialize(withdecomp=False, commonx=x, withdP=withdP))
            
        result = {
            "dfs": dfs,
            "counts": self.counts,
            "label": self.label,
            "tags": self.tags,
            "x": x
        }
        if decomposer is not None:
            result["decomposer"] = decomposer.get_params()
        else:# pragma: no cover
            result["decomposer"] = {}            
        return result
    
    def save(self, target, withdP=False):
        """Saves the DF to disk.

        Args:
            target (str): path to save the vector to.
            withdP (bool): when True, the large, SOAP decomposition from which the
              DF was constructed is also included. Wasteful, but necessary to
              completely represent the DF.

        """
        from six.moves.cPickle import dump
        data = self.serialize(withdP=withdP)
        with open(target, 'wb') as f:
            dump(data, f)

    @property
    def average(self):
        """Returns the averaged radial distribution function for the collection.
        """
        if self._average is None:
            self._average = sum([df.df for df in self])/len(self)
        return self._average

    def project(self, other, epsilon_=None):
        """Projects the RDFs in self into the unique RDFs of other.

        Returns:
            tuple: (:class:`numpy.ndarray` projection, `list`
            exceptions); `projection` is the number of each kind of
            RDF in other with `sum(result) == len(self)`; `exceptions`
            are indices in `self` for which there was no similar :class:`DF` in
            other.
        """
        result = np.zeros(len(other), dtype=int)
        exceptions = []
        for si, df in enumerate(self):
            for oi, bdf in enumerate(other):
                if df.same(bdf, epsilon_):
                    result[oi] += self.counts[si]
                    break
            else:
                exceptions.append(si)
                
        return (result, exceptions)
    
    def refine(self, other, epsilon_=None):
        """Refines this DFCollection to include all DFs in `other` that are not
        already in `self`. Also reduces existing DFs in `self` to be unique.

        Args:
            other (DFCollection): another collection to augment this one with.
        """
        if not isinstance(other, type(self)):
            raise TypeError("Can only refine collections of the same type.")

        #Make a copy of the distribution functions in self; make sure they are
        #unique to begin with.
        dfs = self.unique(epsilon_)
        for io, df in enumerate(other):
            for xi, xdf in enumerate(dfs):
                if xdf.same(df, epsilon_):
                    dfs.counts[xi] += other.counts[io]
                    break
            else:
                #We don't use :meth:`add` here because we have already
                #checked uniqueness and we want to keep the number of
                #identical DFs accurate.
                dfs.dfs.append(df)
                dfs.counts.append(other.counts[io])

        dfs._original = len(self) + len(other)
        dfs._unique = True
        return dfs
        
    def histogram(self, epsilon_=None, savefile=None, **kwargs):
        """Determines how many of each kind of DF are in this collection (using
        the default comparison). Plots the histogram of unique values. This is
        equivalent to calling :meth:`DFCollection.unique.histogram`.

        Args:
            epsilon_ (float): override the global (default) value
              :data:`~gblearn.decomposition.epsilon` for determining if the DFs
              are similar.
            kwargs (dict): arguments will be passed directly to
              :func:`matplotlib.pyplot.bar` for the plotting.
        Returns:
            DFCollection: of the unique values; same result as
              :meth:`DFCollection.unique`.
        """
        if not self._unique:
            result = self.unique(epsilon_)
        else:
            result = self
            
        import matplotlib.pyplot as plt
        total = sum(result.counts)
        plt.figure()
        plt.bar(np.arange(1, len(self)+1), np.array(self.counts, float)/total)
        htitle = "Histogram of Unique DFs in Collection ({}/{})"
        plt.title(htitle.format(len(self), total))
        plt.xlabel("DF Number")
        plt.ylabel("Percentage of Identical DFs ")

        if savefile is not None:
            plt.savefig(savefile)

        from gblearn.base import testmode
        if not testmode:# pragma: no cover
            plt.show()       
        
    def unique(self, epsilon_=None):
        """Returns only the unique DFs in this collection.

        Args:
            epsilon_ (float): override the global (default) value
              :data:`~gblearn.decomposition.epsilon` for determining if the DFs
              are similar.

        Returns:
            DFCollection: of unique DFs within the specified tolerance.
        """
        if self._unique:
            return self
        
        if len(self) == 0:
            #Construct a new, empty collection.
            return self.__class__()
        
        dfs = [self.dfs[0]]
        counts = [self.counts[0]]
        for si, df in enumerate(self[1:]):
            for i, kept in enumerate(dfs):
                if kept.same(df, epsilon_):
                    counts[i] += self.counts[si+1]
                    break
            else:
                dfs.append(df)
                counts.append(self.counts[si+1])

        #We want the return type to be the same as the current instance (radial
        #or angular).
        result = self.__class__(dfs, counts)
        result._unique = True
        result._original = len(self)
        return result
        
    def plot(self, ax=None, savefile=None, shells=None, color='b', title=None,
             xlabel=None, ylabel=None, withavg=False):
        """Plots the collection of DFs on the same axes. Opacity of the line is
        chosen based on how many DFs in the collection are identical to that
        one (within tolerance). This is accomplished by calling
        :meth:`DFCollection.histogram`.

        Args:
            ax (matplotlib.axes.Axes): axes to plot on. If not specified, a new
              figure and new axes will be created.
            savefile (str): path to a file if the figure should be saved.
            shells (list): of `float` values for neighbor shells to be added as
              vertical lines to the plot.
            color: any color option received by :func:`matplotlib.pyplot.plot`.
            title (str): override the default title of the plot.
            xlabel (str): override the default (generic, not very helpful) label for
              the x-axis.
            ylabel (str): override the default label for the y-axis.
            withavg (bool): when True, the average DF for the whole collection is also
              plotted.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            plt.figure()
            axset=plt
        else:
            axset=ax

        cmax = float(max(self.counts))
        total = sum(self.counts)
        nalpha = 0.85 if cmax/total > 0.33 else 0.65
        maxy = 1.
        for di, df in enumerate(self.dfs):
            alpha=nalpha*self.counts[di]/cmax
            axset.plot(df.x, df.df, color=color, alpha=alpha)
            maxy_ = np.max(df.df)
            if maxy_ > maxy:
                maxy = maxy_

        if withavg and len(self) > 0:
            x = self.dfs[0].x
            axset.plot(x, self.average, 'r-')
            maxy_ = np.max(self.average)
            if maxy_ > maxy:# pragma: no cover
                maxy = maxy_

        if len(self) > 0:
            dtype = self.dfs[0].dtype
            unit = "Ang." if dtype == "R" else "Rad."
            tstr = "Radial" if dtype == "R" else "Angular"
        else:# pragma: no cover
            unit = "unknown units"
            tstr = ""
            
        if ax is None:
            if title is None:
                plt.title("{} Distribution Function of Collection".format(tstr))
            else:
                plt.title(title)
            if xlabel is None:
                plt.xlabel("Distance ({})".format(unit))
            else:
                plt.xlabel(xlabel)
            if ylabel is None:
                plt.ylabel("Accumulated Density")
            else:
                plt.ylabel(ylabel)

        _plot_shells(axset, shells, maxy)
        
        if savefile is not None:
            plt.savefig(savefile)

        from gblearn.base import testmode
        if not testmode:# pragma: no cover
            plt.show()
        return axset
    
    def add(self, df):
        """Add a new DF to the collection.

        Args:
            df (DF): instance to add; if it is already in the collection, nothing
              will happen.
        """
        if not isinstance(df, DF):
            raise TypeError("Only DF objects can be added to DF collections.")
        if len(self) > 0 and self[0].dtype != df.dtype:
            emsg = "Only DFs of type {}DF can be added to this collection."
            raise TypeError(emsg.format(self[0].dtype))
        
        do_add = False
        if self._unique:
            #We need to make sure that the DF is unique before adding it.
            for xi, xdf in enumerate(self):
                if not xdf.same(df):# pragma: no cover
                    #I can't seem to get this line to log, no matter what I
                    #throw at it...
                    break
            else:
                do_add = True    
        else:
            if df not in self.dfs:
                do_add = True

        if do_add:
            self.dfs.append(df)
            self.counts.append(1)
            self._average = None
        
    def remove(self, df):
        """Remove an DF from the collection.
        
        Args:
            rdf (DF): instance to remove; if it isn't already in the
              collection, nothing will happen.
        """
        if not isinstance(df, DF):
            return
        if df in self.dfs:
            idf = self.dfs.index(df)
            self.dfs.remove(df)
            del self.counts[idf]
            self._average = None

def _multiproc_execute(obj, method, args):# pragma: no cover
    """Executes the instance method on the given object.

    Args:
        obj: object to execute the method on.
        method (str): name of the method to execute.
        args (tuple): arguments to pass; method called as obj.method(*args).
    """
    #We don't cover this because it won't necessarily be caught be the coverage
    #program that is running on the main thread; it may only execute on other
    #threads.
    if hasattr(obj, method):
        return getattr(obj, method)(*args)
            
class RDFCollection(DFCollection):
    """Represents a radial distribution function collection. See also the
    comments for :class:`DFCollection`.
    """
    @staticmethod
    def from_soap(collection, resolution=75, catom=False):
        """Creates an RDF collection from the specified SOAPVector collection
        using the :math:`l>0` components.

        Args:
            collection (SOAPVectorCollection): collection of SOAP vectors to
              initialize from.
            resolution (int): number of points to sample in the radial domain.
            catom (bool): when True, the DFs are constructed using the density of
              the central atom.
        """
        dfs = DFCollection.dfs_from_soap(collection, "RDF", resolution, catom)
        return RDFCollection(dfs)

class ADFCollection(DFCollection):
    """Represents a collection of Angular Distrubition Functions.  See also the
    comments for :class:`DFCollection`.
    """
    @staticmethod
    def from_soap(collection, resolution=75, catom=False):
        """Creates an ADF collection from the specified SOAPVector collection
        using the :math:`l>0` components.

        Args:
            collection (SOAPVectorCollection): collection of SOAP vectors to
              initialize from.
            resolution (int): number of points to sample in the angular domain.
            catom (bool): when True, the DFs are constructed using the density of
              the central atom.
        """
        #We only have degrees of freedom in the polar angle; the azimuthal got
        #average out to give us rotational invarance.
        dfs = DFCollection.dfs_from_soap(collection, "ADF", resolution, catom)
        return ADFCollection(dfs)
    
class SOAPVector(object):
    """Wrapper for a SOAP vector that facilitates decomposition, plotting and
    analysis in the SOAP space.

    Args:
        P (numpy.ndarray): vector representing the projection of a local atomic
          environment into the SOAP space.
        decomposer (SOAPDecomposer): instance used to decompose the SOAP vector; has
          configuration information for SOAP parameters and caches for rapid
          evaluation of the basis functions.

    Attributes:
        dnP (list): of tuples; result of calling
          :func:`gblearn.decomposition.pissnnl` on the SOAP vector with `l=0`
          components set to zero.
        dcP (list): of tuples; result of calling
          :func:`gblearn.decomposition.pissnnl` on the SOAP vector with `l>0`
          components set to zero.
        nP (numpy.ndarray): a copy of `P` with `l=0` components set to zero.
        cP (numpy.ndarray): a copy of `P` with `l>0` components set to zero.
        rx (numpy.ndarray): sample points in the radial domain at which :attr:`cRDF`
          and :attr:`nRDF` were sampled.
        cRDF (DF): radial distribution function for the central atom.
        nRDF (DF): radial distribution function for all the neighbors of the
          central atom.
        rx (numpy.ndarray): sample points in the angular domain at which
          :attr:`cADF` and :attr:`nADF` were sampled.
        cADF (DF): angular distribution function for the central atom.
        nADF (DF): angular distribution function for all the neighbors of the
          central atom.
    """
    def __init__(self, P, decomposer):
        self.P = P
        self.cP = None
        self.nP = None
        
        self.decomposer = decomposer
        self.dcP = None
        self.dnP = None
        
        self.rx = None
        """numpy.ndarray: cached value of the radial values that we evaluated
        the RDF at; useful if the RDF is requested repeatedly for the same `r`
        values.
        """
        self.cRDF = None
        """numpy.ndarray: cached values of the RDF for the cached
        value in :attr:`rx` with :math:`l=0`.
        """
        self.nRDF = None
        """numpy.ndarray: cached values of the RDF for the cached
        value in :attr:`rx` with :math:`l>0`.
        """

        self.ax = None
        """numpy.ndarray: cached value of the angular values that we evaluated
        the ADF at; useful if the ADF is requested repeatedly for the same
        `theta` values.
        """
        self.cADF = None
        """numpy.ndarray: cached values of the ADF for the cached
        value in :attr:`ax` with :math:`l=0`.
        """
        self.nADF = None
        """numpy.ndarray: cached values of the ADF for the cached
        value in :attr:`ax` with :math:`l>0`.
        """
        
    def __eq__(self, other):
        """Tests equality between two SOAP vectors by comparing *only*
        their P vectors.
        """
        return np.allclose(self.P, other.P)

    def equal(self, other):
        """Performs a more thorough equality check by also comparing
        the SOAP decomposers and RDFs.
        """
        Pok = self == other
        Dok = (self.decomposer.get_params() == other.decomposer.get_params())
        if self.rx is not None:
            if other.rx is not None:
                Rok = (np.allclose(self.rx, other.rx) and
                       ((self.cRDF is None and other.cRDF is None) or
                        (self.cRDF is not None and
                         other.cRDF is not None and
                         np.allclose(self.cRDF.df, other.cRDF.df))) and
                       ((self.nRDF is None and other.nRDF is None) or
                        (self.nRDF is not None and
                         other.nRDF is not None and
                         np.allclose(self.nRDF.df, other.nRDF.df))))
            else:# pragma: no cover
                Rok = False
        else:
            Rok = True
        if self.ax is not None:
            if other.ax is not None:
                Aok = (np.allclose(self.ax, other.ax) and
                       ((self.cADF is None and other.cADF is None) or
                        (self.cADF is not None and
                         other.cADF is not None and
                         np.allclose(self.cADF.df, other.cADF.df))) and
                       ((self.nADF is None and other.nADF is None) or
                        (self.nADF is not None and
                         other.nADF is not None and
                         np.allclose(self.nADF.df, other.nADF.df))))
            else:# pragma: no cover
                Aok = False
        else:
            Aok = True
            
        return Pok and Dok and Rok and Aok
    
    @staticmethod
    def from_element(element, index=0, **kwargs):
        """Returns the SOAP vector for a pure element.

        Args:
            element (str): element name. Must be one of 
              ["Ni", "Cr", "Mg"].
            index (int): for multi-atom bases, which atom to represent
              in the elemental crystal.
            nmax (int): bandwidth limits for the SOAP descriptor radial basis
              functions.
            lmax (int): bandwidth limits for the SOAP descriptor spherical
              harmonics.
            rcut (float): local environment finite cutoff parameter.
            sigma (float): width parameter for the Gaussians on each atom.
            trans_width (float): distance over which the coefficients in the
                radial functions are smoothly transitioned to zero.    
        """
        from gblearn.elements import pissnnl
        P = pissnnl(element, **kwargs)[index]
        decomposer = SOAPDecomposer(1, **kwargs)
        return SOAPVector(P, decomposer)
        
    @staticmethod
    def from_file(filename, decomposer=None, rx=None, ax=None):
        """Restores a SOAP vector from file.
        
        Args:
            filename (str): path to the file that was created by
              :meth:`SOAPVector.save`.
            decomposer (SOAPDecomposer): if this serialization was part of a
              higher level collection, then the decomposer would not have been
              included at the item level. For reconstruction, it is passed in by
              the collection.
            rx (numpy.ndarray): as for `decomposer`, but with the values in the
              domain.
            ax (numpy.ndarray): as for `decomposer`, but with the angular values
              in the domain.
        """
        from six.moves.cPickle import load
        with open(filename, 'rb') as f:
            data = load(f)
        return SOAPVector.from_dict(data, decomposer, rx, ax)

    @staticmethod
    def from_dict(data, decomposer_=None, rx=None, ax=None):
        """Restores a SOAP vector from a serialized dict (i.e., one returned by
        :meth:`serialize`).
        
        Args:
            data (dict): result of calling :meth:`serialize`.
            decomposer_ (SOAPDecomposer): if this serialization was part of a
              higher level collection, then the decomposer would not have been
              included at the item level. For reconstruction, it is passed in by
              the collection.
            rx (numpy.ndarray): as for `decomposer`, but with the values in the
              domain.
            ax (numpy.ndarray): as for `decomposer`, but with the angular values
              in the domain.
        """
        if decomposer_ is not None:
            decomposer = decomposer_
        else:
            decomposer = SOAPDecomposer(**data["decomposer"])
            
        result = SOAPVector(data["P"], decomposer)
        result.dcP = data["dcP"]
        result.dnP = data["dnP"]
        if rx is not None and data["rx"] is None:# pragma: no cover
            result.rx = rx
        else:
            result.rx = data["rx"]
        if ax is not None and data["ax"] is None:# pragma: no cover
            result.ax = ax
        else:
            result.ax = data["ax"]
            
        if data["cRDF"] is not None:
            result.cRDF = DF(data["dcP"], True, result.rx, decomposer,
                             calculate=False)
            result.cRDF.df = data["cRDF"]
        if data["nRDF"] is not None:
            result.nRDF = DF(data["dnP"], False, result.rx, decomposer,
                             calculate=False)
            result.nRDF.df = data["nRDF"]
        if data["cADF"] is not None:
            result.cADF = DF(data["dcP"], True, result.ax, decomposer,
                             calculate=False)
            result.cADF.df = data["cADF"]
        if data["nADF"] is not None:
            result.nADF = DF(data["dnP"], False, result.ax, decomposer,
                             calculate=False)
            result.nADF.df = data["nADF"]

        return result

    def serialize(self, withdecomp=True, commonrx=None,  withdP=True, commonax=None):
        """Returns a serializable dictionary that represents this SOAP Vector.

        Args:
            withdecomp (bool): when True, include the parameters of the SOAP
              decomposer in the dict.
            commonrx (numpy.ndarray): xf this DF is part of a collection, the radial
              values will be the same for every DF; in that case we don't need to
              include the domain in the serialization. This is the parent
              collections common radial vector. If the DFs is the same, it won't
              be serialized. If unspecified, the domain values are serialized.
            withdP (bool): when True, the large, decomposed SOAP vector values are
              serialized. Necessary to completely reconstruct the representation,
              but not necessary for most analysis.
            commonax (numpy.ndarray): same as for `commonrx` but for the angular
              domain.
        """
        result = {
            "P": self.P,
            "dcP": self.dcP if withdP else None,
            "dnP": self.dnP if withdP else None,
            "rx": None,
            "ax": None,
            "cRDF": None,
            "nRDF": None,
            "cADF": None,
            "nADF": None
        }
        if withdecomp:
            result["decomposer"] = self.decomposer.get_params()
        else:
            result["decomposer"] = {}

        if self.rx is not None:
            if commonrx is None or self.rx is not commonrx:
                result["rx"] = self.rx
        if self.ax is not None:
            if commonax is None or self.ax is not commonax:
                result["ax"] = self.ax
                
        if self.cRDF is not None:
            result["cRDF"] = self.cRDF.df
        if self.nRDF is not None:
            result["nRDF"] = self.nRDF.df
        if self.cADF is not None:
            result["cADF"] = self.cADF.df
        if self.nADF is not None:
            result["nADF"] = self.nADF.df

        return result
    
    def save(self, target):
        """Saves the SOAP vector to disk so that RDF and decomposition
        operations don't have to be re-executed later.

        Args:
            target (str): path to save the vector to.
        """
        from six.moves.cPickle import dump
        data = self.serialize()
        with open(target, 'wb') as f:
            dump(data, f)

    def _get_DF(self, x, dfname, xname, catom=False):
        """Gets the specified distribution function, taking the cache into
        account.

        Args:
            x (numpy.ndarray): values in the radial or angular domain to sample
              at.
            dfname (str): name of the attribute on `self` corresponding to the
              cached distribution function to consider before 
              constructing a new one. *Modified* by this routine if a DF is
              constructed.
            xname (str): name of attribute on `self` corresponding to cached
              sample values to consider before
              constructing a new :class:`DF`. *Modified* by this routine if a DF
              is constructed.
            catom (bool): when True, consider the central atom's distribution
              function.
        """
        from numpy import allclose
        cachex = getattr(self, xname)
        cacheDF = getattr(self, dfname)
        
        if (cachex is not None and cachex.shape == x.shape
            and allclose(x, cachex)):
            if cacheDF is not None:
                return cacheDF
        else:
            setattr(self, xname, x)
            setattr(self, dfname, None)

        P = dP = None
        if catom:
            P = self.cP
            dP = self.dcP
        else:
            P = self.nP
            dP = self.dnP
            
        if P is None:
            lrest = self.decomposer.partition([0], catom)
            P = self.P.copy()
            P[lrest] = 0.
            if catom:
                self.cP = P
            else:
                self.nP = P
            
        if dP is None:
            dP = self.decomposer.decompose(P)
            if catom:
                self.dcP = dP
            else:
                self.dnP = dP

        setattr(self, dfname, DF(dP, catom, x, self.decomposer, xname=="rx"))
        return getattr(self, dfname)
            
    def ADF(self, ax, catom=False):
        """Returns the angular distribution function for the SOAP vector.

        Args:
            ax (numpy.ndarray): domain in the angular space to evaluate the ADF
              at.
            catom (bool): when True, the ADF of the central atom is returned;
              otherwise, the central atom is ignored and the neighbor ADF is
              returned.
        """
        if catom:
            return self._get_DF(ax, "cADF", "ax", catom=True)
        else:
            return self._get_DF(ax, "nADF", "ax", catom=False)
            
    def RDF(self, rx, catom=False):
        """Returns the *total* radial distribution function for the SOAP vector.

        Args:
            rx (numpy.ndarray): domain in the radial space to evaluate the RDF
              at.
            catom (bool): when True, the RDF of the central atom is returned;
              otherwise, the central atom is ignored and the neighbor RDF is
              returned.
        """
        if catom:
            return self._get_DF(rx, "cRDF", "rx", catom=True)
        else:
            return self._get_DF(rx, "nRDF", "rx", catom=False)

class SOAPVectorCollection(object):
    """Represents a collection of SOAP vectors that are logically combined (for
    example the local environments at a grain boundary are all part of the grain
    boundary). The collection allows aggregate quantities to be calculated more
    easily.

    Args:
        Pij (numpy.ndarray): a matrix of SOAP vectors, where each *row* is a
          vector.
        decomposer (SOAPDecomposer): instance used to decompose the SOAP
          vectors; has configuration information for SOAP parameters and caches
          for rapid evaluation of the basis functions.
        calculate (bool): when True, the SOAPVectors are calculated based on `P`;
          otherwise, they are left as an empty list.
        kwargs (dict): arguments that can be passed to :class:`SOAPDecomposer`
          constuctor.

    Examples:
        Create a collection of SOAP vectors and then collapse them into a set of
        unique ones (an instance of :class:`RDFCollection`).

        >>> from gblearn.decomposition import SOAPVectorCollection as SVC
        >>> from numpy import load
        >>> P = load("pissnnl.npy")
        >>> svc = SVC(P, lmax=18, nmax=18)
        >>> urdfs = svc.unique(rdf=True)
    """
    def __init__(self, Pij=None, decomposer=None, calculate=True, **kwargs):
        self.P = Pij
        if decomposer is None:
            self.decomposer = SOAPDecomposer(**kwargs)
        else:
            self.decomposer = decomposer

        if self.P is not None and calculate:
            self.vectors = [SOAPVector(P, self.decomposer) for P in self.P]
        else:
            self.vectors = []

        self._rdfs = {}
        """dict: keys are `int` sampling resolutions; values are *unique* RDFs
        (:class:`RDFCollection`) in this :class:`SOAPVectorCollection`.
        """
        self._adfs = {}
        """dict: keys are `int` sampling resolutions; values are *unique* ADFs
        (:class:`ADFCollection`) in this :class:`SOAPVectorCollection`.
        """        

    def __eq__(self, other):
        return all([a==b for a, b in
                    zip(self.vectors, other.vectors)])
    def equal(self, other):
        """Performs a more rigorous equality test for two collections by *also*
        comparing the radial and angular distribution functions.
        """
        return (self == other and
                self._rdfs == other._rdfs and
                self._adfs == other._adfs)
    
    def __iter__(self):
        return iter(self.vectors)
    def __contains__(self, value):
        return value in self.vectors
    def __len__(self):
        return len(self.vectors)
    def __getitem__(self, index):
        if isinstance(index, slice):
            result = SOAPVectorCollection(decomposer=self.decomposer)
            result.vectors = self.vectors[index]
            result.P = self.P[index,:]
            if len(self._rdfs) > 0:
                self._rdfs = {r: dfs[index] for r, dfs in self._rdfs.items()}
            if len(self._adfs) > 0:
                self._adfs = {r: dfs[index] for r, dfs in self._adfs.items()}
            return result
        else:
            return self.vectors[index]
    def __setitem__(self, index, value):
        if not isinstance(value, SOAPVector):
            raise TypeError("Only SOAPVectors can be placed in an SOAP "
                            "vector collection.")
        self.vectors[index] = value

    @staticmethod
    def from_file(filename):
        """Restores a SOAPVectorCollection from file.
        
        Args:
            filename (str): path to the file that was created by
              :meth:`SOAPVectorCollection.save`.
        """
        from six.moves.cPickle import load
        with open(filename, 'rb') as f:
            data = load(f)
        return SOAPVectorCollection.from_dict(data)

    @staticmethod
    def from_dict(data):
        """Restores a SOAPVectorCollection from a serialized dict (i.e., one
        returned by :meth:`serialize`).
        
        Args:
            data (dict): result of calling :meth:`serialize`.
        """
        vecs = []
        decomposer = SOAPDecomposer(**data["decomposer"])
        for vec in data["vectors"]:
            nvec = SOAPVector.from_dict(vec, decomposer, data["rx"], data["ax"])
            vecs.append(nvec)

        result = SOAPVectorCollection(data["P"], decomposer, calculate=False)
        result.vectors = vecs
        result._rdfs = {r: RDFCollection.from_dict(d)
                        for (r, d) in data["RDFs"].items()}
        result._adfs = {r: ADFCollection.from_dict(d)
                        for (r, d) in data["ADFs"].items()}
        return result
    
    def serialize(self, withdP=False):
        """Returns a serializable dictionary that represents this
        :class:`SOAPVectorCollection`.

        Args:
            withdP (bool): when True, the large, SOAP decomposition from which the
              DFs are constructed is also included. Wasteful, but necessary to
              completely represent the DFs.
        """
        if len(self) > 0:
            rx = self.vectors[0].rx
            ax = self.vectors[0].ax
        else:
            rx = None
            ax = None
            
        vecs = []
        for vec in self.vectors:
            vecs.append(vec.serialize(False, rx, withdP, ax))
            
        result = {
            "decomposer": self.decomposer.get_params(),
            "vectors": vecs,
            "P": self.P,
            "RDFs": {r: dfcol.serialize(withdP=withdP)
                     for r, dfcol in self._rdfs.items()},
            "ADFs": {r: dfcol.serialize(withdP=withdP)
                     for r, dfcol in self._adfs.items()},
            "rx": rx,
            "ax": ax
        }
        return result
    
    def save(self, target, withdP=False):
        """Saves the DF to disk.

        Args:
            target (str): path to save the vector to.
            withdP (bool): when True, the large, SOAP decomposition from which the
              DF was constructed is also included. Wasteful, but necessary to
              completely represent the DF.

        """
        from six.moves.cPickle import dump
        data = self.serialize(withdP=withdP)
        with open(target, 'wb') as f:
            dump(data, f)
        
    def RDFs(self, resolution=75, catom=False):
        """Returns the set of *unique* radial distribution functions for this
        collection.

        Args:
            resolution (int): number of points to sample in the radial domain.
            catom (bool): when True, the DFs are constructed using the density of
              the central atom.
        """
        if resolution not in self._rdfs:
            self._rdfs[resolution] = RDFCollection.from_soap(self, resolution, catom)
        return self._rdfs[resolution]  

    def ADFs(self, resolution=100, catom=False):
        """Returns the set of *unique* angular distribution functions for this
        collection.

        Args:
            resolution (int): number of points to sample in the angular domain.
            catom (bool): when True, the DFs are constructed using the density of
              the central atom.
        """
        if resolution not in self._adfs:
            self._adfs[resolution] = ADFCollection.from_soap(self, resolution, catom)
        return self._adfs[resolution]  
