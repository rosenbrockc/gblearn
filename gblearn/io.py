"""Functions for I/O interaction of :class:`gblearn.gb.GrainBoundaryCollection`
objects and the many results they produce with disk storage.

.. note:: Whenever a new representation is added, a property getter and setter
  need to be added *with documentation* to the :class:`ResultStore`. The three
  generic getter and setter functions for numpy arrays, aggregated arrays
  and dictionaries of parameterized aggregated arrays should be sufficient to
  handle any additional representations we may come up with.
"""
from os import path, mkdir
from tqdm import tqdm
import json
tqdm.monitor_interval = 0
import numpy as np
from gblearn import msg
from contextlib import contextmanager

class DiskCollection(object):
    """For large grain boundary collections, it may not be possible to keep all
    SOAP matrices and their derivatives in memory at the same time for
    processing. In these cases, it is preferable to read a matrix in, process
    it, and then delete it from memory right away.

    Args:
        root (str): path to the folder where the array files are stored.
        gbids (list): of ids for each GB; the files should be stored as `id.npy`
          in the `root` directory.
        restricted (bool): when True, run in memory-restricted mode. Otherwise,
          the loaded array files are cached in memory for speedy multiple
          retrieval.

    Attributes:
        cache (dict): keys are GB ids in :attr:`gbids`; values are the
          corresponding :class:`numpy.ndarray` files. This only has values in
          :attr:`restricted` is False.
        files (dict): keys are GB ids, values are paths to the numpy array files.
    """
    def __init__(self, root, gbids, restricted=True):
        from collections import OrderedDict
        from os import mkdir
        self.root = path.abspath(path.expanduser(root))
        if not path.isdir(self.root):
            mkdir(self.root)

        self.files = OrderedDict([(f, path.join(self.root, "{}.npy".format(f)))
                                  for f in gbids])
        self.gbids = gbids
        self.restricted = restricted
        self.cache = {}

    def __len__(self):
        return sum([1 if path.isfile(p) else 0
                    for p in self.files.values()])

    def __setitem__(self, gbid, value):
        gbpath = path.join(self.root, "{}.npy".format(gbid))
        np.save(gbpath, value)

        if not self.restricted:# pragma: no cover
            self.cache[gbid] = value

    @contextmanager
    def __getitem__(self, key):
        if not self.restricted and key in self.cache:# pragma: no cover
            yield self.cache[key]
        else:
            target = self.files[key]
            if not path.isfile(target):
                yield None
            else:
                result = np.load(target)
                if not self.restricted:# pragma: no cover
                    self.cache[key] = result
                yield result
                del result

class ResultStore(object):
    """Represents a collection of results for a
    :class:`~gblearn.gb.GrainBoundaryCollection` that can be cached because they
    are permanent for the particular collection.

    .. note:: The result store creates separate sub-folders `P_` for the full
      SOAP matrices of each GB; `U_` for the unique decompositions at different
      epsilon values; `ASR_` for the ASR representation; `LER_` for the LER
      representations (tied to particular epsilon values and unique
      decompositions); `Scatter_` for the Scatter vectors of each GB.

    .. note:: If any of the results (such as :attr:`P`, :attr:`U`, :attr:`ASR`,
      :attr:`LER`, or :attr:`Scatter`) are not saved in the store, `None` will be returned.

    .. note:: If `root=None` then the results will only be cached in memory and
      *not* to disk.

    Args:
        gbids (list): of identifiers for the individual members of the GB
          collection. The store makes sure that they match for a given directory
          so that results can be kept straight.
        root (str): path to the root directory for this result store.
        restricted (bool): when True, run in memory-restricted mode. Otherwise,
          the loaded array files are cached in memory for speedy multiple
          retrieval.

    Attributes:
        root (str): path to the root directory for this result store.
        gbids (list): of identifiers for the individual members of the GB
          collection.
        reps (list): of `str` representation names that will be handled by the
          result store.
        soapargs (dict): arguments for the SOAP descriptor. See
          :class:`gblearn.soap.SOAPCalculator` for information.
        soapstr (list): of `str` soap arguments that are formatted to form a
          folder name for specific variations of the representations.
    """
    repstr = {
        "soap": ["lmax", "nmax", "rcut"],
        "scatter": ["density", "Layers", "SPH_L", "n_trans", "n_angle1", "n_angle2"]
    }
    reps = {
        "soap": ["P", "U", "ASR", "LER", "features"],
        "scatter": ["Scatter", "heatmaps"]
    }

    def __init__(self, gbids, root=None, restricted=True, padding=10.0):
        self.root = root
        self.restricted = restricted
        self.gbids = gbids
        if root is not None:
            self.root = path.abspath(path.expanduser(root))
            self._make_repdirs()
            self._check_padding(padding)

        #Set the lazy initializers for each of the representations that we
        #handle.
        self.cache = {r: {} for r in self.reps}
        self.repargs = {r: {} for r in self.reps}

    def _check_padding(self, padding):
        """Makes sure that the padding matches the stored value for the grain boundary
        collection.

        Args:
            padding (float): amount of perfect bulk to include as padding around
              the grain boundary before the representation is made.
        """
        padfile = path.join(self.root, "padding.txt")
        if path.isfile(padfile):
            xpads = np.loadtxt(padfile)
            assert abs(xpads - padding) < 1e-8
        else:
            np.savetxt(padfile, [padding])

    def _make_repdirs(self):
        """Creates any missing directories if the store is not running in
        memory-only mode.
        """
        if not path.isdir(self.root):
            mkdir(self.root)

        self._load_gbids()

        for rep, dirnames in self.reps.items():
            repdir = path.join(self.root, rep)
            if not path.isdir(repdir):
                mkdir(repdir)
            for dirname in dirnames:
                folder = path.join(repdir, dirname)
                if not path.isdir(folder):
                    mkdir(folder)

    def configure(self, rep, multires=None, **repargs):
        """Configures the result store to use the specified representation.

        Args:
            rep (str): one of ["soap", "scatter"].
            repargs (dict): specific arguments for the representation `rep`.
        """
        assert rep in self.repargs
        key = ""

        if multires is not None:
            self.repargs[rep] = multires
            for args in multires:
                if any(a not in args for a in self.repstr[rep]):
                    eargs = ', '.join(["`{}`".format(a) for a in self.repstr[rep]])
                    emsg = "{} are required arguments for configuring a {} store.".format(eargs, rep)
                    raise ValueError(emsg)
                key += self._construct_rep_string(rep, **args) + "___"
                self._reset_attrs(key, rep)
            return


        self.repargs[rep] = repargs
        if any(a not in self.repargs[rep] for a in self.repstr[rep]):
            eargs = ', '.join(["`{}`".format(a) for a in self.repstr[rep]])
            emsg = "{} are required arguments for configuring a {} store.".format(eargs, rep)
            raise ValueError(emsg)
        key = self._construct_rep_string(rep, **repargs)
        self._reset_attrs(key, rep)

    def _construct_rep_string(self, rep, **repargs):

        pmap = {
            int: ":d",
            float: ":.2f"
        }
        ptypes = [type(repargs[n]) for n in self.repstr[rep]]
        pstrs = ["{%d%s}" % (i, pmap[ptype]) for i, ptype in enumerate(ptypes)]
        pvals = [repargs[n] for n in self.repstr[rep]]
        return '_'.join(pstrs).format(*pvals)

    def repattr(self, rep):
        """Returns the name of the attribute for the representation folder string.
        """
        return "{}str".format(rep)

    def _reset_attrs(self, key, rep):
        """Resets the strings representing the specified `rep`.
        """
        attrname = self.repattr(rep)
        if hasattr(self, attrname):
            delattr(self, attrname)
        setattr(self, attrname, key)
        self._clobber_reps()

    def _clobber_reps(self):
        """Resets all the cached representation values to `None`.
        """
        self.cache = {r: {} for r in self.reps}

    def _load_gbids(self):
        """Load any existing gbmap to make sure we are dealing with the same GB
        collection.
        """
        idfile = path.join(self.root, "gbids.json")
        if path.isfile(idfile):
            with open(idfile) as f:
                _gbids = json.load(f)
            #Make sure that the ids match.
            assert sorted(self.gbids) == sorted(_gbids)
        else:
            with open(idfile, 'w') as f:
                json.dump(list(self.gbids), f)

    @property
    def P(self):
        """SOAP Representation for each GB in the collection.

        .. note:: The SOAP matrices returned will be for the currently
          configured values of `lmax`, `nmax` and `rcut` in the store.

        Returns:
            dict: keys are `gbid`, values are the SOAP reprsentation for that
            particular GB.
        """
        return self._np_get("P")

    @P.setter
    def P(self, value):
        """Sets the value of the SOAP representation for each of the GBs in this
        collection.

        .. warning:: You *must* make sure that the SOAP parameters for the store
          are set correctly before calling this method. Otherwise, the matrices
          will be stored in the wrong location.

        Args:
            value (dict): keys are `gbid`, values are :class:`numpy.ndarray`
              SOAP matrices for the GBs.
        """
        self._np_set("P", value)

    @property
    def Scatter(self):
        """Scatter Representation for each GB in the collection.

        Returns:
            dict: keys are 'gbid', values are the Scatter representation for that
            particular GB
        """
        return self._np_get("Scatter")

    @Scatter.setter
    def Scatter(self, value):
        """Sets the value of the Scatter representation for each of the GBs in this
        collection.

        Args:
            value (dict): keys are `gbid`, values are :class:`numpy.ndarray`
              Scatter matrices for the GBs.
        """
        self._np_set("Scatter", value)

    def _find_rep(self, attr):
        """Finds which representation the unique `attr` name belongs to.
        """
        result = None
        for rep, attrs in self.reps.items():
            if attr in attrs:
                result = rep
                break
        else: #pragma: no cover
            raise ValueError("Cannot find the representation that {} belongs to.".format(attr))

        return result

    def _np_get(self, attr):
        """Restores a :class:`numpy.ndarray` based representation collection for
        each GB in the result store.

        Args:
            attr (str): name of the representation being retrieved.

        Returns:
            dict: keys are gbid; values are the numpy arrays.
        """
        rep = self._find_rep(attr)
        cache = self.cache[rep].get(attr)
        if cache is not None:
            return cache

        #Handle the memory-only case.
        if self.root is None:
            return {}

        rpath = path.join(self.root, rep, attr)
        if not hasattr(self, self.repattr(rep)):
            return {}
        target = path.join(rpath, getattr(self, self.repattr(rep)))
        result = DiskCollection(target, self.gbids, self.restricted)
        self.cache[rep][attr] = result
        return result

    def _np_set(self, attr, value):
        """Sets a :class:`numpy.ndarray` value for each GB in the collection.

        .. note:: If a root directory is set for this result store, then these
          values will be saved to disk.

        Args:
            attr (str): name of the representation to set values for.
            value (dict): keys are gbid, values are the numpy arrays to set.
        """
        if len(value) != len(self.gbids):
            msg.err("The number of GBs in the specified set does not match "
                    "the result store configuration.")
            return

        #Handle the memory-only case.
        rep = self._find_rep(attr)
        if self.root is None:
            self.cache[rep][attr] = value
            return

        rpath = path.join(self.root, rep, attr)
        target = path.join(rpath, getattr(self, self.repattr(rep)))
        if not path.isdir(target):# pragma: no cover
            mkdir(target)

        dc = DiskCollection(target, self.gbids, self.restricted)
        self.cache[rep][attr] = dc

        saved = []
        for gbid, A in tqdm(value.items()):
            dc[gbid] = A
            saved.append(gbid)

        assert len(np.setdiff1d(self.gbids, saved)) == 0

    @property
    def features(self):
        """Gets the list of unique ids that describe each set of unique
        enviroments as a function of `eps`.

        .. warning:: These value are dependent on the particular SOAP parameter
          #set. They are saved accordingly.

        Returns:
            dict: keys are values of `epsilon`, rounded to 5 decimal places;
            values are a list of `(gbid, eid)` tuples.
        """
        return self._agg_get("features")

    @features.setter
    def features(self, value):
        """Sets the value of the feature descriptors in this store. It is
        stored according to the current SOAP parameter set.

        .. warning:: If a file already exists for a particular unique
          decomposition at a value of `eps`, it will *not* be overwritten, but
          skipped.

        Args:
            value (dict): keys are epsilon value controlling when two LAEs are
              similar; values are lists of `(gbid, eid)` tuples.
        """
        self._agg_set("features", value)

    @property
    def U(self):
        """Gets the value of the unique decomposition in this store.

        .. warning:: These value are dependent on the particular SOAP parameter
          set. They are saved accordingly.

        Returns:
            dict: keys are values of `epsilon`, rounded to 5 decimal places;
            values are those returned by :func:`~gblearn.reduced.unique`.
        """
        return self._agg_get("U")

    @U.setter
    def U(self, value):
        """Sets the value of the unique decomposition in this store. It is
        stored according to the current SOAP parameter set.

        .. warning:: If a file already exists for a particular unique
          decomposition at a value of `eps`, it will *not* be overwritten, but
          skipped.

        Args:
            value (dict): keys are epsilon value controlling when two LAEs are
              similar; values are the same as returned by
              :meth:`~gblearn.gb.GrainBoundaryCollection.uniquify`.
        """
        self._agg_set("U", value)

    def _agg_get(self, attr):
        """Returns a dictionary of aggregated representations for th given name.

        Args:
            attr (str): name of the representation to get values for.
        """
        rep = self._find_rep(attr)
        cache = self.cache[rep].get(attr)
        if cache is not None:
            return cache

        #Handle the memory-only case.
        if self.root is None:
            return {}

        from glob import glob
        from gblearn.utility import chdir
        try:
            from cPickle import load
        except ImportError:
            from pickle import load

        result = {}
        rpath = path.join(self.root, rep, attr)
        target = path.join(rpath, getattr(self, self.repattr(rep)))
        if not path.isdir(target):
            return result

        with chdir(target):
            for pkl in glob("*.pkl"):
                seps = pkl[:-4]
                eps = float(seps)
                with open(pkl, 'rb') as f:
                    result[eps] = load(f)

        self.cache[rep][attr] = result
        return result

    def _agg_set(self, attr, value):
        """Sets the value of a parameter-sensitive collection of *aggregated*
        representations.

        Args:
            attr (str): name of the representation to set values for.
            value (dict): keys are float parameter values; values are
              :class:`numpy.ndarray` aggregated representation.
        """
        rep = self._find_rep(attr)
        cache = self.cache[rep].get(attr)
        if cache is None:
            self.cache[rep][attr] = {}
        self.cache[rep][attr].update(value)

        #Handle the memory-only case.
        if self.root is None:
            return

        rpath = path.join(self.root, rep, attr)
        target = path.join(rpath, getattr(self, self.repattr(rep)))
        if not path.isdir(target):
            mkdir(target)
        
        try:
            from cPickle import dump
        except ImportError:
            from pickle import dump
        for eps, utup in value.items():
            upath = path.join(target, "{0:.5f}.pkl".format(eps))
            if not path.isfile(upath):
                with open(upath, 'wb') as f:
                    dump(utup, f)

    @property
    def ASR(self):
        """Returns the Averaged SOAP representation for this GB collection using
        the currently configured SOAP parameters (see :attr:`SOAP`).
        """
        return self._single_get("ASR")

    @ASR.setter
    def ASR(self, value):
        """Stores the Averaged SOAP representation for this GB collection using
        the currently configured SOAP parameters (see :attr:`SOAP`).

        .. note:: If the file already exists for the ASR at the current SOAP
          parameters, then it will *not* be re-saved.

        Args:
            value (numpy.ndarray): ASR for the GB collection.
        """
        self._single_set("ASR", value)

    def _single_get(self, attr):
        """Returns a representation for the GB system that is based in a single
        :class:`numpy.ndarray` *without* parameter dependence (except for SOAP).
        """
        rep = self._find_rep(attr)
        cache = self.cache[rep].get(attr)
        if cache is not None:
            return cache

        #Handle the memory-only case.
        if self.root is None:
            return

        result = None
        rpath = path.join(self.root, rep, attr)
        target = path.join(rpath, "{}.npy".format(getattr(self, self.repattr(rep))))
        if path.isfile(target):
            result = np.load(target)

        self.cache[rep][attr] = result
        return result

    def _single_set(self, attr, value):
        rep = self._find_rep(attr)
        self.cache[rep][attr] = value
        if self.root is None:
            return

        rpath = path.join(self.root, rep, attr)
        target = path.join(rpath, "{}.npy".format(getattr(self, self.repattr(rep))))
        if not path.isfile(target):
            np.save(target, value)

    @property
    def LER(self):
        """Gets the LER for the current SOAP configuration.

        Returns:
            dict: keys are values of `epsilon`, rounded to 5 decimal places;
            values are the Local Environment Representation for the GB
            collection. The keys in this `dict` are linked with those in
            :attr:`U`.
        """
        return self._agg_get("LER")

    @LER.setter
    def LER(self, value):
        """Sets the LER in the store for the current SOAP configuration.

        Args:
            value (dict): keys are values of `epsilon`, rounded to 5 decimal places;
              values are the Local Environment Representation for the GB
              collection. The keys in this `dict` are linked with those in
              :attr:`U`.
        """
        self._agg_set("LER", value)
