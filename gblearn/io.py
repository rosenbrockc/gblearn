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
      epsilon values; `ASR_` for the ASR representation; and `LER_` for the LER
      representations (tied to particular epsilon values and unique
      decompositions).

    .. note:: If any of the results (such as :attr:`P`, :attr:`U`, :attr:`ASR`
      or :attr:`LER`) are not saved in the store, `None` will be returned.

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
        soapargs (dict): arguments for the SOAP descriptor. See
          :class:`gblearn.soap.SOAPCalculator` for information.

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
    soapstr = ["lmax", "nmax", "rcut"]
    reps = ["P", "U", "ASR", "LER"]
    
    def __init__(self, gbids, root=None, restricted=True, **soapargs):
        self.root = root
        self.restricted = restricted
        if root is not None:
            self.root = path.abspath(path.expanduser(root))
            for r in self.reps:
                setattr(self, r + '_', path.join(self.root, r))

        if any(a not in soapargs for a in self.soapstr):
            eargs = ', '.join(["`{}`".format(a) for a in self.soapstr])
            emsg = "{} are required arguments for ResultStore.".format(eargs)
            raise ValueError(emsg)
            
        self.gbids = gbids
        self.soapargs = {}
        self.soapargs.update(soapargs)

        #Set the lazy initializers for each of the representations that we
        #handle.
        self._clobber_reps()
        #Create the directories and validate the grain boundary ids that were
        #passed in against any existing ones.
        if self.root is not None:
            self._setup_dirs()

    def _clobber_reps(self):
        """Resets all the cached representation values to `None`.
        """
        for r in self.reps:
            setattr(self, '_' + r, None)
        
    def _setup_dirs(self):
        """Creates any missing directories if the store is not running in
        memory-only mode.
        """
        if not path.isdir(self.root):
            mkdir(self.root)

        #Load any existing gbmap to make sure we are dealing with the same GB
        #collection.
        import json
        idfile = path.join(self.root, "gbids.json")        
        if path.isfile(idfile):
            with open(idfile) as f:
                _gbids = json.load(f)
            #Make sure that the ids match.
            assert sorted(self.gbids) == sorted(_gbids)
        else:
            with open(idfile, 'w') as f:
                json.dump(self.gbids, f)
            
        dirs = ["{}_".format(r) for r in self.reps]
        for sdir in dirs:
            if not path.isdir(getattr(self, sdir)):
                mkdir(getattr(self, sdir))
        
    @property
    def SOAP(self):
        """Returns the current SOAP paramater configuration.

        Returns:
            tuple: of SOAP arguments used to form the directory name for
            particular parameter sets. See :attr:`soapstr`.
        """
        return tuple([self.soapargs[a] for a in self.soapstr])

    @SOAP.setter
    def SOAP(self, value):
        """Sets the current SOAP parameter configuration.
        
        Args:
            value (dict): of key-value SOAP parameter pairs.
        """
        self.soapargs.update(value)
        
        #Clobber all of the cached values since we undid the SOAP parameters.
        self._clobber_reps()
        
    @property
    def SOAP_str(self):
        """Returns the directory string for the current SOAP configuration. It
        is defined by `nmax_lmax_rcut`.
        """
        if any(p is None for p in self.SOAP):
            raise ValueError("You can't use the result store until SOAP "
                             "parameters have been set. Pass them to the "
                             "constructor, or use property SOAP.")            
        return "{0:d}_{1:d}_{2:.2f}".format(*self.SOAP)

    @property
    def P(self):
        """SOAP Representation for each GB in the collection.

        .. note:: The SOAP matrices returned will be for the currently
          configured values of `lmax`, `nmax` and `rcut` in the store.

        Returns:
            dict: keys are `gbid`, values are the SOAP reprentation for that
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
    
    def _np_get(self, attr):
        """Restores a :class:`numpy.ndarray` based representation collection for
        each GB in the result store.

        Args:
            attr (str): name of the representation being retrieved.

        Returns:
            dict: keys are gbid; values are the numpy arrays.
        """
        cache = getattr(self, '_' + attr)
        if cache is not None:
            return cache

        #Handle the memory-only case.
        if self.root is None:
            return {}

        rpath = getattr(self, attr + '_')
        target = path.join(rpath, self.SOAP_str)
        result = DiskCollection(target, self.gbids, self.restricted)
        setattr(self, '_' + attr, result)
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
        if self.root is None:
            setattr(self, '_' + attr, value)
            return

        rpath = getattr(self, attr + '_')
        target = path.join(rpath, self.SOAP_str)
        if not path.isdir(target):
            mkdir(target)

        dc = DiskCollection(target, self.gbids, self.restricted)
        setattr(self, '_' + attr, dc)
            
        saved = []
        for gbid, A in tqdm(value.items()):
            dc[gbid] = A
            saved.append(gbid)

        assert len(np.setdiff1d(self.gbids, saved)) == 0

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
        cache = getattr(self, '_' + attr)
        if cache is not None:
            return cache

        #Handle the memory-only case.
        if self.root is None:
            return {}
        
        from glob import glob
        from gblearn.utility import chdir
        from cPickle import load
        
        result = {}
        rpath = getattr(self, attr + '_')
        target = path.join(rpath, self.SOAP_str)
        if not path.isdir(target):
            return result
        
        with chdir(target):
            for pkl in glob("*.pkl"):
                seps = pkl[:-4]
                eps = float(seps)
                with open(pkl, 'rb') as f:
                    result[eps] = load(f)

        setattr(self, '_' + attr, result)
        return result

    def _agg_set(self, attr, value):
        """Sets the value of a parameter-sensitive collection of *aggregated*
        representations.

        Args:
            attr (str): name of the representation to set values for.
            value (dict): keys are float parameter values; values are
              :class:`numpy.ndarray` aggregated representation.
        """
        if getattr(self, '_' + attr) is None:
            setattr(self, '_' + attr, {})
        getattr(self, '_' + attr).update(value)
        
        #Handle the memory-only case.
        if self.root is None:
            return

        rpath = getattr(self, attr + '_')
        target = path.join(rpath, self.SOAP_str)
        if not path.isdir(target):
            mkdir(target)
            
        from cPickle import dump
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
        cache = getattr(self, '_' + attr)
        if cache is not None:
            return cache

        #Handle the memory-only case.
        if self.root is None:
            return

        rpath = getattr(self, attr + '_')
        target = path.join(rpath, "{}.npy".format(self.SOAP_str))
        if path.isfile(target):
            return np.load(target)

    def _single_set(self, attr, value):
        setattr(self, '_' + attr, value)
        
        #Handle the memory-only case.
        if self.root is None:
            return
        
        rpath = getattr(self, attr + '_')
        target = path.join(rpath, "{}.npy".format(self.SOAP_str))
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
