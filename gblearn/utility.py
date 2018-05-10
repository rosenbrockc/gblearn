import numpy as np
from contextlib import contextmanager

@contextmanager
def chdir(target):
    """Provides a context for performing operations in a new
    directory. When the operation completes, the working directory is
    restored.
    """
    from os import getcwd, chdir
    current = getcwd()
    try:
        chdir(target)
        yield target
    finally:
        chdir(current)

def colorspace(size):# pragma: no cover
    """Returns an cycler over a linear color space with 'size' entries.

    Args:
        size (int): the number of colors to define in the space.

    Returns:
        itertools.cycle: iterable cycler with 'size' colors.
    """
    from matplotlib import cm
    from itertools import cycle
    import numpy as np
    rbcolors = cm.rainbow(np.linspace(0, 1, size))
    return cycle(rbcolors)

def _get_reporoot():
    """Returns the absolute path to the repo root directory on the current
    system.
    """
    from os import path
    import gblearn
    gblpath = path.abspath(gblearn.__file__)
    return path.dirname(path.dirname(gblpath))

reporoot = _get_reporoot()
"""The absolute path to the repo root on the local machine.
"""
