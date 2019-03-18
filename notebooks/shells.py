#Quippy dependent
def shells(element, n=6, rcut=6.):
    """Returns the neighbor shells for the specified element.

    Args:
        element (str): name of the element.
        n (int): maximum number of shells to return.
        rcut (float): maximum cutoff to consider in looking for unique shells.
    """
    global _shells
    if element not in _shells:
        a = atoms(element)
        a.set_cutoff(rcut)
        a.calc_connect()
        result = []
        for i in a.indices:
            for neighb in a.connect[i]:
                dist = neighb.distance
                deltain = [abs(dist-s) < 1e-5 for s in result]
                if not any(deltain):
                    result.append(dist)

        _shells[element] = sorted(result)

    return _shells[element][0:min((n, len(_shells[element])))]

def test_shells(elements, models):
    """Tests the nearest neighbor shell distances for the elements.
    """
    for e in elements:
        result = shells(e)
        modelfile = models("{}.shells.npy".format(e))
        assert np.allclose(result, np.load(modelfile))
