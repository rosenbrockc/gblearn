"""Once the unique LAEs for a grain boundary collection have been calculated
using SOAP and the similarity metric, we can generate the local environment
representation by accumulating the fraction of each unique LAE within a given
GB.
"""
def accumulate(gbc, U):
    """Accumulates the total number of occurances of each type of LAE in the
    whole GB system.
    
    Args:
        gbc (GrainBoundaryCollection): to accumulate LAE presence across.
        U (OrderedDict): collection of globally unique LAEs.
        
    Returns: 
        dict: with `(PID, EID)` keys and values a list of all the other `(PID, EID)`
        LAE indices in the entire system.
    """
    inverse = {u: [] for u in U}
    for gbid, gb in gbc.items():
        for vid, uid in enumerate(gb.LAEs):
            inverse[uid].append((gbid, vid))
            
    return inverse
