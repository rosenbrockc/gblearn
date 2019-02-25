"""Class representing an LAE and methods for analyzing it
"""

class LAE(object):
    """Represents an LAE of a GrainBoundary

    Args:
        id (int): The number id corresponding to the LAE
        soap (numpy.ndarray): The soap vector that describes the LAE

    Attributes:
        id (int): The number id corresponding to the LAE
        soap (numpy.ndarray): The soap vector that describes the LAE
    """
    def __init__(self, id, soap):
        self.id = id
        self.soap = soap
