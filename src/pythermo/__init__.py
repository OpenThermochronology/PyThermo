"""
Classes and methods for performing various modeling and computational tasks in the field of low-temperature thermochronology.

This package includes the following modules:

    - .crystal      : Classes and methods for determining radiation damage levels, diffusivities, and thermochronometric dates from different mineral systems

    - .tT_path      : Methods for creating discretized time-temperature (tT) paths and performing crystal-independent radiation damage annealing.

    - .tT_model     : Methods for modeling and plotting thermochronometric date results for comparison to measured data sets. A key focus here is thermal history modeling.

    - .constants    : Important universal constants used by the various modules.

"""

__all__ = ['constants', 'crystal', 'tT_path', 'tT_model']

from .constants import *
from .crystal import *
from .tT_path import *
from .tT_model import *
