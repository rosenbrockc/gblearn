import pytest
from gblearn.base import set_testmode
set_testmode(True)
from gblearn.base import testmode

import matplotlib
from gblearn.base import testmode
matplotlib.use("Agg" if testmode else "TkAgg")
