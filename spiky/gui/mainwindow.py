"""Main window."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pprint
import inspect
from collections import OrderedDict

import pandas as pd
import numpy as np
import numpy.random as rnd
from galry import QtGui, QtCore

from spiky.io.selection import get_indices
from spiky.utils.colors import COLORMAP
import spiky.utils.logger as log
from spiky.utils.settings import SETTINGS

class MainWindow(QtGui.QMainWindow):
    pass
    