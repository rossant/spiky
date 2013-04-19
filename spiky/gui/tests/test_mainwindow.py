"""Unit tests for the main window."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import time

import numpy as np
import numpy.random as rnd
import pandas as pd
from galry import show_window

from spiky.gui.mainwindow import MainWindow
# from spiky.io.tests.mock_data import (setup, teardown, create_correlation_matrix,
        # nspikes, nclusters, nsamples, nchannels, fetdim, ncorrbins)
from spiky.io.loader import KlustersLoader
from spiky.io.selection import select
from spiky.io.tools import check_dtype, check_shape
from spiky.gui.mainwindow import MainWindow


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_mainwindow():

    show_window(MainWindow)
    
if __name__ == '__main__':
    test_mainwindow()
    
    