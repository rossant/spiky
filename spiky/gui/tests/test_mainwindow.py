"""Unit tests for the main window."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import time

import numpy as np
import numpy.random as rnd
import pandas as pd

from spiky.gui.mainwindow import MainWindow
from spiky.io.tests.mock_data import (setup, teardown, create_correlation_matrix,
        nspikes, nclusters, nsamples, nchannels, fetdim, ncorrbins)
from spiky.io.loader import KlustersLoader
from spiky.io.selection import select
from spiky.io.tools import check_dtype, check_shape
from spiky.views import ClusterView
from spiky.views.tests.utils import show_view, get_data


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_mainwindow():
    keys = ('cluster_groups,group_colors,group_names,'
            'cluster_sizes').split(',')
    data = get_data()
    kwargs = {k: data[k] for k in keys}
    
    
    
    