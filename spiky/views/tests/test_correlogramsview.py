"""Unit tests for correlograms view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd

from spiky.io.tests.mock_data import (setup, teardown, create_correlograms,
        nspikes, nclusters, nsamples, nchannels, fetdim, ncorrbins)
from spiky.io.loader import KlustersLoader
from spiky.io.selection import select
from spiky.io.tools import check_dtype, check_shape
from spiky.views import CorrelogramsView
from spiky.views.tests.utils import show_view, get_data


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_correlogramsview():
    keys = ('correlograms,clusters_selected,cluster_colors,'
            'ncorrbins').split(',')
           
    data = get_data()
    kwargs = {k: data[k] for k in keys}
    
    kwargs['correlograms'] = create_correlograms(nclusters, ncorrbins)
    
    # Show the view.
    show_view(CorrelogramsView, **kwargs)
    
    