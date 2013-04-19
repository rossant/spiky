"""Unit tests for correlation matrix view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd

from spiky.io.tests.mock_data import (setup, teardown, create_correlation_matrix,
        nspikes, nclusters, nsamples, nchannels, fetdim, ncorrbins)
from spiky.io.loader import KlustersLoader
from spiky.io.selection import select
from spiky.io.tools import check_dtype, check_shape
from spiky.utils.userpref import USERPREF
from spiky.views import CorrelationMatrixView
from spiky.views.tests.utils import show_view, get_data


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_correlationmatrixview():
    data = get_data()
    
    kwargs = {}
    kwargs['correlation_matrix'] = create_correlation_matrix(nclusters)
    kwargs['cluster_colors_full'] = data['cluster_colors_full']
    
    kwargs['operators'] = [
        lambda self: (self.close() 
            if USERPREF['test_auto_close'] != False else None),
    ]
    
    # Show the view.
    show_view(CorrelationMatrixView, **kwargs)
    
    