"""Unit tests for cluster view."""

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
from spiky.views import ClusterView
from spiky.views.tests.utils import show_view, get_data


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_clusterview():
    keys = ('cluster_groups,group_colors,group_names,'
            'cluster_sizes').split(',')
    data = get_data()
    kwargs = {k: data[k] for k in keys}
    
    kwargs['cluster_colors'] = data['cluster_colors_full']
    
    def operator(self):
        self.view.select([2,4])
        
    kwargs['operator'] = operator
    
    # Show the view.
    show_view(ClusterView, **kwargs)
    
    
    