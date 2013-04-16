"""Unit tests for feature view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd

from spiky.io.tests.mock_data import (setup, teardown,
                            nspikes, nclusters, nsamples, nchannels, fetdim)
from spiky.io.loader import KlustersLoader
from spiky.io.selection import select
from spiky.io.tools import check_dtype, check_shape
from spiky.views import FeatureView
from spiky.views.tests.utils import show_view, get_data


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_featureview():
        
    keys = ('features,masks,clusters,clusters_selected,cluster_colors,'
            'fetdim,nchannels,nextrafet').split(',')
           
    data = get_data()
    kwargs = {k: data[k] for k in keys}
    
    # Show the view.
    show_view(FeatureView, **kwargs)
    
    