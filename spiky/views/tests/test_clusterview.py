"""Unit tests for cluster view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import time

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
    
    kwargs['operators'] = [
        lambda self: self.view.select([2,4]),
        lambda self: self.view.add_group("MyGroup", [1,2,6]),
        lambda self: self.view.rename_group(3, "New group"),
        lambda self: self.view.change_group_color(3, 2),
        lambda self: self.view.change_cluster_color(1, 4),
        lambda self: self.view.move_to_noise(0),
        # lambda self: self.close(),
    ]
    
    # Show the view.
    show_view(ClusterView, **kwargs)
    
    
    