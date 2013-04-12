"""Unit tests for loader module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd
from nose import with_setup

from spiky.io.tests.mock_data import (setup, teardown,
                            nspikes, nclusters, nsamples, nchannels, fetdim)
from spiky.io.loader import KlustersLoader
from spiky.io.selection import select
from spiky.io.tools import check_dtype, check_shape
from spiky.views import WaveformView
from spiky.views.tests import show_view


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
@with_setup(setup, teardown)
def test_waveformview():
    
    # Mock data folder.
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '../../io/tests/mockdata')
    
    # Load data files.
    xmlfile = os.path.join(dir, 'test.xml')
    l = KlustersLoader(xmlfile)
    
    # Get full data sets.
    clusters_selected = [1, 3, 10]
    l.select(clusters=clusters_selected)
    
    features = l.get_features()
    masks = l.get_masks()
    waveforms = l.get_waveforms()
    clusters = l.get_clusters()
    cluster_colors = l.get_cluster_colors()
    spiketimes = l.get_spiketimes()
    probe = l.get_probe()
    
    # Show the view.
    show_view(WaveformView, waveforms=waveforms,
                              clusters=clusters,
                              cluster_colors=cluster_colors,
                              clusters_selected=clusters_selected,
                              masks=masks,
                              geometrical_positions=probe,)
    
    