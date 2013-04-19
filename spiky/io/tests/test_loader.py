"""Unit tests for loader module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
from collections import Counter

import numpy as np
import numpy.random as rnd
import pandas as pd
import shutil

from spiky.io.tests.mock_data import (setup, teardown,
                            nspikes, nclusters, nsamples, nchannels, fetdim)
from spiky.io.loader import KlustersLoader
from spiky.io.selection import select
from spiky.io.tools import check_dtype, check_shape, get_array


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_klusters_loader():
    # Open the mock data.
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    xmlfile = os.path.join(dir, 'test.xml')
    l = KlustersLoader(xmlfile)
    
    # Get full data sets.
    features = l.get_features()
    masks = l.get_masks()
    waveforms = l.get_waveforms()
    clusters = l.get_clusters()
    spiketimes = l.get_spiketimes()
    nclusters = len(Counter(clusters))
    
    probe = l.get_probe()
    cluster_colors = l.get_cluster_colors()
    cluster_groups = l.get_cluster_groups()
    group_colors = l.get_group_colors()
    group_names = l.get_group_names()
    cluster_sizes = l.get_cluster_sizes()
    
    maxcluster = clusters.max()
    
    # Check the shape of the data sets.
    # ---------------------------------
    assert check_shape(features, (nspikes, nchannels * fetdim + 1))
    assert check_shape(masks, (nspikes, nchannels))
    assert check_shape(waveforms, (nspikes, nsamples, nchannels))
    assert check_shape(clusters, (nspikes,))
    assert check_shape(spiketimes, (nspikes,))
    
    assert check_shape(probe, (nchannels, 2))
    assert check_shape(cluster_colors, (nclusters,))
    assert check_shape(cluster_groups, (nclusters,))
    assert check_shape(group_colors, (3,))
    assert check_shape(group_names, (3,))
    assert check_shape(cluster_sizes, (nclusters,))
    
    
    # Check the data type of the data sets.
    # -------------------------------------
    assert check_dtype(features, np.float32)
    assert check_dtype(masks, np.float32)
    # HACK: Panel has no dtype(s) attribute
    # assert check_dtype(waveforms, np.float32)
    assert check_dtype(clusters, np.int32)
    assert check_dtype(spiketimes, np.float32)
    
    assert check_dtype(probe, np.float32)
    assert check_dtype(cluster_colors, np.int32)
    assert check_dtype(cluster_groups, np.int32)
    assert check_dtype(group_colors, np.int32)
    assert check_dtype(group_names, object)
    assert check_dtype(cluster_sizes, np.int32)
    
    
    # Check selection.
    # ----------------
    index = nspikes / 2
    waveform = select(waveforms, index)
    cluster = clusters[index]
    spikes_in_cluster = np.nonzero(clusters == cluster)[0]
    nspikes_in_cluster = len(spikes_in_cluster)
    l.select(clusters=[cluster])
    
    
    # Check the size of the selected data.
    # ------------------------------------
    assert check_shape(l.get_features(), (nspikes_in_cluster, 
                                          nchannels * fetdim + 1))
    assert check_shape(l.get_masks(full=True), (nspikes_in_cluster, 
                                                nchannels * fetdim + 1))
    assert check_shape(l.get_waveforms(), 
                       (nspikes_in_cluster, nsamples, nchannels))
    assert check_shape(l.get_clusters(), (nspikes_in_cluster,))
    assert check_shape(l.get_spiketimes(), (nspikes_in_cluster,))
    
    
    # Check waveform sub selection.
    # -----------------------------
    waveforms_selected = l.get_waveforms()
    assert np.array_equal(get_array(select(waveforms_selected, index)), 
        get_array(waveform))
    
    
    # Close the loader.
    # -----------------
    l.close()

    