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
from spiky.io.selection import select, get_indices
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
    
def test_klusters_loader_control():
    # Open the mock data.
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    xmlfile = os.path.join(dir, 'test.xml')
    l = KlustersLoader(xmlfile)
    
    # Take all spikes in cluster 3.
    spikes = get_indices(l.get_clusters(clusters=3))
    
    # Put them in cluster 4.
    l.set_cluster(spikes, 4)
    spikes_new = get_indices(l.get_clusters(clusters=4))
    
    # Ensure all spikes in old cluster 3 are now in cluster 4.
    assert np.all(np.in1d(spikes, spikes_new))
    
    # Change cluster groups.
    clusters = [2, 3, 4]
    group = 0
    l.set_cluster_groups(clusters, group)
    groups = l.get_cluster_groups(clusters)
    assert np.all(groups == group)
    
    # Change cluster colors.
    clusters = [2, 3, 4]
    color = 12
    l.set_cluster_colors(clusters, color)
    colors = l.get_cluster_colors(clusters)
    assert np.all(colors == color)
    
    # Change group name.
    group = 0
    name = l.get_group_names(group)
    name_new = 'Noise new'
    assert name == 'Noise'
    l.set_group_names(group, name_new)
    assert l.get_group_names(group) == name_new
    
    # Change group color.
    groups = [1, 2]
    colors = l.get_group_colors(groups)
    color_new = 10
    l.set_group_colors(groups, color_new)
    assert np.all(l.get_group_colors(groups) == color_new)
    
    # Add cluster and group.
    spikes = get_indices(l.get_clusters(clusters=3))[:10]
    # Create new group 100.
    l.add_group(100, 'New group', 10)
    # Create new cluster 10000 and put it in group 100.
    l.add_cluster(10000, 100, 10)
    # Put some spikes in the new cluster.
    l.set_cluster(spikes, 10000)
    clusters = l.get_clusters(spikes=spikes)
    assert np.all(clusters == 10000)
    groups = l.get_cluster_groups(10000)
    assert groups == 100
    l.set_cluster(spikes, 2)
    
    # Remove the new cluster and group.
    l.remove_cluster(10000)
    l.remove_group(100)
    assert np.all(~np.in1d(10000, l.get_clusters()))
    assert np.all(~np.in1d(100, l.get_cluster_groups()))
    
    
    