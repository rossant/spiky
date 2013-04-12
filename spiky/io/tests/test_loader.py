"""Unit tests for loader module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd
from nose import with_setup
import shutil

from spiky.io.tests.mock_data import (create_waveforms, create_features, 
    create_clusters, create_masks, create_xml, create_probe,
    create_cluster_colors)
from spiky.io.loader import KlustersLoader
from spiky.io.selection import select
from spiky.io.tools import save_binary, save_text, check_dtype, check_shape


# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
# Mock parameters.
nspikes = 1000
nclusters = 15
nsamples = 20
nchannels = 32
fetdim = 3


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
def setup():
    
    # Create mock data.
    waveforms = create_waveforms(nspikes, nsamples, nchannels)
    features = create_features(nspikes, nchannels, fetdim)
    clusters = create_clusters(nspikes, nclusters)
    cluster_colors = create_cluster_colors(nclusters - 1)
    masks = create_masks(nspikes, nchannels, fetdim)
    xml = create_xml(nchannels, nsamples, fetdim)
    probe = create_probe(nchannels)
    
    # Create mock directory if needed.
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    # Create mock files.
    save_binary(os.path.join(dir, 'test.spk.1'), waveforms)
    save_text(os.path.join(dir, 'test.fet.1'), features,
        header=nchannels * fetdim + 1)
    save_text(os.path.join(dir, 'test.clu.1'), clusters, header=nclusters)
    save_text(os.path.join(dir, 'test.clucol.1'), cluster_colors)
    save_text(os.path.join(dir, 'test.mask.1'), masks, header=nclusters)
    save_text(os.path.join(dir, 'test.xml'), xml)
    save_text(os.path.join(dir, 'test.probe'), probe)
    
def teardown():
    # Erase the temporary data directory.
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    if os.path.exists(dir):
        shutil.rmtree(dir)
    

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
@with_setup(setup, teardown)
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
    cluster_colors = l.get_cluster_colors()
    spiketimes = l.get_spiketimes()
    probe = l.get_probe()
    
    maxcluster = clusters.max()
    
    # Check the shape of the data sets.
    assert check_shape(features, (nspikes, nchannels * fetdim + 1))
    assert check_shape(masks, (nspikes, nchannels))
    assert check_shape(waveforms, (nspikes, nsamples, nchannels))
    assert check_shape(clusters, (nspikes,))
    assert check_shape(cluster_colors, (maxcluster + 1,))
    assert check_shape(spiketimes, (nspikes,))
    assert check_shape(probe, (nchannels, 2))
    
    # Check the data type of the data sets.
    assert check_dtype(features, np.float32)
    assert check_dtype(masks, np.float32)
    # HACK: Panel has no dtype(s) attribute
    # assert check_dtype(waveforms, np.float32)
    assert check_dtype(clusters, np.int32)
    assert check_dtype(cluster_colors, np.int32)
    assert check_dtype(spiketimes, np.float32)
    assert check_dtype(probe, np.float32)
    
    # Check selection.
    index = nspikes / 2
    waveform = select(waveforms, index).values
    cluster = clusters[index]
    spikes_in_cluster = np.nonzero(clusters == cluster)[0]
    nspikes_in_cluster = len(spikes_in_cluster)
    l.select(clusters=[cluster])
    
    # Check the size of the selected data.
    assert check_shape(l.get_features(), (nspikes_in_cluster, 
                                          nchannels * fetdim + 1))
    assert check_shape(l.get_masks(full=True), (nspikes_in_cluster, 
                                                nchannels * fetdim + 1))
    assert check_shape(l.get_waveforms(), 
                       (nspikes_in_cluster, nsamples, nchannels))
    assert check_shape(l.get_clusters(), (nspikes_in_cluster,))
    assert check_shape(l.get_spiketimes(), (nspikes_in_cluster,))
    
    # Check waveform sub selection.
    waveforms_selected = l.get_waveforms()
    assert np.array_equal(select(waveforms_selected, index).values, waveform)
    
    # Close the loader.
    l.close()
