"""Unit tests for controller module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np

from spiky.control.controller import Controller
from spiky.io.tests.mock_data import (setup, teardown,
                            nspikes, nclusters, nsamples, nchannels, fetdim)
from spiky.io.loader import KlustersLoader
from spiky.io.selection import select, get_indices
from spiky.io.tools import check_dtype, check_shape, get_array


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def load():
    # Open the mock data.
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '../../io/tests/mockdata')
    xmlfile = os.path.join(dir, 'test.xml')
    l = KlustersLoader(xmlfile)
    c = Controller(l)
    return (l, c)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_controller_1():
    l, c = load()
    
    # Select three clusters
    clusters = [2, 4, 6]
    spikes = l.get_spikes(clusters=clusters)
    cluster_spikes = l.get_clusters(clusters=clusters)
    # Select half of the spikes in these clusters.
    spikes_sample = spikes[::2]
    
    
    # Merge these clusters.
    cluster_new = c.merge_clusters(clusters)
    assert np.array_equal(l.get_spikes(cluster_new), spikes)
    assert np.all(~np.in1d(clusters, get_indices(l.get_cluster_groups('all'))))
    
    # Undo.
    assert c.can_undo()
    c.undo()
    assert np.array_equal(l.get_spikes(cluster_new), [])
    assert np.all(np.in1d(clusters, get_indices(l.get_cluster_groups('all'))))
    assert np.array_equal(l.get_clusters(clusters=clusters), cluster_spikes)
    
    # Redo.
    assert c.can_redo()
    c.redo()
    assert np.array_equal(l.get_spikes(cluster_new), spikes)
    assert np.all(~np.in1d(clusters, get_indices(l.get_cluster_groups('all'))))
    
    
    # Split the newly created cluster into two clusters.
    cluster_split = c.split_clusters(cluster_new, spikes_sample)[0]
    assert cluster_split == cluster_new + 1
    assert np.array_equal(l.get_spikes(cluster_split), spikes_sample)
    
    # Undo.
    c.undo()
    assert np.array_equal(l.get_spikes(cluster_new), spikes)
    
    # Redo.
    c.redo()
    assert np.array_equal(l.get_spikes(cluster_split), spikes_sample)
    
def test_controller_recolor_clusters():
    l, c = load()
    group = 1
    cluster = 3
    
    # Change cluster color.
    color_old = l.get_cluster_colors(cluster)
    c.change_cluster_color(cluster, 12)
    assert l.get_cluster_colors(cluster) == 12
    
    # Undo.
    c.undo()
    assert l.get_cluster_colors(cluster) == color_old
    
    # Redo.
    c.redo()
    assert l.get_cluster_colors(cluster) == 12
    
def test_controller_move_clusters():
    l, c = load()
    group = 1
    clusters = [3, 5, 7]
    
    c.move_clusters(clusters, group)
    assert np.all(l.get_cluster_groups(clusters) == 1)
    
    # Undo.
    c.undo()
    assert np.all(l.get_cluster_groups(clusters) == 2)
    
    # Redo.
    c.redo()
    assert np.all(l.get_cluster_groups(clusters) == 1)
    
def test_controller_rename_groups():
    l, c = load()
    group = 1
    
    # Rename groups.
    name = 'My group'
    c.rename_group(group, name)
    
    # Undo.
    c.undo()
    assert l.get_group_names(group) == 'MUA'
    
    # Redo.
    c.redo()
    assert l.get_group_names(group) == name
    
def test_controller_recolor_groups():
    l, c = load()
    group = 1
    # Change group color.
    color = l.get_group_colors(group)
    c.change_group_color(group, 10)
    assert l.get_group_colors(group) == 10
    
    # Undo.
    c.undo()
    assert l.get_group_colors(group) == color
    
    # Redo.
    c.redo()
    assert l.get_group_colors(group) == 10
    
def test_controller_add_group():
    l, c = load()
    
    # Add a group.
    group = 3
    c.add_group(group, 'My group', 2)
    assert np.all(~np.in1d(l.get_cluster_groups(), group))
    assert l.get_group_names(group) == 'My group'
    assert l.get_group_colors(group) == 2
    
    # Undo.
    c.undo()
    assert np.all(~np.in1d(l.get_cluster_groups(), group))
    
    # Redo.
    c.redo()
    assert np.all(~np.in1d(l.get_cluster_groups(), group))
    assert l.get_group_names(group) == 'My group'
    assert l.get_group_colors(group) == 2
    
def test_controller_remove_group():
    l, c = load()
    
    # Remove a group.
    group = 1
    c.remove_group(group)
    assert np.all(~np.in1d(l.get_cluster_groups(), group))
    
    # Undo.
    c.undo()
    assert np.all(~np.in1d(l.get_cluster_groups(), group))
    assert l.get_group_names(group) == 'MUA'
    
    # Redo.
    c.redo()
    assert np.all(~np.in1d(l.get_cluster_groups(), group))
    
    
    