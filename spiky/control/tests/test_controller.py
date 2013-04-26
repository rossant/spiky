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
def test_controller_merge():
    l, c = load()
    
    # Select three clusters
    clusters = [2, 4, 6]
    spikes = l.get_spikes(clusters=clusters)
    cluster_spikes = l.get_clusters(clusters=clusters)
    # Select half of the spikes in these clusters.
    spikes_sample = spikes[::2]
    
    
    # Merge these clusters.
    action, output = c.merge_clusters(clusters)
    cluster_new = output['to_select']
    
    assert action == 'merge_clusters'
    assert np.array_equal(output['to_select'], cluster_new)
    assert np.array_equal(output['to_invalidate'], 
        sorted(set(clusters).union(set([cluster_new]))))
    assert np.array_equal(output['to_compute'], cluster_new)
        
    assert np.array_equal(l.get_spikes(cluster_new), spikes)
    assert np.all(~np.in1d(clusters, get_indices(l.get_cluster_groups('all'))))
    
    # Undo.
    assert c.can_undo()
    action, output = c.undo()
    
    assert action == 'merge_clusters_undo'
    assert np.array_equal(output['to_select'], clusters)
    assert np.array_equal(output['to_invalidate'], 
        sorted(set(clusters).union(set([cluster_new]))))
    assert np.array_equal(output['to_compute'], clusters)
    
    assert np.array_equal(l.get_spikes(cluster_new), [])
    assert np.all(np.in1d(clusters, get_indices(l.get_cluster_groups('all'))))
    assert np.array_equal(l.get_clusters(clusters=clusters), cluster_spikes)
    
    # Redo.
    assert c.can_redo()
    action, output = c.redo()
    
    assert action == 'merge_clusters'
    assert np.array_equal(output['to_select'], cluster_new)
    assert np.array_equal(output['to_invalidate'], 
        sorted(set(clusters).union(set([cluster_new]))))
    assert np.array_equal(output['to_compute'], cluster_new)
    
    assert np.array_equal(l.get_spikes(cluster_new), spikes)
    assert np.all(~np.in1d(clusters, get_indices(l.get_cluster_groups('all'))))
    
def test_controller_split():
    l, c = load()
    
    # Select three clusters
    clusters = [2, 4, 6]
    spikes = l.get_spikes(clusters=clusters)
    cluster_spikes = l.get_clusters(clusters=clusters)
    # Select half of the spikes in these clusters.
    spikes_sample = spikes[::2]
    action, output = c.merge_clusters(clusters)
    cluster_new = output['to_select']
    
    # Split the newly created cluster into two clusters.
    action, output = c.split_clusters(cluster_new, spikes_sample)
    cluster_split = sorted(set(output['to_select']) - set([cluster_new]))[0]
    
    assert action == 'split_clusters'
    assert np.array_equal(output['to_select'], [cluster_new, cluster_split])
    assert np.array_equal(output['to_invalidate'], [cluster_new, cluster_split])
    assert np.array_equal(output['to_compute'], [cluster_new, cluster_split])
    
    assert np.array_equal(l.get_spikes(cluster_split), spikes_sample)
    
    
    # Undo.
    action, output = c.undo()
    
    assert action == 'split_clusters_undo'
    assert np.array_equal(output['to_select'], [cluster_new])
    assert np.array_equal(output['to_invalidate'], [cluster_new, cluster_split])
    assert np.array_equal(output['to_compute'], [cluster_new])
    
    assert np.array_equal(l.get_spikes(cluster_new), spikes)
    
    # Redo.
    action, output = c.redo()
    
    assert action == 'split_clusters'
    assert np.array_equal(output['to_select'], [cluster_new, cluster_split])
    assert np.array_equal(output['to_invalidate'], [cluster_new, cluster_split])
    assert np.array_equal(output['to_compute'], [cluster_new, cluster_split])
    
    assert np.array_equal(l.get_spikes(cluster_split), spikes_sample)
    
def test_controller_misc():
    l, c = load()
    
    # Select three clusters
    clusters = [2, 4, 6]
    spikes = l.get_spikes(clusters=clusters)
    cluster_spikes = l.get_clusters(clusters=clusters)
    
    
    # Merge these clusters.
    action, output = c.merge_clusters(clusters)
    cluster_new = output['to_select']
    assert np.array_equal(l.get_spikes(cluster_new), spikes)
    assert np.all(~np.in1d(clusters, get_indices(l.get_cluster_groups('all'))))
    
    
    # Undo.
    assert c.can_undo()
    c.undo()
    assert np.array_equal(l.get_spikes(cluster_new), [])
    assert np.all(np.in1d(clusters, get_indices(l.get_cluster_groups('all'))))
    assert np.array_equal(l.get_clusters(clusters=clusters), cluster_spikes)
    
    # Move clusters.
    action, output = c.move_clusters(clusters, 0)
    
    assert action == 'move_clusters'
    assert np.array_equal(output['to_select'], clusters)
    
    assert np.all(l.get_cluster_groups(clusters) == 0)
    
    assert c.can_undo()
    assert not c.can_redo()
    
    
    # Undo
    action, output = c.undo()
    
    assert action == 'move_clusters_undo'
    assert np.array_equal(output['to_select'], clusters)
    
    assert np.all(l.get_cluster_groups(clusters) == 2)
    
    
    assert not c.can_undo()
    assert c.can_redo()
    
    # Merge clusters.
    c.merge_clusters(clusters)
    
    assert c.can_undo()
    assert not c.can_redo()
    
def test_controller_recolor_clusters():
    l, c = load()
    group = 1
    cluster = 3
    
    # Change cluster color.
    color_old = l.get_cluster_colors(cluster)
    action, output = c.change_cluster_color(cluster, 12)
    
    assert action == 'change_cluster_color'
    assert output['to_select'] is None
    
    assert l.get_cluster_colors(cluster) == 12
    
    
    # Undo.
    action, output = c.undo()
    assert l.get_cluster_colors(cluster) == color_old
    
    assert action == 'change_cluster_color_undo'
    assert output['to_select'] is None
    
    
    # Redo.
    action, output = c.redo()
    assert l.get_cluster_colors(cluster) == 12
    
    assert action == 'change_cluster_color'
    assert output['to_select'] is None
    
def test_controller_move_clusters():
    l, c = load()
    group = 1
    clusters = [3, 5, 7]
    
    action, output = c.move_clusters(clusters, group)
    
    assert action == 'move_clusters'
    assert np.array_equal(output['to_select'], clusters)
    
    assert np.all(l.get_cluster_groups(clusters) == 1)
    
    
    # Undo.
    action, output = c.undo()
    
    assert action == 'move_clusters_undo'
    assert np.array_equal(output['to_select'], clusters)
    
    assert np.all(l.get_cluster_groups(clusters) == 2)
    
    
    # Redo.
    action, output = c.redo()
    
    assert action == 'move_clusters'
    assert np.array_equal(output['to_select'], clusters)
    
    assert np.all(l.get_cluster_groups(clusters) == 1)
    
def test_controller_rename_groups():
    l, c = load()
    group = 1
    
    # Rename groups.
    name = 'My group'
    action, output = c.rename_group(group, name)
    
    assert action == 'rename_group'
    
    
    # Undo.
    action, output = c.undo()
    
    assert action == 'rename_group_undo'
    
    assert l.get_group_names(group) == 'MUA'
    
    
    # Redo.
    action, output = c.redo()
    
    assert action == 'rename_group'
    
    assert l.get_group_names(group) == name
    
def test_controller_recolor_groups():
    l, c = load()
    group = 1
    
    # Change group color.
    color = l.get_group_colors(group)
    action, output = c.change_group_color(group, 10)
    
    assert action == 'change_group_color'
    
    assert l.get_group_colors(group) == 10
    
    
    # Undo.
    action, output = c.undo()
    
    assert action == 'change_group_color_undo'
    assert np.array_equal(output['group_to_select'], group)
    
    assert l.get_group_colors(group) == color
    
    
    # Redo.
    action, output = c.redo()
    
    assert action == 'change_group_color'
    assert np.array_equal(output['group_to_select'], group)
    
    assert l.get_group_colors(group) == 10
    
def test_controller_add_group():
    l, c = load()
    
    # Add a group.
    group = 3
    action, output = c.add_group(group, 'My group', 2)
    
    assert action == 'add_group'
    
    assert np.all(~np.in1d(l.get_cluster_groups(), group))
    assert l.get_group_names(group) == 'My group'
    assert l.get_group_colors(group) == 2
    
    
    # Undo.
    action, output = c.undo()
    
    assert action == 'add_group_undo'
    
    assert np.all(~np.in1d(l.get_cluster_groups(), group))
    
    
    # Redo.
    action, output = c.redo()
    
    assert action == 'add_group'
    
    assert np.all(~np.in1d(l.get_cluster_groups(), group))
    assert l.get_group_names(group) == 'My group'
    assert l.get_group_colors(group) == 2
    
def test_controller_remove_group():
    l, c = load()
    
    # Remove a group.
    group = 1
    action, output = c.remove_group(group)
    
    assert action == 'remove_group'
    
    assert np.all(~np.in1d(l.get_cluster_groups(), group))
    
    
    # Undo.
    action, output = c.undo()
    
    assert action == 'remove_group_undo'
    
    assert np.all(~np.in1d(l.get_cluster_groups(), group))
    assert l.get_group_names(group) == 'MUA'
    
    
    # Redo.
    action, output = c.redo()
    
    assert action == 'remove_group'
    
    assert np.all(~np.in1d(l.get_cluster_groups(), group))
    
    
    