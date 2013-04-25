"""The Controller offers high-level methods to change the data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import inspect
from collections import Counter

import numpy as np
import pandas as pd

from spiky.control.stack import Stack
import spiky.utils.logger as log
from spiky.io.selection import get_indices, select
from spiky.io.tools import get_array
from spiky.utils.colors import next_color


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def get_pretty_arg(item):
    if isinstance(item, (pd.Series)):
        return '[{0:s}, ..., {1:s}]'.format(*map(str, item.values[[0, -1]]))
    if isinstance(item, (pd.Int64Index, pd.Index)):
        return '[{0:s}, ..., {1:s}]'.format(*map(str, item.values[[0, -1]]))
    return str(item).replace('\n', '')

def get_pretty_action(method_name, args, kwargs, verb='Process'):
    args_str = ', '.join(map(get_pretty_arg, args))
    kwargs_str = ', '.join([key + '=' + str(val)
        for key, val in kwargs.iteritems()])
    if kwargs_str:
        kwargs_str = ', ' + kwargs_str
    return '{3:s} action {0:s}({1:s}{2:s})'.format(
        method_name, args_str, kwargs_str, verb)


# -----------------------------------------------------------------------------
# Controller
# -----------------------------------------------------------------------------
class Controller(object):
    """Implement actions that can be undone and redone.
    
    An Action object is:
        
        (method_name, args, kwargs)
    
    """
    def __init__(self, loader):
        self.loader = loader
        # Create the action stack.
        self.stack = Stack(maxsize=20)
    
    
    # Actions.
    # --------
    # Merge.
    def _merge_clusters(self, clusters_old, cluster_groups, cluster_colors,
        cluster_merged):
        # Get spikes in clusters to merge.
        # spikes = self.loader.get_spikes(clusters=clusters_to_merge)
        spikes = get_indices(clusters_old)
        clusters_to_merge = get_indices(cluster_groups)
        # Add new cluster.
        # group = self.loader.get_cluster_groups(clusters_to_merge) \
            # [clusters_to_merge[0]]
        # color_old = self.loader.get_cluster_colors(clusters_to_merge) \
            # [clusters_to_merge[0]]
        group = get_array(cluster_groups)[0]
        color_old = get_array(cluster_groups)[0]
        color_new = next_color(color_old)
        self.loader.add_cluster(cluster_merged, group, color_new)
        # Set the new cluster to the corresponding spikes.
        self.loader.set_cluster(spikes, cluster_merged)
        # Remove old clusters.
        for cluster in clusters_to_merge:
            self.loader.remove_cluster(cluster)
        self.loader.unselect()
        return 'merge', (clusters_to_merge, cluster_merged)
        
    def _merge_clusters_undo(self, clusters_old, cluster_groups, 
        cluster_colors, cluster_merged):
        # Get spikes in clusters to merge.
        spikes = self.loader.get_spikes(clusters=cluster_merged)
        clusters_to_merge = get_indices(cluster_groups)
        # Add old clusters.
        for cluster, group, color in zip(
                clusters_to_merge, cluster_groups, cluster_colors):
            self.loader.add_cluster(cluster, group, color)
        # Set the new clusters to the corresponding spikes.
        self.loader.set_cluster(spikes, clusters_old)
        # Remove merged cluster.
        self.loader.remove_cluster(cluster_merged)
        self.loader.unselect()
        return 'merge_undo', (clusters_to_merge, cluster_merged)
        
        
    # Split.
    def _split_clusters(self, clusters_old, cluster_groups, 
        cluster_colors, clusters_new):
        spikes = get_indices(clusters_old)
        # Find groups and colors of old clusters.
        cluster_indices_old = sorted(Counter(clusters_old).keys())
        cluster_indices_new = sorted(Counter(clusters_new).keys())
        # Get group and color of the new clusters, from the old clusters.
        groups = self.loader.get_cluster_groups(cluster_indices_old)
        colors = self.loader.get_cluster_colors(cluster_indices_old)
        # Add clusters.
        for cluster_new, group, color in zip(cluster_indices_new, 
                groups, colors):
            self.loader.add_cluster(cluster_new, group, next_color(color))
        # clusters_empty = sorted(set(cluster_indices_old) - 
            # set(cluster_indices_new))
        # print cluster_indices_old, cluster_indices_new, clusters_empty
        # for cluster in clusters_empty:
            # self.loader.remove_cluster(cluster)
        # Set the new clusters to the corresponding spikes.
        self.loader.set_cluster(spikes, clusters_new)
        # Remove empty clusters.
        clusters_empty = self.loader.remove_empty_clusters()
        self.loader.unselect()
        # return np.array(sorted(set(cluster_indices_old).union(set(cluster_indices_new))))
        return 'split', (cluster_indices_old, cluster_indices_new,
            clusters_empty)
        
    def _split_clusters_undo(self, clusters_old, cluster_groups, 
        cluster_colors, clusters_new):
        spikes = get_indices(clusters_old)
        # Find groups and colors of old clusters.
        cluster_indices_old = sorted(Counter(clusters_old).keys())
        cluster_indices_new = sorted(Counter(clusters_new).keys())
        # Add clusters that were removed after the split operation.
        clusters_empty = sorted(set(cluster_indices_old) - 
            set(cluster_indices_new))
        for cluster in clusters_empty:
            self.loader.add_cluster(cluster, select(cluster_groups, cluster),
                select(cluster_colors, cluster))
        # Set the new clusters to the corresponding spikes.
        self.loader.set_cluster(spikes, clusters_old)
        # Remove clusters.
        # for cluster_new in cluster_indices_new:
            # self.loader.remove_cluster(cluster_new)
        # Remove empty clusters.
        clusters_empty = self.loader.remove_empty_clusters()
        self.loader.unselect()
        # return cluster_indices_old
        return 'split_undo', (cluster_indices_old, cluster_indices_new, 
            clusters_empty)
        
        
    # Change cluster color.
    def _change_cluster_color(self, cluster, color_old, color_new):
        self.loader.set_cluster_colors(cluster, color_new)
        
    def _change_cluster_color_undo(self, cluster, color_old, color_new):
        self.loader.set_cluster_colors(cluster, color_old)
        
        
    # Move clusters.
    def _move_clusters(self, clusters, groups_old, group_new):
        self.loader.set_cluster_groups(clusters, group_new)
        
    def _move_clusters_undo(self, clusters, groups_old, group_new):
        self.loader.set_cluster_groups(clusters, groups_old)
      
      
    # Rename group.
    def _rename_group(self, group, name_old, name_new):
        self.loader.set_group_names(group, name_new)
        
    def _rename_group_undo(self, group, name_old, name_new):
        self.loader.set_group_names(group, name_old)
    
    
    # Change group color.
    def _change_group_color(self, group, color_old, color_new):
        self.loader.set_group_colors(group, color_new)
        
    def _change_group_color_undo(self, group, color_old, color_new):
        self.loader.set_group_colors(group, color_old)
    
    
    # Add group.
    def _add_group(self, group, name, color):
        self.loader.add_group(group, name, color)
        
    def _add_group_undo(self, group, name, color):
        self.loader.remove_group(group)
    
    
    # Remove group.
    def _remove_group(self, group, name, color):
        self.loader.remove_group(group)
        
    def _remove_group_undo(self, group, name, color):
        self.loader.add_group(group, name, color)
    
    
    
    # Internal action methods.
    # ------------------------
    def _process(self, method_name, *args, **kwargs):
        """Create, register, and process an action."""
        # Create the action.
        action = (method_name, args, kwargs)
        # Add the action to the stack.
        self.stack.add(action)
        # Log the action.
        log.info(get_pretty_action(method_name, args, kwargs))
        # Process the action.
        # The actual action is implemented in '_method' with a leading
        # underscore.
        r = getattr(self, '_' + method_name)(*args, **kwargs)
        if r is None:
            r = method_name, (args, kwargs)
        return r
    
    def __getattr__(self, method_name):
        assert inspect.ismethod('_' + method_name)
        return lambda *args, **kwargs: self._process(method_name, 
            *args, **kwargs)
    
    
    # Public action methods.
    # ----------------------
    def merge_clusters(self, clusters):
        clusters_to_merge = clusters
        cluster_merged = self.loader.get_new_clusters(1)[0]
        clusters_old = self.loader.get_clusters(clusters=clusters_to_merge)
        cluster_groups = self.loader.get_cluster_groups(clusters_to_merge)
        cluster_colors = self.loader.get_cluster_colors(clusters_to_merge)
        return self._process('merge_clusters', clusters_old, cluster_groups, 
            cluster_colors, cluster_merged)
        # return cluster_merged
        
    def split_clusters(self, clusters, spikes):
        # Old clusters for all spikes to split.
        clusters_old = self.loader.get_clusters(spikes=spikes)
        assert np.all(np.in1d(clusters_old, clusters))
        # Old cluster indices.
        cluster_indices_old = np.sort(Counter(clusters_old).keys())
        nclusters = len(cluster_indices_old)
        # New clusters indices.
        clusters_indices_new = self.loader.get_new_clusters(nclusters)
        # Generate new clusters array.
        clusters_new = clusters_old.copy()
        # Assign new clusters.
        for cluster_old, cluster_new in zip(cluster_indices_old,
                clusters_indices_new):
            clusters_new[clusters_old == cluster_old] = cluster_new
        cluster_groups = self.loader.get_cluster_groups(cluster_indices_old)
        cluster_colors = self.loader.get_cluster_colors(cluster_indices_old)
        return self._process('split_clusters', clusters_old, cluster_groups,
            cluster_colors, clusters_new)
        # return clusters_indices_new
        
    def change_cluster_color(self, cluster, color):
        color_old = self.loader.get_cluster_colors(cluster)
        color_new = color
        self._process('change_cluster_color', cluster, color_old, color_new)
        
    def move_clusters(self, clusters, group):
        groups_old = self.loader.get_cluster_groups(clusters)
        group_new = group
        self._process('move_clusters', clusters, groups_old, group_new)
      
    def rename_group(self, group, name):
        name_old = self.loader.get_group_names(group)
        name_new = name
        self._process('rename_group', group, name_old, name_new)
        
    def change_group_color(self, group, color):
        color_old = self.loader.get_group_colors(group)
        color_new = color
        self._process('change_group_color', group, color_old, color_new)
    
    def add_group(self, group, name, color):
        self._process('add_group', group, name, color)
        
    def remove_group(self, group):
        name = self.loader.get_group_names(group)
        color = self.loader.get_group_colors(group)
        self._process('remove_group', group, name, color)
        
    
    
    # Stack methods.
    # --------------
    def undo(self):
        """Undo an action if possible."""
        action = self.stack.undo()
        if action is None:
            return
        # Get the undone action.
        method_name, args, kwargs = action
        # Undo the action.
        # Log the action.
        log.info(get_pretty_action(method_name, args, kwargs, verb='Undo'))
        # The undo action is implemented in '_method_undo'.
        return getattr(self, '_' + method_name + '_undo')(*args, **kwargs)
        
    def redo(self):
        action = self.stack.redo()
        if action is None:
            return
        # Get the redo action.
        method_name, args, kwargs = action
        # Redo the action.
        # Log the action.
        log.info(get_pretty_action(method_name, args, kwargs, verb='Redo'))
        # The redo action is implemented in '_method'.
        return getattr(self, '_' + method_name)(*args, **kwargs)
        
    def can_undo(self):
        return self.stack.can_undo()
        
    def can_redo(self):
        return self.stack.can_redo()

