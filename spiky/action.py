import collections
from copy import deepcopy as dcopy
import numpy as np
from colors import COLORMAP
from dataio import get_clusters_info
import spiky.tasks as tasks


# Base class for actions
# ----------------------
class Action(object):
    # Initialization methods
    # ----------------------
    def __init__(self, dh, sdh):
        self.dh = dh
        self.sdh = sdh
        
    def set_params(self, *args, **kwargs):
        pass
    
    
    # Execution methods
    # -----------------
    def execute(self):
        pass
        
    def unexecute(self):
        pass

    def reexecute(self):
        pass

        
    # Selection methods
    # -----------------
    def selected_clusters_after_undo(self):
        return []

    def selected_clusters_after_redo(self):
        return []
        
        
    # State methods
    # -------------
    def save_oldstate(self):
        # save old information (for undo)
        self._old_nclusters = self.dh.nclusters
        self._old_clusters = self.dh.clusters
        self._old_clusters_info = dcopy(self.dh.clusters_info['clusters_info'])
        self._old_groups_info = dcopy(self.dh.clusters_info['groups_info'])
        
    def restore_oldstate(self):
        # restore old information (for undo)
        self.dh.nclusters = self._old_nclusters
        self.dh.clusters = self._old_clusters
        self.dh.clusters_info['clusters_info'] = self._old_clusters_info
        self.dh.clusters_info['groups_info'] = self._old_groups_info
       
    def save_newstate(self):
        # save old information (for undo)
        self._new_nclusters = self.dh.nclusters
        self._new_clusters = self.dh.clusters
        self._new_clusters_info = dcopy(self.dh.clusters_info['clusters_info'])
        self._new_groups_info = dcopy(self.dh.clusters_info['groups_info'])
        
    def restore_newstate(self):
        # restore old information (for undo)
        self.dh.nclusters = self._new_nclusters
        self.dh.clusters = self._new_clusters
        self.dh.clusters_info['clusters_info'] = self._new_clusters_info
        self.dh.clusters_info['groups_info'] = self._new_groups_info

    def __repr__(self):
        return super(Action, self).__repr__()


# Merge/Split actions
# -------------------
class MergeAction(Action):
    def set_params(self, clusters_to_merge):
        self.clusters_to_merge = np.array(clusters_to_merge)

    def execute(self):
        # if no clusters to merge: do nothing
        if len(self.clusters_to_merge) == 0:
            return
        
        # invalidate the cross correlograms of the clusters to merge
        self.sdh.invalidate(self.clusters_to_merge)
        
        # get the index of the new cluster
        self.new_cluster = self.dh.new_cluster()
        
        # copy and update the clusters array
        clusters = self.dh.clusters.copy()
        # update the clusters array
        ind = np.in1d(clusters, self.clusters_to_merge)
        # if old cluster in clusters to merge, then assign to new cluster
        clusters[ind] = self.new_cluster
        
        # get clusters info
        clusters_info = get_clusters_info(clusters)
        nclusters = len(clusters_info)
        
        colors = np.zeros(nclusters, dtype=np.int32)
        groups = np.zeros(nclusters, dtype=np.int32)
        new_color = self.dh.clusters_info['clusters_info'][self.clusters_to_merge[0]]['color']
        new_group = self.dh.clusters_info['clusters_info'][self.clusters_to_merge[0]]['groupidx']
        for clusteridx, info in clusters_info.iteritems():
            # if the cluster has not been changed
            if clusteridx != self.new_cluster and clusteridx not in self.clusters_to_merge:
                # new color = old color
                info['color'] = self.dh.clusters_info['clusters_info'][clusteridx]['color']
                info['groupidx'] = self.dh.clusters_info['clusters_info'][clusteridx]['groupidx']
            # otherwise, set the new color
            else:
                info['color'] = new_color
                info['groupidx'] = new_group
        
        # update
        self.dh.nclusters = nclusters
        self.dh.clusters = clusters
        self.dh.clusters_info['clusters_info'] = clusters_info
        
        # Update correlation matrix.
        tasks.TASKS.correlation_matrix_queue.process(self.dh)
        

    def unexecute(self):
        self.sdh.invalidate(self.clusters_to_merge)
        
        # Update correlation matrix.
        tasks.TASKS.correlation_matrix_queue.process(self.dh)

    def reexecute(self):
        self.sdh.invalidate(self.clusters_to_merge)
        
        # Update correlation matrix.
        tasks.TASKS.correlation_matrix_queue.process(self.dh)
        
    def selected_clusters_after_undo(self):
        return self.clusters_to_merge

    def selected_clusters_after_redo(self):
        return [self.new_cluster]
        
    
class SplitAction(Action):
    def set_params(self, spikes_to_split):
        self.spikes_to_split = np.array(spikes_to_split)

    # @profile
    def execute(self):
        
        # if no clusters to merge: do nothing
        if len(self.spikes_to_split) == 0:
            return
        
        clusters = self.dh.clusters.copy()
        
        # array with clusters to split
        clusters_to_split = np.unique(self._old_clusters[self.spikes_to_split])
        clusters_to_split.sort()
        nclusters_to_split = len(clusters_to_split)
        # create nclusters_to_split new clusters
        nc = self.dh.new_cluster()
        new_clusters = np.arange(nc, nc + nclusters_to_split)
        
        # invalidate the cross correlograms of the clusters to merge
        self.sdh.invalidate(clusters_to_split)
        
        for cluster, new_cluster in zip(clusters_to_split, new_clusters):
            # spikes which are in the current cluster
            spikes_in_cluster = np.nonzero(self._old_clusters == cluster)[0]
            # the spikes in the current cluster to be split
            spikes_in_cluster_to_split = np.in1d(spikes_in_cluster, self.spikes_to_split)
            spikes_in_cluster_to_split = spikes_in_cluster[spikes_in_cluster_to_split]
            # assign the new cluster to the split spikes
            clusters[spikes_in_cluster_to_split] = new_cluster
            
        # record the list of newly created clusters
        self.new_clusters = new_clusters
        self.clusters_to_split = clusters_to_split
        
        # get clusters info
        clusters_info = get_clusters_info(clusters)
        nclusters = len(clusters_info)
        
        for clusteridx, info in clusters_info.iteritems():
            # if the cluster has not been changed
            if clusteridx not in new_clusters:
                info['color'] = self.dh.clusters_info['clusters_info'][clusteridx]['color']
                info['groupidx'] = self.dh.clusters_info['clusters_info'][clusteridx]['groupidx']
                
        # group of new cluster = group of corresponding old cluster
        for old_clusteridx, new_clusteridx in zip(clusters_to_split, new_clusters):
            clusters_info[new_clusteridx]['groupidx'] = self._old_clusters_info[old_clusteridx]['groupidx']
                
                
        # update
        self.dh.nclusters = nclusters
        self.dh.clusters = clusters
        self.dh.clusters_info['clusters_info'] = clusters_info
        
        # Update correlation matrix.
        tasks.TASKS.correlation_matrix_queue.process(self.dh)

    def unexecute(self):
        self.sdh.invalidate(self.clusters_to_split)
        
        # Update correlation matrix.
        tasks.TASKS.correlation_matrix_queue.process(self.dh)
        
    def reexecute(self):
        self.sdh.invalidate(self.clusters_to_split)
        
        # Update correlation matrix.
        tasks.TASKS.correlation_matrix_queue.process(self.dh)
        
    def selected_clusters_after_undo(self):
        return self.clusters_to_split

    def selected_clusters_after_redo(self):
        return np.hstack((self.clusters_to_split, self.new_clusters))
        

# Clusters info/Group actions
# ---------------------------
class MoveToGroupAction(Action):
    def set_params(self, clusters, groupidx):
        self.clusters = clusters
        self.groupidx = groupidx
        
    def execute(self):
        for clusteridx in self.clusters:
            self.dh.clusters_info['clusters_info'][clusteridx]['groupidx'] = self.groupidx
        
        # Update correlation matrix.
        tasks.TASKS.correlation_matrix_queue.process(doupdate=False)
        
    def selected_clusters_after_undo(self):
        return self.clusters

    def selected_clusters_after_redo(self):
        return self.clusters
        
        
class ChangeGroupColorAction(Action):
    def set_params(self, groups, color):
        self.groups = groups
        self.color = color
        
    def execute(self):
        for groupidx in self.groups:
            self.dh.clusters_info['groups_info'][groupidx]['color'] = self.color
        
        
class ChangeClusterColorAction(Action):
    def set_params(self, clusters, color):
        self.clusters = clusters
        self.color = color
        
    def execute(self):
        for clusteridx in self.clusters:
            self.dh.clusters_info['clusters_info'][clusteridx]['color'] = self.color
        
    def selected_clusters_after_undo(self):
        return self.clusters

    def selected_clusters_after_redo(self):
        return self.clusters
        
    
        
class RenameGroupAction(Action):
    def set_params(self, groupidx, name):
        self.groupidx = groupidx
        self.name = name
        
    def execute(self):
        self.dh.clusters_info['groups_info'][self.groupidx]['name'] = self.name
        
        
class AddGroupAction(Action):
    def execute(self):
        self.groupidx = max(self.dh.clusters_info['groups_info'].keys()) + 1
        self.color = np.mod(max([self.dh.clusters_info['groups_info'][c]['color'] for c in self.dh.clusters_info['groups_info'].keys()]) + 1, len(COLORMAP))
        self.name = "Group %d" % self.groupidx
        self.dh.clusters_info['groups_info'][self.groupidx] = {
            'groupidx': self.groupidx,
            'name': self.name,
            'spkcount': 0,
            'color': self.color
        }
        
        
class RemoveGroupsAction(Action):
    def set_params(self, groups):
        self.groups = groups
        
    def execute(self):
        for groupidx in self.groups:
            del self.dh.clusters_info['groups_info'][groupidx]
        
        

# Action Manager
# --------------
class ActionManager(object):
    def __init__(self, dh, sdh):
        self.stack = []
        self.unstack = []
        self.dh = dh
        self.sdh = sdh
        
    def do(self, action_class, *args, **kwargs):
        action = action_class(self.dh, self.sdh)
        action.set_params(*args, **kwargs)
        action.save_oldstate()
        action.execute()
        action.save_newstate()
        self.stack.append(action)
        # when adding a new action, clear the unstack
        self.unstack = []
        return action
        
    def undo(self):
        if len(self.stack) > 0:
            action = self.stack.pop()
            self.unstack.append(action)
            action.unexecute()
            action.restore_oldstate()
            return action
        
    def redo(self):
        if len(self.unstack) > 0:
            action = self.unstack.pop()
            self.stack.append(action)
            action.reexecute()
            action.restore_newstate()
            return action
            
    def undo_enabled(self):
        return len(self.stack) > 0
            
    def redo_enabled(self):
        return len(self.unstack) > 0
        
    
if __name__ == '__main__':
    from spiky import *
    pr = KlustersDataProvider()
    pr.load("../experimental/data/subset41test")
    
    dh = pr.holder
    am = ActionManager(dh)
    
    def merge_test():
        print "*** INITIAL ***"
        print dh.nclusters
        # print dh.clusters
        print dh.clusters_info
        print
        
        am.do(MergeAction, [2,3])
        
        print "*** MERGE 2, 3 ==> 100 ***"
        print dh.nclusters
        # print dh.clusters
        print dh.clusters_info
        print
        
        am.undo()
        
        
        print "*** UNDO ***"
        print dh.nclusters
        # print dh.clusters
        print dh.clusters_info
        print 
        
        
        
        am.do(MergeAction, [4,5,6,7])
        
        print "*** MERGE 4, 5, 6, 7 ==> 200 ***"
        print dh.nclusters
        # print dh.clusters
        print dh.clusters_info
        print
        
        am.undo()
        
        
        print "*** UNDO ***"
        print dh.nclusters
        # print dh.clusters
        print dh.clusters_info
        print 
        
        
        am.redo()
        
        
        print "*** REDO ***"
        print dh.nclusters
        # print dh.clusters
        print dh.clusters_info
        print 
        
    
    def split_test():
        print "*** INITIAL ***"
        print dh.nclusters
        print dh.clusters
        print dh.clusters_info
        print
        
        am.do(SplitAction, [8])
        
        print "*** SPLIT 8 ***"
        print dh.nclusters
        print dh.clusters
        print dh.clusters_info
        print
        
        am.undo()
        
        
        print "*** UNDO ***"
        print dh.nclusters
        print dh.clusters
        print dh.clusters_info
        print 
        
        
    merge_test()
    # split_test()