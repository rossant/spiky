import collections
from copy import deepcopy as dcopy
import numpy as np
from colors import COLORMAP
from dataio import get_clusters_info

class Action(object):
    def __init__(self, dh):
        self.dh = dh
        
    def set_params(self, *args, **kwargs):
        pass
    
    def execute(self):
        pass
        
    def unexecute(self):
        self.restore_state()

    def __repr__(self):
        return super(Action, self).__repr__()

    def save_state(self):
        # save old information (for redo)
        self.old_nclusters = self.dh.nclusters
        self.old_clusters = self.dh.clusters
        self.old_clusters_info = dcopy(self.dh.clusters_info['clusters_info'])
        self.old_groups_info = dcopy(self.dh.clusters_info['groups_info'])
        
    def restore_state(self):
        # restore old information (for redo)
        self.dh.nclusters = self.old_nclusters
        self.dh.clusters = self.old_clusters
        self.dh.clusters_info['clusters_info'] = self.old_clusters_info
        self.dh.clusters_info['groups_info'] = self.old_groups_info
        
        
class MergeAction(Action):
    def set_params(self, clusters_to_merge):
        self.clusters_to_merge = np.array(clusters_to_merge)

    def execute(self):
        # if no clusters to merge: do nothing
        if len(self.clusters_to_merge) == 0:
            return
        
        # save old information (for redo)
        self.save_state()
        
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
        
    
class SplitAction(Action):
    def set_params(self, spikes_to_split):
        self.spikes_to_split = np.array(spikes_to_split)

    def execute(self):
        # if no clusters to merge: do nothing
        if len(self.spikes_to_split) == 0:
            return
        
        # save old information (for redo)
        self.save_state()
        
        clusters = self.dh.clusters.copy()
        
        # array with clusters to split
        clusters_to_split = np.unique(self.old_clusters[self.spikes_to_split])
        clusters_to_split.sort()
        nclusters_to_split = len(clusters_to_split)
        # create nclusters_to_split new clusters
        nc = self.dh.new_cluster()
        new_clusters = np.arange(nc, nc + nclusters_to_split)
        
        for cluster, new_cluster in zip(clusters_to_split, new_clusters):
            # spikes which are in the current cluster
            spikes_in_cluster = np.nonzero(self.old_clusters == cluster)[0]
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
        
        # colors = np.zeros(nclusters, dtype=np.int32)
        # groups = np.zeros(nclusters, dtype=np.int32)
        for clusteridx, info in clusters_info.iteritems():
            # if the cluster has not been changed
            if clusteridx not in new_clusters:
                info['color'] = self.dh.clusters_info['clusters_info'][clusteridx]['color']
                info['groupidx'] = self.dh.clusters_info['clusters_info'][clusteridx]['groupidx']
                
        # group of new cluster = group of corresponding old cluster
        for old_clusteridx, new_clusteridx in zip(clusters_to_split, new_clusters):
            clusters_info[new_clusteridx]['groupidx'] = self.old_clusters_info[old_clusteridx]['groupidx']
                
                
        # update
        self.dh.nclusters = nclusters
        self.dh.clusters = clusters
        self.dh.clusters_info['clusters_info'] = clusters_info
        
        
class ActionManager(object):
    def __init__(self, dh):
        self.stack = []
        self.unstack = []
        self.dh = dh
        
    def do(self, action_class, *args, **kwargs):
        action = action_class(self.dh)
        action.set_params(*args, **kwargs)
        action.execute()
        self.stack.append(action)
        # when adding a new action, clear the unstack
        self.unstack = []
        return action
        
    def undo(self):
        if len(self.stack) > 0:
            action = self.stack.pop()
            self.unstack.append(action)
            action.unexecute()
            return action
        
    def redo(self):
        if len(self.unstack) > 0:
            action = self.unstack.pop()
            self.stack.append(action)
            action.execute()
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