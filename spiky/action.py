import collections
import numpy as np
from colors import COLORMAP

class Action(object):
    def __init__(self, dh):
        self.dh = dh
        
    def set_params(self, *args, **kwargs):
        pass
    
    def execute(self):
        pass
        
    def unexecute(self):
        pass

    def __repr__(self):
        return super(Action, self).__repr__()

        
class MergeAction(Action):
    def set_params(self, clusters_to_merge):
        self.clusters_to_merge = np.array(clusters_to_merge)

    def execute(self):
        # if no clusters to merge: do nothing
        if len(self.clusters_to_merge) == 0:
            return
        
        # save old information (for redo)
        self.old_nclusters = self.dh.nclusters
        self.old_clusters = self.dh.clusters
        self.old_cluster_names = self.dh.clusters_info.names
        self.old_spkcounts = self.dh.clusters_info.spkcounts
        self.old_cluster_indices = self.dh.clusters_info.cluster_indices
        self.old_colors = self.dh.clusters_info.colors
        self.old_groups = self.dh.clusters_info.groups
        
        self.new_cluster = self.dh.new_cluster()
        
        clusters = self.dh.clusters.copy()
        # update the clusters array
        ind = np.in1d(clusters, self.clusters_to_merge)
        clusters[ind] = self.new_cluster
        
        spkcounts = collections.Counter(clusters)
        cluster_keys = sorted(spkcounts.keys())
        spkcounts = np.array([spkcounts[key] for key in cluster_keys])
        cluster_names = np.array(cluster_keys)#map(str, cluster_keys)
        nclusters = len(cluster_names)
        # for each cluster absolute index, its relative index
        cluster_indices = dict([(key, i) for i, key in enumerate(cluster_keys)])
        
        colors = np.zeros(nclusters, dtype=np.int32)
        groups = np.zeros(nclusters, dtype=np.int32)
        new_color = self.dh.clusters_info.colors[self.dh.clusters_info.cluster_indices[self.clusters_to_merge[0]]]
        new_group = self.dh.clusters_info.groups[self.dh.clusters_info.cluster_indices[self.clusters_to_merge[0]]]
        for cluster in cluster_names:
            # old and new cluster relative indices
            cluster_rel_new = cluster_indices[cluster]
            # if the cluster has not been changed
            if cluster != self.new_cluster and cluster not in self.clusters_to_merge:
                # new color = old color
                cluster_rel_old = self.dh.clusters_info.cluster_indices[cluster]
                colors[cluster_rel_new] = self.dh.clusters_info.colors[cluster_rel_old]
                groups[cluster_rel_new] = self.dh.clusters_info.groups[cluster_rel_old]
            # otherwise, set the new color
            else:
                colors[cluster_rel_new] = new_color
                groups[cluster_rel_new] = new_group
        
        # update
        self.dh.nclusters = nclusters
        self.dh.clusters = clusters
        self.dh.clusters_info.names = cluster_names
        self.dh.clusters_info.spkcounts = spkcounts
        self.dh.clusters_info.cluster_indices = cluster_indices
        self.dh.clusters_info.colors = colors
        self.dh.clusters_info.groups = groups
        
    def unexecute(self):
        
        # save old information (for redo)
        self.dh.nclusters = self.old_nclusters
        self.dh.clusters = self.old_clusters
        self.dh.clusters_info.names = self.old_cluster_names
        self.dh.clusters_info.spkcounts = self.old_spkcounts
        self.dh.clusters_info.cluster_indices = self.old_cluster_indices
        self.dh.clusters_info.colors = self.old_colors
        self.dh.clusters_info.groups = self.old_groups
        
    
class SplitAction(Action):
    def set_params(self, spikes_to_split):
        self.spikes_to_split = np.array(spikes_to_split)

    def execute(self):
        # if no clusters to merge: do nothing
        if len(self.spikes_to_split) == 0:
            return
        
        # save old information (for redo)
        self.old_nclusters = self.dh.nclusters
        self.old_clusters = self.dh.clusters
        self.old_cluster_names = self.dh.clusters_info.names
        self.old_spkcounts = self.dh.clusters_info.spkcounts
        self.old_cluster_indices = self.dh.clusters_info.cluster_indices
        self.old_colors = self.dh.clusters_info.colors
        self.old_groups = self.dh.clusters_info.groups
        
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
        
        # print clusters_to_split, new_clusters
        
        spkcounts = collections.Counter(clusters)
        cluster_keys = sorted(spkcounts.keys())
        spkcounts = np.array([spkcounts[key] for key in cluster_keys])
        cluster_names = np.array(cluster_keys)#map(str, cluster_keys)
        nclusters = len(cluster_names)
        # for each cluster absolute index, its relative index
        cluster_indices = dict([(key, i) for i, key in enumerate(cluster_keys)])
        
        colors = np.zeros(nclusters, dtype=np.int32)
        groups = np.zeros(nclusters, dtype=np.int32)
        new_colors = np.mod(self.old_colors[-1] + 1 + np.arange(nclusters_to_split), len(COLORMAP))
        new_groups = self.old_groups[[self.old_cluster_indices[c] for c in clusters_to_split]]
        i = 0
        for cluster in cluster_names:
            # old and new cluster relative indices
            cluster_rel_new = cluster_indices[cluster]
            # if the cluster has not been changed
            if cluster not in new_clusters:
                # new color = old color
                cluster_rel_old = self.dh.clusters_info.cluster_indices[cluster]
                colors[cluster_rel_new] = self.dh.clusters_info.colors[cluster_rel_old]
                groups[cluster_rel_new] = self.dh.clusters_info.groups[cluster_rel_old]
            # otherwise, set the new color
            else:
                colors[cluster_rel_new] = new_colors[i]
                groups[cluster_rel_new] = new_groups[i]
                i += 1
                
        # update
        self.dh.nclusters = nclusters
        self.dh.clusters = clusters
        self.dh.clusters_info.names = cluster_names
        self.dh.clusters_info.spkcounts = spkcounts
        self.dh.clusters_info.cluster_indices = cluster_indices
        self.dh.clusters_info.colors = colors
        self.dh.clusters_info.groups = groups
        
    def unexecute(self):
        
        # save old information (for redo)
        self.dh.nclusters = self.old_nclusters
        self.dh.clusters = self.old_clusters
        self.dh.clusters_info.names = self.old_cluster_names
        self.dh.clusters_info.spkcounts = self.old_spkcounts
        self.dh.clusters_info.cluster_indices = self.old_cluster_indices
        self.dh.clusters_info.colors = self.old_colors
        self.dh.clusters_info.groups = self.old_groups
        
        
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
        print dh.clusters_info.names
        print dh.clusters_info.spkcounts
        print dh.clusters_info.cluster_indices
        print dh.clusters_info.colors
        print
        
        am.do(MergeAction, [2,3], 100)
        
        print "*** MERGE 2, 3 ==> 100 ***"
        print dh.nclusters
        # print dh.clusters
        print dh.clusters_info.names
        print dh.clusters_info.spkcounts
        print dh.clusters_info.cluster_indices
        print dh.clusters_info.colors
        print
        
        am.undo()
        
        
        print "*** UNDO ***"
        print dh.nclusters
        # print dh.clusters
        print dh.clusters_info.names
        print dh.clusters_info.spkcounts
        print dh.clusters_info.cluster_indices
        print dh.clusters_info.colors
        print 
        
        
        
        am.do(MergeAction, [4,5,6,7], 200)
        
        print "*** MERGE 4, 5, 6, 7 ==> 200 ***"
        print dh.nclusters
        # print dh.clusters
        print dh.clusters_info.names
        print dh.clusters_info.spkcounts
        print dh.clusters_info.cluster_indices
        print dh.clusters_info.colors
        print
        
        am.undo()
        
        
        print "*** UNDO ***"
        print dh.nclusters
        # print dh.clusters
        print dh.clusters_info.names
        print dh.clusters_info.spkcounts
        print dh.clusters_info.cluster_indices
        print dh.clusters_info.colors
        print 
        
        
        am.redo()
        
        
        print "*** REDO ***"
        print dh.nclusters
        # print dh.clusters
        print dh.clusters_info.names
        print dh.clusters_info.spkcounts
        print dh.clusters_info.cluster_indices
        print dh.clusters_info.colors
        print 
        
    
    def split_test():
        print "*** INITIAL ***"
        print dh.nclusters
        print dh.clusters
        print dh.clusters_info.names
        print dh.clusters_info.spkcounts
        print dh.clusters_info.cluster_indices
        print dh.clusters_info.colors
        print
        
        am.do(SplitAction, [8])
        
        print "*** SPLIT 8 ***"
        print dh.nclusters
        print dh.clusters
        print dh.clusters_info.names
        print dh.clusters_info.spkcounts
        print dh.clusters_info.cluster_indices
        print dh.clusters_info.colors
        print
        
        am.undo()
        
        
        print "*** UNDO ***"
        print dh.nclusters
        print dh.clusters
        print dh.clusters_info.names
        print dh.clusters_info.spkcounts
        print dh.clusters_info.cluster_indices
        print dh.clusters_info.colors
        print 
        
        
        
        # am.do(SplitAction, np.arange(100))
        
        # print "*** SPLIT [0+1, 10+1, 20+1, 30+1, 40+1] ***"
        # print dh.nclusters
        # # print dh.clusters
        # print dh.clusters_info.names
        # print dh.clusters_info.spkcounts
        # print dh.clusters_info.cluster_indices
        # print dh.clusters_info.colors
        # print
        
        # am.undo()
        
        
        # print "*** UNDO ***"
        # print dh.nclusters
        # # print dh.clusters
        # print dh.clusters_info.names
        # print dh.clusters_info.spkcounts
        # print dh.clusters_info.cluster_indices
        # print dh.clusters_info.colors
        # print 
        
        
        # am.redo()
        
        
        # print "*** REDO ***"
        # print dh.nclusters
        # # print dh.clusters
        # print dh.clusters_info.names
        # print dh.clusters_info.spkcounts
        # print dh.clusters_info.cluster_indices
        # print dh.clusters_info.colors
        # print 
        
        
    split_test()