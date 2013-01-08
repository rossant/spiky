import collections
import numpy as np


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
    def set_params(self, clusters_to_merge, new_cluster):
        self.clusters_to_merge = np.array(clusters_to_merge)
        self.new_cluster = new_cluster

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
        
    
    
        
if __name__ == '__main__':
    from spiky import *
    pr = KlustersDataProvider()
    pr.load("../experimental/data/subset41test")
    
    dh = pr.holder
    am = ActionManager(dh)
    
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
    