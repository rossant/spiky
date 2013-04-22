"""This module implements a cache system for keeping cluster first- and
second-order statistics in memory, and updating them when necessary."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import Counter, namedtuple
from itertools import product

import numpy as np


# -----------------------------------------------------------------------------
# Cache
# -----------------------------------------------------------------------------
class StatsCache(object):
    """Keep cluster and cluster pair statistics in memory."""
    def __init__(self):
        self.cluster_stats = {}
        self.cluster_pair_stats = {}
    
    
    # Setter methods
    # --------------
    def set_cluster_stats(self, cluster, stats):
        self.cluster_stats[cluster] = stats

    def set_cluster_pair_stats(self, cluster0, cluster1, stats):
        self.cluster_pair_stats[cluster0, cluster1] = stats
    
    def __setitem__(self, key, value):
        if type(key) is tuple:
            self.set_cluster_pair_stats(key[0], key[1], value)
        else:
            self.set_cluster_stats(key, value)
    
    
    # Getter methods
    # --------------
    def get_cluster_stats(self, cluster):
        return self.cluster_stats.get(cluster)

    def get_cluster_pair_stats(self, cluster0, cluster1):
        return self.cluster_pair_stats.get((cluster0, cluster1))
    
    def __getitem__(self, key):
        if type(key) is tuple:
            if key in self.cluster_pair_stats:
                return self.get_cluster_pair_stats(key[0], key[1])
            else:
                return None
                # raise IndexError(("Cluster pair ({0:d}, {1:d}) is not "
                                  # "in the cache.").format(*key))
        else:
            if key in self.cluster_stats:
                return self.get_cluster_stats(key)
            else:
                return None
                # raise IndexError(("Cluster {0:d} is not "
                                  # "in the cache.").format(key))
            
    
    # Update methods
    # --------------
    def invalidate(self, clusters):
        # Remove cluster statistics.
        for cluster in clusters:
            self.clusters_stats.pop(cluster)
        # Remove cluster pairs statistics.
        for cluster0, cluster1 in self.cluster_pair_stats.keys():
            if cluster0 in clusters or cluster1 in clusters:
                self.cluster_pair_stats.pop((cluster0, cluster1))
    
        
if __name__ == '__main__':
    c = StatsCache()
    c[3] = dict(nspikes=3)
    c[4,3] = dict(correlograms=[1, 2, 3])
    
    print c[4,3]
        