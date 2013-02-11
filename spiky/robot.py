import numpy as np


class Robot(object):
    def __init__(self, dh):
        self.dh = dh
        self.nclusters = self.dh.nclusters
        
    def get_similar_clusters(self):
        indices = np.argsort(self.dh.correlation_matrix, axis=None)[::-1]
        best_pairs = [(i // self.nclusters, np.mod(i, self.nclusters)) for i in indices]
        return best_pairs
    
    def on_merge(self, clusters, newcluster):
        pass
        
    def on_split(self, spikes_to_split, clusters_to_split, new_clusters):
        pass


if __name__ == '__main__':
    
    from spiky import *
    prov = MockDataProvider()
    prov.load()
    
    dh = prov.holder
    
    robot = Robot(dh)
    print robot.get_similar_clusters()
    