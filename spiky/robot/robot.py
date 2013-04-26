"""Robot selecting automatically the best clusters to show to the user."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import Counter

import numpy as np


# -----------------------------------------------------------------------------
# Robot
# -----------------------------------------------------------------------------
class Robot(object):
    """Robot object, takes the data parameters and returns propositions of
    clusters to select."""
    def __init__(self, features=None, spiketimes=None, clusters=None, 
        masks=None, cluster_groups=None):
        self.features = features
        self.spiketimes = spiketimes
        self.clusters = clusters
        self.masks = masks
        self.cluster_groups = cluster_groups
        self.correlograms = None
        self.correlation_matrix = None
        
    
    # Internal methods.
    # -----------------
    def _update(self):
        if self.clusters is not None:
            self.clusters_unique = np.array(sorted(Counter(
                self.clusters).keys()))
    
    
    # Data update methods.
    # --------------------
    def update(self, **kwargs):
        for key, value in kwargs.iteritems():
            setattr(self, key, value)
        # Update the internal variables.
        self._update()
            
    def merged(self, clusters_to_merge, cluster_new):
        """Called to signify the robot that a merge has happened.
        No data update happens here, rather, self.update needs to be called
        with the updated data."""
        pass
            
    def split(self, clusters_old, clusters_new):
        """Called to signify the robot that a split has happened."""
        pass
        
    def correlograms_updated(self, correlograms):
        """Called to signify the robot that the correlograms have been
        updated."""
        pass
        
    def correlation_matrix_updated(self, correlation_matrix):
        """Called to signify the robot that the correlation matrix has been
        updated."""
        pass
        
    
    # Robot output methods.
    # ---------------------
    def next_clusters(self):
        """Return a set of clusters that are candidates for merging.
        """
        # Stupid robot.
        if len(self.clusters_unique) >= 2:
            return np.random.choice(self.clusters_unique, 2, replace=False)
        
        return None
    
    
    