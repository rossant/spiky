"""This module implements a cache system for keeping cluster first- and
second-order statistics in memory, and updating them when necessary."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import Counter, namedtuple
from itertools import product

import numpy as np

from spiky.stats.indexed_matrix import IndexedMatrix


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def is_default_slice(item):
    return (isinstance(item, slice) and item.start is None and item.stop is None
        and item.step is None)

def is_indices(item):
    return (isinstance(item, list) or isinstance(item, tuple) or 
        isinstance(item, np.ndarray) or isinstance(item, (int, long)))
        

# -----------------------------------------------------------------------------
# Stats cache
# -----------------------------------------------------------------------------
class StatsCache(object):
    def __init__(self, indices, ncorrbin=None):
        n = len(indices)
        self.correlograms = IndexedMatrix(indices,
            shape=(n, n, ncorrbin))
        self.correlation_matrix = IndexedMatrix(indices)
    
    def invalidate(self, clusters):
        self.correlograms.invalidate(clusters)
        self.correlation_matrix.invalidate(clusters)
        
        
        