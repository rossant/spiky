"""Unit tests for stats.cache module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from nose.tools import raises
import numpy as np

from spiky.stats.cache import IndexedMatrix, StatsCache


# -----------------------------------------------------------------------------
# Cache tests
# -----------------------------------------------------------------------------
def test_cache():
    indices = [2, 3, 5, 7]
    cache = StatsCache(indices, ncorrbin=50)
    
    np.array_equal(cache.correlograms.blank_indices(), indices)
    np.array_equal(cache.correlation_matrix.blank_indices(), indices)
    
    cache.correlograms[2, :] = 0
    cache.correlograms[:, 2] = 0
    
    np.array_equal(cache.correlograms.blank_indices(), [3, 5, 7])
    np.array_equal(cache.correlation_matrix.blank_indices(), [3, 5, 7])
    
    cache.invalidate(2)
    
    np.array_equal(cache.correlograms.blank_indices(), indices)
    np.array_equal(cache.correlation_matrix.blank_indices(), indices)
    
    
    