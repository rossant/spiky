"""Unit tests for stats.cache module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from spiky.stats.cache import StatsCache


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_cache():
    cache = StatsCache()
    
    x = np.random.rand(10)
    y = np.random.rand(10)
    
    cache[2] = x
    cache[2,3] = y
    
    assert np.array_equal(cache[2], x)
    assert cache[3] == None
    assert np.array_equal(cache[2,3], y)
    assert cache[2,2] == None
    
    
    