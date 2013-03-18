"""Unit tests for selection module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from spiky.io.selection import select, get_spikes_in_clusters, to_array


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def generate_clusters(indices, nspikes=100):
    """Generate all spikes in cluster 0, except some in cluster 1."""
    # 2 different clusters, with 3 spikes in cluster 1
    clusters = np.zeros(nspikes, dtype=np.int32)
    clusters[indices] = 1
    return clusters
    
def test_cluster_selection():
    indices = [10, 20, 25]
    clusters = generate_clusters(indices)
    clusters_selected = [1, 2]
    spikes = get_spikes_in_clusters(clusters_selected, clusters, False)
    assert np.array_equal(np.nonzero(spikes)[0], indices)
    spikes = get_spikes_in_clusters(clusters_selected, clusters, True)
    assert np.array_equal(spikes, indices)

def test_select_numpy():
    indices = [10, 20, 25]
    clusters = generate_clusters(indices)
    assert np.array_equal(select(clusters, [9, 11]), [0, 0])
    assert np.array_equal(select(clusters, [10, 99]), [1, 0])
    assert np.array_equal(select(clusters, [20, 25, 25]), [1, 1, 1])

def test_select_pandas():
    indices = [10, 20, 25]
    clusters = generate_clusters(indices)
    
    # test selection of Series (1D)
    clusters = pd.Series(clusters)
    assert np.array_equal(select(clusters, [9, 11]), [0, 0])
    assert np.array_equal(select(clusters, [10, 99]), [1, 0])
    assert np.array_equal(select(clusters, [20, 25, 25]), [1, 1, 1])
    
    # test selection of Series (3D)
    clusters = pd.DataFrame(clusters)
    assert np.array_equal(np.array(select(clusters, [9, 11])).ravel(), [0, 0])
    assert np.array_equal(np.array(select(clusters, [10, 99])).ravel(), [1, 0])
    assert np.array_equal(np.array(select(clusters, [20, 25, 25])).ravel(), [1, 1, 1])
    
    # test selection of Panel (4D)
    clusters = pd.Panel(np.expand_dims(clusters, 3))
    assert np.array_equal(np.array(select(clusters, [9, 11])).ravel(), [0, 0])
    assert np.array_equal(np.array(select(clusters, [10, 99])).ravel(), [1, 0])
    assert np.array_equal(np.array(select(clusters, [20, 25, 25])).ravel(), [1, 1, 1])
    
    # test recursive selection
    assert np.array_equal(to_array(select(select(clusters, [10, 25]), 25)), [1])
