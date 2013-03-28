"""Functions for selecting portions of arrays."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Selection functions
# -----------------------------------------------------------------------------
def select_numpy(data, spikes):
    """Select a portion of an array with the corresponding spike indices.
    The first axis of data corresponds to the spikes."""
    
    assert isinstance(data, np.ndarray)
    assert isinstance(spikes, np.ndarray)
    
    # spikes can contain boolean masks...
    if spikes.dtype == np.bool:
        data_selection = np.compress(spikes, data, axis=0)
    # or spike indices.
    else:
        data_selection = np.take(data, spikes, axis=0)
    return data_selection

def select_pandas(data, spikes):
    return data.ix[spikes]

def select(data, indices=None):
    """Select portion of the data, with the only assumption that indices are
    along the first axis.
    
    data can be a NumPy or Pandas object.
    
    """
    # indices=None means select all.
    if indices is None:
        return data
        
    if not hasattr(indices, '__len__'):
        indices = [indices]
        
    # Ensure indices is an array of indices or boolean masks.
    if not isinstance(indices, np.ndarray):
        # Deal with empty indices.
        if not len(indices):
            if data.ndim == 1:
                return np.array([])
            elif data.ndim == 2:
                return np.array([[]])
            elif data.ndim == 3:
                return np.array([[[]]])
        else:
            if type(indices[0]) in (int, np.int32, np.int64):
                indices = np.array(indices, dtype=np.int32)
            elif type(indices[0]) == bool:
                indices = np.array(indices, dtype=np.bool)
            else:
                indices = np.array(indices)
    
    # Use NumPy or Pandas version
    if type(data) == np.ndarray:
        return select_numpy(data, indices)
    else:
        return select_pandas(data, indices)

def get_spikes_in_clusters(clusters_selected, clusters, return_indices=False):
    spike_indices = np.in1d(clusters, clusters_selected)
    if not return_indices:
        return spike_indices
    else:
        return np.nonzero(spike_indices)[0]
    
def to_array(data):
    """Convert a Pandas object to a NumPy array."""
    return np.atleast_1d(np.array(data).squeeze())
    
