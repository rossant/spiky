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

def select(data, spikes=None):
    """Select portion of the data, with the only assumption that spikes are
    along the first axis.
    
    data can be a NumPy or Pandas object.
    
    """
    # spikes=None means select all.
    if spikes is None:
        return data
        
    if not hasattr(spikes, '__len__'):
        spikes = [spikes]
        
    # Ensure spikes is an array of indices or boolean masks.
    if not isinstance(spikes, np.ndarray):
        # Deal with empty spikes.
        if not len(spikes):
            if data.ndim == 1:
                return np.array([])
            elif data.ndim == 2:
                return np.array([[]])
            elif data.ndim == 3:
                return np.array([[[]]])
        else:
            if type(spikes[0]) in (int, np.int32, np.int64):
                spikes = np.array(spikes, dtype=np.int32)
            elif type(spikes[0]) == bool:
                spikes = np.array(spikes, dtype=np.bool)
            else:
                spikes = np.array(spikes)
    
    # Use NumPy or Pandas version
    if type(data) == np.ndarray:
        return select_numpy(data, spikes)
    else:
        return select_pandas(data, spikes)

def get_spikes_in_clusters(clusters_selected, clusters, return_indices=False):
    spike_indices = np.in1d(clusters, clusters_selected)
    if not return_indices:
        return spike_indices
    else:
        return np.nonzero(spike_indices)[0]
    
def to_array(data):
    """Convert a Pandas object to a NumPy array."""
    return np.atleast_1d(np.array(data).squeeze())
    
