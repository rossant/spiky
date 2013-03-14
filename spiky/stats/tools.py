"""Tools for computation of cluster statistics."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np



def matrix_of_pairs(dict):
    """Convert a dictionary (ci, cj) => value to a matrix."""
    keys = np.array(dict.keys())
    max = keys.max()
    matrix = np.zeros((max + 1, max + 1))
    for (ci, cj), val in dict.iteritems():
        matrix[ci, cj] = val
    return matrix

    