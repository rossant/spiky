"""This module implements the computation of the cross-correlograms between
clusters."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import Counter
from itertools import product

import numpy as np


# -----------------------------------------------------------------------------
# Cross-correlograms
# -----------------------------------------------------------------------------
def compute_spike_delays(spiketimes, clusters, clusters_to_update=None,
    halfwidth=None):
    """Compute the delays between every pair of spikes within width.
    
    The correlograms are histograms of these delays.
    
    """
    # size of the histograms
    nspikes = len(spiketimes)
    
    # delays will contain all delays for each pair of clusters
    delays = {}

    # unique clusters
    counter = Counter(clusters)
    clusters_unique = sorted(counter.keys())
    nclusters = len(clusters_unique)
    cluster_max = clusters_unique[-1]
    
    # clusters to update
    if clusters_to_update is None:
        clusters_to_update = clusters_unique
    clusters_mask = np.zeros(cluster_max + 1, dtype=np.bool)
    clusters_mask[clusters_to_update] = True
    
    # initialize the correlograms
    for (cl0, cl1) in product(clusters_to_update, clusters_to_update):
        delays[(cl0, cl1)] = []

    # loop through all spikes, across all neurons, all sorted
    for i in range(nspikes):
        t0, cl0 = spiketimes[i], clusters[i]
        # pass clusters that do not need to be processed
        if clusters_mask[cl0]:
            # i, t0, c0: current spike index, spike time, and cluster
            # boundaries of the second loop
            t0min, t0max = t0 - halfwidth, t0 + halfwidth
            j = i + 1
            # go forward in time up to the correlogram half-width
            # for j in range(i + 1, nspikes):
            while j < nspikes:
                t1, cl1 = spiketimes[j], clusters[j]
                # pass clusters that do not need to be processed
                if clusters_mask[cl1]:
                    # compute only correlograms if necessary
                    # and avoid computing symmetric pairs twice
                    # add the delay
                    if t0min <= t1 <= t0max:
                        d = t1 - t0
                        delays[(cl0, cl1)].append(d)
                        delays[(cl1, cl0)].append(-d)
                    else:
                        break
                j += 1
    
    return delays

def compute_correlograms(spiketimes, clusters, clusters_to_update=None,
    halfwidth=.01, bin=.001):

    # Compute all delays between any two close spikes.
    delays_pairs = compute_spike_delays(spiketimes, clusters,
                                  clusters_to_update=clusters_to_update,
                                  halfwidth=halfwidth)
    
    # Compute the histogram bins.
    n = int(np.ceil(halfwidth / bin))
    bins = np.arange(2 * n + 1) * bin - n * bin
    
    # Compute the histograms of the delays.
    correlograms = {}
    for (cl0, cl1), delays in delays_pairs.iteritems():
        h, _ = np.histogram(delays, bins=bins)
        h[(len(h) + 1) / 2] = 0
        correlograms[(cl0, cl1)] = h
        
    return correlograms
    
    
if __name__ == '__main__':
    
    
    filename = r"D:\Git\spiky\_test\data\test.clu.1"
    from spiky.io.loader import KlustersLoader
    l = KlustersLoader(filename)
    
    spiketimes = l.get_spiketimes()
    clusters = l.get_clusters()
    
    corr = compute_correlograms(spiketimes, clusters,
        halfwidth=.1, bin=.001)
    
    