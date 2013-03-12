
from collections import Counter
from itertools import product
from math import ceil
import numpy as np

def poisson(n):
    return np.cumsum(np.random.exponential(size=n, scale=.01))

@profile
def compute(spiketimes, clusters, clusters_to_update=None, freq=None, bin=.001, width=.02):

    # half-size of the histograms
    n = int(np.ceil(width / bin))
    
    # size of the histograms
    nspikes = len(spiketimes)
    
    # convert in seconds
    if freq:
        spiketimes = spiketimes / float(freq)

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
            t0min, t0max = t0 - width, t0 + width
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
                        delays[(cl0, cl1)].append(t1 - t0)
                    else:
                        break
                j += 1
    
    return delays

nspikes = 200000
nclusters = 200
fulltrain = poisson(nspikes)
clusters = np.random.randint(low=0, high=nclusters, size=nspikes)

compute(fulltrain, clusters)
