"""This module implements the computation of the correlation matrix between
clusters."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import Counter

import numpy as np

from statstools import matrix_of_pairs


# -----------------------------------------------------------------------------
# Correlation matrix
# -----------------------------------------------------------------------------
def compute_statistics(Fet1, Fet2, spikes_in_clusters, masks):
    """Return Gaussian statistics about each cluster."""
    nPoints = Fet1.shape[0]
    nDims = Fet1.shape[1]
    nClusters = len(spikes_in_clusters)
    
    # precompute the mean and variances of the masked points for each feature
    # contains 1 when the corresponding point is masked
    masked = np.zeros_like(masks)
    masked[masks == 0] = 1
    nmasked = np.sum(masked, axis=0)
    nmasked[nmasked == 0] = 1
    nu = np.sum(Fet2 * masked, axis=0) / nmasked
    nu = nu.reshape((1, -1))
    sigma2 = np.sum(((Fet2 - nu) * masked) ** 2, axis=0) / nmasked
    sigma2 = sigma2.reshape((1, -1))
    # expected features
    y = Fet1 * masks + (1 - masks) * nu
    z = masks * Fet1**2 + (1 - masks) * (nu ** 2 + sigma2)
    eta = z - y ** 2
    # print nu
    # print nmasked
    
    LogP = np.zeros((nPoints, nClusters))
    stats = {}
    
    for c in xrange(nClusters):
        MyPoints = spikes_in_clusters[c]
        
        # print MyPoints
        
        # now, take the modified features here
        MyFet2 = np.take(y, MyPoints, axis=0)
        
        if len(MyPoints) > nDims:
            # log of the proportion in cluster c
            LogProp = np.log(len(MyPoints) / float(nPoints))
            Mean = np.mean(MyFet2, axis=0).reshape((1, -1))
            # stats for cluster c
            CovMat = np.cov(MyFet2, rowvar=0)
            
            # HACK: avoid instability issues, kind of works
            CovMat += np.diag(1e-3 * np.ones(nDims))
            
            # now, add the diagonal modification to the covariance matrix
            # the eta just for the current cluster
            etac = np.take(eta, MyPoints, axis=0)
            d = np.sum(etac, axis=0) / nmasked
            # add diagonal
            CovMat += np.diag(d)
            CovMatinv = np.linalg.inv(CovMat)
            LogDet = np.log(np.linalg.det(CovMat))
            
            stats[c] = (Mean, CovMat, CovMatinv, LogDet, len(MyPoints))

    return stats

def compute_correlations(features, clusters, masks,
        clusters_to_update=None):
    """Compute the correlation matrix between every pair of clusters.
    
    Use an approximation of the original Klusters grouping assistant, with
    an integral instead of a sum (integral of the product of the Gaussian 
    densities).
    
    A dictionary pairs => value is returned.
    
    """
    
    nPoints = features.shape[0]
    nDims = features.shape[1]
    
    c = Counter(clusters)
    spikes_in_clusters = [np.nonzero(clusters == clu)[0] for clu in sorted(c)]
    
    stats = compute_statistics(features, features, spikes_in_clusters, masks)
    
    # print stats
    
    if clusters_to_update is None:
        clusters_to_update = sorted(stats.keys())
    else:
        clusters_to_update = np.intersect1d(clusters_to_update, stats.keys())
    nClusters = len(clusters_to_update)

    matrix = {}
    
    coeff = np.log(2 * np.pi) * (-nDims / 2.)
    for i, ci in enumerate(clusters_to_update):
        mui, Ci, Ciinv, logdeti, npointsi = stats[ci]
        for k, cj in enumerate(clusters_to_update[i:]):
            j = i + k
            muj, Cj, Cjinv, logdetj, npointsj = stats[cj]
            dmu = (muj - mui).reshape((-1, 1))
            
            Csum = Ci + Cj
            Csuminv = np.linalg.inv(Csum)
            
            p = (coeff +
                (-.5 * np.log(np.linalg.det(Csum))) +
                (-.5) * np.dot(np.dot(dmu.T, Csuminv), dmu))
            alpha = float(npointsi) / nPoints
            # matrix[i, j] = p# + np.log(alpha)*
            expp = np.exp(p)[0,0]
            matrix[(ci, cj)] = expp
            matrix[(cj, ci)] = expp
    
    # # Symmetrize the matrix.
    # matrix = matrix + matrix.T
    
    # # Remove the diagonal and replace it with the minimum value.
    # matrix[range(nClusters), range(nClusters)] = 0
    # nonzero = matrix[matrix != 0]
    # if nonzero.size > 0:
        # matrix[matrix == 0] = nonzero.min()
    
    return matrix
    

# if __name__ == '__main__':
    
    # filename = r"D:\Git\spiky\_test\data\test.clu.1"
    # from spiky.io.loader import KlustersLoader
    # l = KlustersLoader(filename)
    
    # features = l.get_features()
    # clusters = l.get_clusters()
    # masks = l.get_masks(full=True)
    
    # C = compute_correlations(features, clusters, masks)
    
    # C = matrix_of_pairs(C)
    
    # from pylab import imshow, show
    
    # imshow(C, interpolation='none')
    # show()
    
    