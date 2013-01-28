import numpy as np
from collections import Counter

# pi(x) = wi * N(x | mui, Sigmai)
# a_ij = 1/Nj sum_Cj pi(x)
# c_ij = a_ij / sum_k a_kj

def correlation_matrix(features, spikes_in_clusters):
    """
    spikes_in_clusters: a list with, for each cluster_rel, the list of
        spike relative indices
    """
    # ensure float dtype
    features = np.asarray(features, dtype=np.float32)
    nclusters = len(spikes_in_clusters)
    
    # weights
    w = np.array([len(ind) for ind in spikes_in_clusters], dtype=np.float32)
    
    # for each cluster: mean and correlation matrix
    nspikes, ndim = features.shape
    
    # M: nclusters x ndim: mean of each cluster
    # C: nclusters x ndim x ndim: for each cluster, the inverse correlation matrix
    # D: nclusters: the det of the covariance matrix for each cluster
    M = np.zeros((nclusters, ndim))
    C = np.zeros((nclusters, ndim, ndim))
    D = np.zeros(nclusters)
    
    # compute the mean and covariance matrices of each cluster
    for cluster in xrange(nclusters):
        fet = np.take(features, spikes_in_clusters[cluster], axis=0)
        M[cluster, :] = fet.mean(axis=0)
        cov = np.cov(fet, rowvar=0)
        covinv = np.linalg.inv(cov)
        C[cluster, ...] = covinv
        D[cluster] = np.linalg.det(cov)
    
    # P: nspikes x nclusters: for each spike, the weighted proba it belongs 
    # to any cluster
    # D = D.reshape((1, -1, 1, 1))
    # coeff = (2 * np.pi) ** (-ndim * .5) * D ** (-.5)
    # X = features.reshape((nspikes, 1, ndim, 1))
    # mu = M.reshape((1, nclusters, ndim, 1))
    # covinv = C.reshape((1, nclusters, ndim, ndim))
    # U = X - mu
    # P = covinv * U * np.swapaxes(U, 2, 3)
    # P = P.sum(axis=-1).sum(axis=-1)
    # P = np.exp(-.5 * P)
    # P = coeff * P
    
    P = np.zeros((nspikes, nclusters))
    
    # compute the pdf
    D = D.reshape((1, -1))
    coeff = (2 * np.pi) ** (-ndim * .5) * D ** (.5)
    X = features.reshape((nspikes, 1, ndim))
    mu = M.reshape((1, nclusters, ndim))
    U = X - mu
    P = np.zeros((nspikes, nclusters))
    for i in xrange(ndim):
        for j in xrange(ndim):
            P += U[:,:,i] * U[:,:,j] * C[:,i,j].reshape((1, -1))
    P2 = np.exp(-.5 * P)
    P3 = coeff * P2
    
    # A: nclusters x nclusters: like P, but averaged cluster-wise vertically
    A = np.zeros((nclusters, nclusters))
    for cluster in xrange(nclusters):
        A[cluster, :] = np.take(P, spikes_in_clusters[cluster], axis=0).mean(axis=0)
    
    # C: like A, but divided by the sum on each row
    N = A.sum(axis=1).reshape((-1, 1))
    C = A / N
    
    return C


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    features = np.loadtxt("../_test/data/test.fet.1", np.int32, skiprows=1)
    features = features[:, :-1]
    
    features = np.array(features, dtype=np.float32)
    features /= features.max()
    
    clusters = np.loadtxt("../_test/data/test.clu.1", np.int32, skiprows=1)
    
    c = Counter(clusters)
    spikes_in_clusters = [np.nonzero(clusters == clu)[0] for clu in sorted(c)]
    
    # print features.shape, clusters.shape
    # print spikes_in_clusters 
    # print sorted(c)
    
    C = correlation_matrix(features, spikes_in_clusters)
    
    # print C
    
    plt.imshow(C, interpolation='none')
    
    plt.show()
    
    
    