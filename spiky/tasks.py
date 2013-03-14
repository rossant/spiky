import os
import re
from galry import *
import tools
from collections import OrderedDict
import numpy as np
from colors import COLORMAP
from collections import Counter
from itertools import product
from copy import deepcopy as dcopy
import numpy.random as rdn
import inspect
import spiky.signals as ssignals
import spiky
import qtools
from qtools import inthread#, TasksInProcess
# import spiky.views as sviews
# import spiky.dataio as sdataio
import rcicons


class Tasks(object):
    """Singleton class storing all tasks, so that they can be all joined
    upon GUI closing."""
    def __init__(self):
        self.tasks = []
        
    def join(self):
        for task in self.tasks:
            task.join()
        
    def __setattr__(self, name, value):
        from threading import current_thread
        # print "adding", name, current_thread().ident
        super(Tasks, self).__setattr__(name, value)
        if isinstance(value, qtools.TasksBase):
            self.tasks.append(value)
        
        
TASKS = Tasks()
    


@inthread
class ClusterSelectionQueue(object):
    def __init__(self, du, dh):
        self.du = du
        self.dh = dh
        
    def select(self, clusters):
        self.dh.select_clusters(clusters)
        ssignals.emit(self.du, 'ClusterSelectionChanged', clusters)

        
def compute_correlograms(spiketimes, clusters, clusters_to_update=None, freq=None, bin=.001, width=.02):

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
                        d = t1 - t0
                        delays[(cl0, cl1)].append(d)
                        delays[(cl1, cl0)].append(-d)
                    else:
                        break
                j += 1
    
    return delays
    
        
# Correlograms
# ------------
# @inthread
# TODO: compute all correlograms in one pass
class ClusterCache(object):
    def __init__(self, dh, sdh, width=None, bin=None):
        self.dh = dh
        self.sdh = sdh
        
        if width is None:
            width = .2
        if bin is None:
            bin = .001
        
        self.width = width
        self.bin = bin
        self.histlen = 2 * int(np.ceil(width / bin))
        
        # cache correlograms, spike trains
        self.correlograms = {}
    
    # @profile
    def invalidate(self, clusters):
        # invalidate correlograms
        keys = dcopy(self.correlograms.keys())
        for cl0, cl1 in keys:
            if cl0 in clusters or cl1 in clusters:
                del self.correlograms[(cl0, cl1)]

    def process(self, clusters):
        """Compute or retrieve from the cache the spiketimes, spikecounts and
        correlograms of all the specified clusters. Populates the SDH with 
        the retrieved quantities."""
        
        nclusters = len(clusters)
        if nclusters == 0:
            return
        
        # SDH spiketimes and correlograms
        correlograms = []
        
        bin = self.bin
        width = self.width
        n = int(np.ceil(width / bin))
        bins = np.arange(2 * n + 1) * bin - n * bin
        
        # find clusters to update
        
        clusters_to_update = []
        # all requested clusters
        for (cl0, cl1) in product(clusters, clusters):
            if cl1 >= cl0:
                # only those clusters which are not already in the cache
                if (cl0, cl1) not in self.correlograms:
                    clusters_to_update.append(cl0)
                    clusters_to_update.append(cl1)
        clusters_to_update = list(set(clusters_to_update))
                    
        delays = compute_correlograms(self.dh.spiketimes,
            self.dh.clusters, clusters_to_update, freq=self.dh.freq,
            bin=bin, width=width)
        
        for (cl0, cl1) in product(clusters, clusters):
            if cl1 >= cl0:
                if (cl0, cl1) not in self.correlograms:
                    h, _ = np.histogram(delays[(cl0,cl1)], bins=bins)
                    h[len(h)/2] = 0
                    self.correlograms[(cl0,cl1)] = h
                correlograms.append(self.correlograms[(cl0,cl1)])
                
        # populate SDH
        # self.sdh.spiketimes = spiketimes
        self.sdh.correlograms = np.vstack(correlograms) * 1.
        
        self.update_signal()
    
    def update_signal(self):
        """Raise the update signal, specifying that the correlogram view
        needs to be updated."""
        ssignals.emit(self, 'CorrelogramsUpdated')
      

# Correlation matrix
# ------------------
def MGProbsOLD(Fet1, Fet2, spikes_in_clusters, masks):#Clu2):

    nPoints = Fet1.shape[0] #size(Fet1, 1)
    nDims = Fet1.shape[1] #size(Fet1, 2)
    # nClusters = Clu2.max() #max(Clu2)
    nClusters = len(spikes_in_clusters)

    LogP = np.zeros((nPoints, nClusters))
    for c in xrange(nClusters):
        # MyPoints = np.nonzero(Clu2==c)[0]
        MyPoints = spikes_in_clusters[c]
        MyFet2 = Fet2[MyPoints, :]
        if len(MyPoints) > nDims:
            LogProp = np.log(len(MyPoints) / float(nPoints)) # log of the proportion in cluster c
            Mean = np.mean(MyFet2, axis=0).reshape((1, -1))  #
            CovMat = np.cov(MyFet2, rowvar=0) # stats for cluster c
            
            # HACK: avoid instability issues, kind of works
            CovMat += np.diag(1e-3 * np.ones(nDims))
            
            
            LogDet = np.log(np.linalg.det(CovMat))   #

                
            # print c
            # print LogDet
            
            
            dx = Fet1 - Mean #repmat(Mean, nPoints, 1) # distance of each point from cluster
            # y = dx / CovMat
            # print Fet1.shape, Mean.shape, dx.shape, CovMat.shape
            y = np.linalg.solve(CovMat.T, dx.T).T
            LogP[:,c] = np.sum(y*dx, axis=1)/2. + LogDet/2. - LogProp + np.log(2*np.pi)*nDims/2. # -Log Likelihood
                # -log of joint probability that the point lies in cluster c and has given coords.
            
            # print LogP[:,c]
            # print
                
        else:
            LogP[:,c] = np.inf

    JointProb = np.exp(-LogP)

    # # if any points have all probs zero, set them to cluster 1
    JointProb[np.sum(JointProb, axis=1) == 0, 0] = 1e-9 #eps

    # #probability that point belongs to cluster, given coords
    # p = JointProb / repmat(sum(JointProb,2), 1, nClusters) 
    p = JointProb / np.sum(JointProb, axis=1).reshape((-1, 1))
    
    # print p
    # print
    
    return p

    
# take masks into account now
def MGProbs(Fet1, Fet2, spikes_in_clusters, masks):

    nPoints = Fet1.shape[0] #size(Fet1, 1)
    nDims = Fet1.shape[1] #size(Fet1, 2)
    # nClusters = Clu2.max() #max(Clu2)
    nClusters = len(spikes_in_clusters)
    
    # precompute the mean and variances of the masked points for each feature
    # contains 1 when the corresponding point is masked
    masked = np.zeros_like(masks)
    masked[masks == 0] = 1
    nmasked = np.sum(masked, axis=0)
    nu = np.sum(Fet2 * masked, axis=0) / nmasked
    nu = nu.reshape((1, -1))
    sigma2 = np.sum(((Fet2 - nu) * masked) ** 2, axis=0) / nmasked
    sigma2 = sigma2.reshape((1, -1))
    # expected features
    y = Fet1 * masks + (1 - masks) * nu
    z = masks * Fet1**2 + (1 - masks) * (nu ** 2 + sigma2)
    eta = z - y ** 2
    
    LogP = np.zeros((nPoints, nClusters))
    for c in xrange(nClusters):
        # MyPoints = np.nonzero(Clu2==c)[0]
        MyPoints = spikes_in_clusters[c]
        # MyFet2 = Fet2[MyPoints, :]
        # now, take the modified features here
        # MyFet2 = y[MyPoints, :]
        MyFet2 = np.take(y, MyPoints, axis=0)
        if len(MyPoints) > nDims:
            LogProp = np.log(len(MyPoints) / float(nPoints)) # log of the proportion in cluster c
            Mean = np.mean(MyFet2, axis=0).reshape((1, -1))
            CovMat = np.cov(MyFet2, rowvar=0) # stats for cluster c
            
            
            
            # HACK: avoid instability issues, kind of works
            CovMat += np.diag(1e-3 * np.ones(nDims))
            
            
            
            # now, add the diagonal modification to the covariance matrix
            # the eta just for the current cluster
            etac = np.take(eta, MyPoints, axis=0)
            d = np.sum(etac, axis=0) / nmasked
            # add diagonal
            CovMat += np.diag(d)
            
            
            
            # add diagonal terms to avoid singular matrices
            # CovMat += np.diag(sigma2.flatten())
            
            # need to compute the inverse to have the diagonal coefficients
            # TODO: optimize?
            CovMatinv = np.linalg.inv(CovMat)
            
            
            
            LogDet = np.log(np.linalg.det(CovMat))   #

            # dx = Fet1 - Mean #repmat(Mean, nPoints, 1) # distance of each point from cluster
            # we take the expected features
            dx = y - Mean #repmat(Mean, nPoints, 1) # distance of each point from cluster
            # y = dx / CovMat
            # print Fet1.shape, Mean.shape, dx.shape, CovMat.shape
            # TODO: we don't need that anymore if we compute the inverse of the cov matrix
            y2 = np.linalg.solve(CovMat.T, dx.T).T
            correction = np.sum(eta * np.diag(CovMatinv).reshape((1, -1)), axis=1)
            LogP[:,c] = (np.sum(y2*dx, axis=1)/2. + correction / 2.) + LogDet/2. - LogProp + np.log(2*np.pi)*nDims/2. # -Log Likelihood
                # -log of joint probability that the point lies in cluster c and has given coords.
                
            # print c
            # print LogP[:,c]
            # print
                
        else:
            LogP[:,c] = np.inf

    JointProb = np.exp(-LogP)

    # # if any points have all probs zero, set them to cluster 1
    JointProb[np.sum(JointProb, axis=1) == 0, 0] = 1e-9 #eps

    # #probability that point belongs to cluster, given coords
    # p = JointProb / repmat(sum(JointProb,2), 1, nClusters) 
    p = JointProb / np.sum(JointProb, axis=1).reshape((-1, 1))
    
    return p

    
    
def get_stats(Fet1, Fet2, spikes_in_clusters, masks):
    # Fet1, Fet2 = features, features
        
    nPoints = Fet1.shape[0] #size(Fet1, 1)
    nDims = Fet1.shape[1] #size(Fet1, 2)
    # nClusters = Clu2.max() #max(Clu2)
    nClusters = len(spikes_in_clusters)

    # precompute the mean and variances of the masked points for each feature
    # contains 1 when the corresponding point is masked
    masked = np.zeros_like(masks)
    masked[masks == 0] = 1
    nmasked = np.sum(masked, axis=0)
    nu = np.sum(Fet2 * masked, axis=0) / nmasked
    nu = nu.reshape((1, -1))
    sigma2 = np.sum(((Fet2 - nu) * masked) ** 2, axis=0) / nmasked
    sigma2 = sigma2.reshape((1, -1))
    # expected features
    y = Fet1 * masks + (1 - masks) * nu
    z = masks * Fet1**2 + (1 - masks) * (nu ** 2 + sigma2)
    eta = z - y ** 2

    LogP = np.zeros((nPoints, nClusters))

    stats = {}

    for c in xrange(nClusters):
        # MyPoints = np.nonzero(Clu2==c)[0]
        MyPoints = spikes_in_clusters[c]
        # MyFet2 = Fet2[MyPoints, :]
        # now, take the modified features here
        # MyFet2 = y[MyPoints, :]
        MyFet2 = np.take(y, MyPoints, axis=0)
        if len(MyPoints) > nDims:
            LogProp = np.log(len(MyPoints) / float(nPoints)) # log of the proportion in cluster c
            Mean = np.mean(MyFet2, axis=0).reshape((1, -1))
            CovMat = np.cov(MyFet2, rowvar=0) # stats for cluster c
            
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
    
    
    
    
    
def correlation_matrix(features, clusters, masks):
    # print masks.shape
    c = Counter(clusters)
    spikes_in_clusters = [np.nonzero(clusters == clu)[0] for clu in sorted(c)]
    nClusters = len(spikes_in_clusters)
    
    P = MGProbs(features, features, spikes_in_clusters, masks)
    ErrorMat = np.zeros((nClusters, nClusters))
    for c in xrange(nClusters):
        # MyPoints = np.nonzero(Clu2==c)[0]
        MyPoints = spikes_in_clusters[c]
        ErrorMat[c,:] = np.mean(P[MyPoints, :], axis=0)

    return ErrorMat
    
def correlation_matrix_KL(features, clusters, masks):
    
    nPoints = features.shape[0] #size(Fet1, 1)
    nDims = features.shape[1] #size(Fet1, 2)
    # nClusters = Clu2.max() #max(Clu2)
    # nClusters = len(spikes_in_clusters)
    
    # print masks.shape
    c = Counter(clusters)
    spikes_in_clusters = [np.nonzero(clusters == clu)[0] for clu in sorted(c)]
    nClusters = len(spikes_in_clusters)
    
    stats = get_stats(features, features, spikes_in_clusters, masks)
    
    clusterslist = sorted(stats.keys())
    matrix_KL = np.zeros((nClusters, nClusters))

    for ci in clusterslist:
        for cj in clusterslist:
            mui, Ci, Ciinv, logdeti, _ = stats[ci]
            muj, Cj, Cjinv, logdetj, _ = stats[cj]
            dmu = (muj - mui).reshape((-1, 1))
            
            # KL divergence
            dkl = .5 * (np.trace(np.dot(Cjinv, Ci)) + np.dot(np.dot(dmu.T, Cjinv), dmu) - logdeti + logdetj - nDims)
            
            matrix_KL[ci, cj] = dkl
    
    m, M = matrix_KL.min(), matrix_KL.max()
    matrix_KL = 1 - (matrix_KL - m) / (M - m)
    matrix_KL[matrix_KL == 1] = 0
    return matrix_KL
    
     
def correlation_matrix2(features, clusters, masks):
    # print features
    # print clusters
    # print masks
    
    nPoints = features.shape[0] #size(Fet1, 1)
    nDims = features.shape[1] #size(Fet1, 2)
    # nClusters = Clu2.max() #max(Clu2)
    # nClusters = len(spikes_in_clusters)
    
    # print masks.shape
    c = Counter(clusters)
    spikes_in_clusters = [np.nonzero(clusters == clu)[0] for clu in sorted(c)]
    nClusters = len(spikes_in_clusters)
    
    stats = get_stats(features, features, spikes_in_clusters, masks)
    
    clusterslist = sorted(stats.keys())
    matrix = np.zeros((nClusters, nClusters))

    for i, ci in enumerate(clusterslist):
        mui, Ci, Ciinv, logdeti, npointsi = stats[ci]
        for j, cj in enumerate(clusterslist):
            muj, Cj, Cjinv, logdetj, npointsj = stats[cj]
            dmu = (muj - mui).reshape((-1, 1))
            
            p = np.log(2*np.pi)*(-nDims/2.)+(-.5*np.log(np.linalg.det(Ci+Cj)))+(-.5)*np.dot(np.dot(dmu.T, np.linalg.inv(Ci+Cj)), dmu)
            alpha = float(npointsi) / nPoints
            # print ci, cj, p
            matrix[i, j] = p# + np.log(alpha)
    
    matrix[range(nClusters), range(nClusters)] = 0
    matrix[matrix==0] = matrix[matrix!=0].min()
    
    return np.exp(matrix)
    
    
class CorrelationMatrixQueue(object):
    def __init__(self, dh):
        self.dh = dh
        
    def process(self):
        correlation_matrix = correlation_matrix2(
            self.dh.features[:,:-self.dh.nextrafet], self.dh.clusters,
            self.dh.masks_complete[:,:-self.dh.nextrafet])
        return correlation_matrix
        
    @staticmethod
    def process_done(_result=None):
        ssignals.emit(None, 'CorrelationMatrixUpdated', _result)
        
        
        