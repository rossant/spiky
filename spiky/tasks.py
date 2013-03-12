import os
import re
from galry import *
import tools
from collections import OrderedDict
import numpy as np
from colors import COLORMAP
from collections import Counter
from copy import deepcopy as dcopy
import numpy.random as rdn
import inspect
import spiky.signals as ssignals
import spiky
import qtools
from qtools import inthread, inprocess
from qtools.qtpy.QtCore import QObject
# import spiky.views as sviews
# import spiky.dataio as sdataio
import rcicons


class Tasks(object):
    """Singleton class storing all tasks, so that they can be all joined
    upon GUI closing."""
    def __init__(self):
        self.tasks = {}
        
    def join(self):
        for name, task in self.tasks.iteritems():
            # print "join", name
            task.join()
    
    def add(self, name, value):
        self.tasks[name] = value
        
    def __setattr__(self, name, value):
        # from threading import current_thread
        # print "adding", name, current_thread().ident
        super(Tasks, self).__setattr__(name, value)
        if isinstance(value, qtools.TasksBase):
            # print name
            self.add(name, value)
    
    def __getattr__(self, name):
        return self.tasks.get(name)
        
        
TASKS = Tasks()
    


@inthread
class ClusterSelectionQueue(object):
    def __init__(self, du, dh):
        self.du = du
        self.dh = dh
        
    def select(self, clusters):
        self.dh.select_clusters(clusters)
        ssignals.emit(self.du, 'ClusterSelectionChanged', clusters)

    
        
# Correlograms
# ------------

def compute_correlograms(spiketimes, clusters,
        pairs,
        freq=None, bin=.001, width=.2):
    """Compute all pairwise cross-correlograms between neurons.
    
    """
    counter = Counter(clusters)
    # sorted list of all cluster unique indices
    clusters_unique = sorted(counter.keys())
    # list of all pairs (ci <= cj)
    # if pairs is None:
        # pairs = [(ci, cj) for ci in clusters_unique for cj in clusters_unique if cj >= ci]
    
    # half-size of the histograms
    n = int(np.ceil(width / bin))
    
    # size of the histograms
    bins = np.arange(2 * n + 1) * bin - n * bin
    nspikes = len(spiketimes)
    
    # convert in seconds
    if freq:
        spiketimes = spiketimes / float(freq)

    # delays will contain all delays for each pair of clusters
    delays = {}
    # correlograms will contain all requested correlograms
    correlograms = []
    
    # get the pairs for which the correlograms need to be computed
    clusters_to_update = [cl0 for (cl0, _) in pairs]
    
    # initialize the correlograms
    for (cl0, cl1) in pairs:
        delays[(cl0, cl1)] = []
        
    # loop through all spikes, across all neurons, all sorted
    for i in range(nspikes):
        t0, cl0 = spiketimes[i], clusters[i]
        # pass clusters that do not need to be processed
        if cl0 not in clusters_to_update:
            continue
        # i, t0, c0: current spike index, spike time, and cluster
        inner_list = [xrange(i+1, nspikes),
                      xrange(i-1, -1, -1)]
        # boundaries of the second loop
        t0min, t0max = t0 - width, t0 + width
        # go forward and backtime in time up to the correlogram half-width
        for inner in inner_list:
            for j in inner:
                t1, cl1 = spiketimes[j], clusters[j]
                # pass clusters that do not need to be processed
                if (cl0, cl1) not in pairs:
                    continue
                # compute only correlograms if necessary
                # and avoid computing symmetric pairs twice
                # add the delay
                if t0min <= t1 <= t0max:
                    delays[(cl0, cl1)].append(t1 - t0)
                else:
                    break
    # compute the histograms of all delays, for each pair of clusters
    for (cl0, cl1) in pairs:
        h, _ = np.histogram(delays[(cl0, cl1)], bins=bins)
        h[len(h) / 2] = 0
        correlograms.append(h)
    
    return correlograms

      

class CorrelogramsQueue(object):
    def compute_correlograms(self, *args, **kwargs):
        return compute_correlograms(*args, **kwargs)

    @staticmethod
    def compute_correlograms_done(self, *args, **kwargs):
        """Raise the update signal, specifying that the correlogram view
        needs to be updated."""
        result = kwargs.get('_result')
        if result is not None and len(result) > 0:
            correlograms = np.vstack(result) * 1.
            ssignals.emit(self, 'CorrelogramsUpdated', correlograms)
        

class CorrelogramsManager(object):
    def __init__(self, dh, sdh, width=None, bin=None):
        # super(CorrelogramsManager, self).__init__(parent)
        
        self.dh = dh
        self.sdh = sdh
        
        if width is None:
            width = .2
        if bin is None:
            bin = .001
        
        self.width = width
        self.bin = bin
        self.histlen = 2 * int(np.ceil(width / bin))
        
        # cache correlograms
        self.correlograms = {}
        self.spiketimes = dh.spiketimes
        self.clusters = dh.clusters
        self.freq = dh.freq
    
        self.task = inprocess(CorrelogramsQueue)()
        
        
        counter = Counter(self.clusters)
        # sorted list of all cluster unique indices
        clusters_unique = sorted(counter.keys())
        pairs = [(ci, cj) for ci in clusters_unique for cj in clusters_unique if cj >= ci]
        
        # compute all
        self.task.compute_correlograms(
            self.spiketimes,
            self.clusters,
            pairs,
            freq=self.freq, bin=self.bin, width=self.width)
    
        # ssignals.SIGNALS.CorrelogramsComputed.connect(self.slotCorrelogramsComputed, QtCore.Qt.UniqueConnection)
    
    # def slotCorrelogramsComputed(self, pairs, correlograms):
        # for (cl0, cl1), corr in zip(pairs, correlograms):
            # self.correlograms[(cl0, cl1)] = corr
        # ssignals.emit(self, 'CorrelogramsUpdated')
    
    def invalidate(self, clusters):
        pass

    def process(self, clusters):
        if hasattr(self.dh, 'correlograms'):
            print self.dh.correlograms.shape
        # k = len(clusters)
        # pairs = [(clusters[i], clusters[j]) for i in xrange(k) for j in xrange(i+1,k)]
        # pairs = list(set(pairs) - set(self.correlograms.keys()))
        # print clusters, pairs
        # self.task.compute_correlograms(
            # self.spiketimes,
            # self.clusters,
            # pairs=pairs,
            # freq=self.freq, bin=self.bin, width=self.width)
        
    def join(self):
        # print "join"
        self.task.join()
    

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
            
            stats[c] = (Mean, CovMat, CovMatinv, LogDet)

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
            mui, Ci, Ciinv, logdeti = stats[ci]
            muj, Cj, Cjinv, logdetj = stats[cj]
            dmu = (muj - mui).reshape((-1, 1))
            
            # KL divergence
            dkl = .5 * (np.trace(np.dot(Cjinv, Ci)) + np.dot(np.dot(dmu.T, Cjinv), dmu) - logdeti + logdetj - nDims)
            
            matrix_KL[ci, cj] = dkl
    
    m, M = matrix_KL.min(), matrix_KL.max()
    matrix_KL = 1 - (matrix_KL - m) / (M - m)
    matrix_KL[matrix_KL == 1] = 0
    return matrix_KL
    
    
# @inprocess
class CorrelationMatrixQueue(object):
    def __init__(self, dh):
        self.dh = dh
        
    def process(self):
        # print "processing...",
        correlation_matrix = correlation_matrix_KL(
            self.dh.features, self.dh.clusters, self.dh.masks_complete)
        # import time
        # time.sleep(5)
        # ssignals.emit(self, 'CorrelationMatrixUpdated')
        # print "ok"
        return correlation_matrix
        
    @staticmethod
    def process_done(_result=None):
        # print "processing done, emitting signal...",
        ssignals.emit(None, 'CorrelationMatrixUpdated', _result)
        # print "ok"
        
        
        
        