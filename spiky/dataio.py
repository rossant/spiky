import collections
import threading
import os.path
import numpy as np
import numpy.random as rdn
from copy import deepcopy as dcopy
import h5py
from galry import *
from colors import COLORMAP
import signals
from xmltools import parse_xml
from spiky.qtqueue import qtjobqueue
from collections import Counter
import cPickle
# from tools import Info

__all__ = [
    'get_clusters_info',
    'DataHolder',
    'SelectDataHolder',
    'DataProvider',
    'SpikeDetektH5DataProvider',
    'KlustersDataProvider',
    'H5DataProvider',
    'MockDataProvider',
    ]



# Text loading
# ------------
def load_text(file, dtype, skiprows=0):
    return np.loadtxt(file, dtype=dtype, skiprows=skiprows)

def load_text_fast(filename, dtype, skiprows=0, delimiter=' '):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            first = True
            skip = 0
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    if item:
                        yield dtype(item)
                    else:
                        skip += 1
                if first:
                    load_text_fast.rowlength = len(line) - skip
                    # print load_text_fast.rowlength
                first = False
    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, load_text_fast.rowlength))
    # print data.shape
    return data

try:
    import pandas as pd
    # make sure that read_csv is available
    assert hasattr(pd, 'read_csv')
    
    def load_text_pandas(filename, dtype, skiprows=0, delimiter=' '):
        with open(filename, 'r') as f:
            for _ in xrange(skiprows):
                f.readline()
            x = pd.read_csv(f, header=None, sep=delimiter).values.astype(dtype).squeeze()
        # return pd.read_csv(filename, skiprows=skiprows, sep=delimiter).values.astype(dtype)
        # print x.shape
        return x
    
except (ImportError, AssertionError):
    log_warn("You'd better have Pandas v>=0.10")
    load_text_pandas = load_text
    
def save_text(file, data):
    return np.savetxt(file, data, fmt='%d', newline='\n')
        
def load_binary(file, dtype=None, count=None):
    if dtype is None:
        dtype = np.dtype(np.int16)
    if count is None:
        X = np.fromfile(file, dtype=dtype)
    else:
        X = np.fromfile(file, dtype=dtype, count=count)
    return X

def get_clusters_info(clusters, groupidx=0):
    spkcounts = collections.Counter(clusters)
    cluster_keys = sorted(spkcounts.keys())
    nclusters = len(cluster_keys)
    clusters_info = {}
    for clusteridx in cluster_keys:
        spkcount = spkcounts[clusteridx]
        clusters_info[clusteridx] = {
                'clusteridx': clusteridx,
                'color': np.mod(clusteridx, len(COLORMAP)),
                'spkcount': spkcount,
                'groupidx': groupidx,
            }
    return clusters_info

    
def save_pickle(file, obj):
    with open(file, 'wb') as f:
        cPickle.dump(obj, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        obj = cPickle.load(f)
    return obj
    
    
# Correlograms
# ------------
@qtjobqueue
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
        self.spiketimes = {}
    
    # @profile
    def invalidate(self, clusters):
        # invalidate spike times
        keys = dcopy(self.spiketimes.keys())
        for cl in keys:
            if cl in clusters:
                del self.spiketimes[cl]
                
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
        spiketimes = []
        correlograms = []
        
        # process spiketimes
        for cluster in clusters:
            # retrieve or compute spiketimes
            if cluster not in self.spiketimes:
                spikes = self.dh.spiketimes[self.dh.clusters == cluster]
                self.spiketimes[cluster] = spikes
            else:
                spikes = self.spiketimes[cluster]
            spiketimes.append(spikes)
            
        # print "hey"
            
        # OLD, SLOW VERSION
        # # process correlograms
        # for i in xrange(len(clusters)):
            # # first spike train
            # spk0 = spiketimes[i] / float(self.dh.freq)
            # for j in xrange(i, len(clusters)):
                # # second spike train
                # spk1 = spiketimes[j] / float(self.dh.freq)
                
                # cl0 = clusters[i]
                # cl1 = clusters[j]
                
                # # compute correlogram
                # if (cl0, cl1) not in self.correlograms:
                    # corr = get_correlogram(spk0, spk1, width=self.width, bin=self.bin,
                        # duration=self.dh.duration)
                    # self.correlograms[(cl0, cl1)] = corr
                # else:
                    # corr = self.correlograms[(cl0, cl1)]
                # correlograms.append(corr)
               
        # NEW, OPTIMIZED VERSION 
        bin = self.bin
        width = self.width
        n = int(np.ceil(width / bin))
        bins = np.arange(2 * n + 1) * bin - n * bin
        
        clusters_unique = clusters
        fulltrain = self.dh.spiketimes / float(self.dh.freq)
        clusters = self.dh.clusters
        nspikes = self.dh.nspikes
        
        # cluster pairs that are requested
        requested_pairs = [(clusters_unique[i], clusters_unique[j]) for i in xrange(nclusters) for j in xrange(i, nclusters)]
        # cluster pairs which do not need to be computed again
        existing_pairs = self.correlograms.keys()
        updating_pairs = set(requested_pairs) - set(existing_pairs)
        # list of clusters to explore for cl0
        clusters_to_update1 = [cl0 for (cl0, _) in updating_pairs]
        
        # corr will contain all delays for each pair of clusters
        corr = {}
        # initialize the correlograms
        for (cl0, cl1) in requested_pairs:
            corr[(cl0, cl1)] = []
        # loop through all spikes, across all neurons, all sorted
        for i in xrange(nspikes):
            # current spike and cluster
            t0 = fulltrain[i]
            cl0 = clusters[i]
            if cl0 not in clusters_to_update1:
                continue
            # go forward in time up to the correlogram half-width
            for j in xrange(i+1, nspikes):
                # next spike and cluster
                t1 = fulltrain[j]
                cl1 = clusters[j]
                if (cl0, cl1) not in updating_pairs:
                    continue
                # compute only correlograms if necessary
                # and avoid computing symmetric pairs twice
                # add the delay
                if t1 <= t0 + width:
                    # if t1 != t0:
                    corr[(cl0, cl1)].append(t1 - t0)
                else:
                    break
            # go backward in time up to the correlogram half-width
            for j in xrange(i-1, -1, -1):
                t1 = fulltrain[j]
                cl1 = clusters[j]
                if (cl0, cl1) not in updating_pairs:
                    continue
                # compute only correlograms if necessary
                # and avoid computing symmetric pairs twice
                if t1 >= t0 - width:
                    # if t1 != t0:
                    corr[(cl0,cl1)].append(t1 - t0)
                else:
                    break
        # compute the histograms of all delays, for each pair of clusters
        # print bins
        for (cl0, cl1) in requested_pairs:
            if (cl0, cl1) not in self.correlograms:
                h, _ = np.histogram(corr[(cl0,cl1)], bins=bins)
                h[len(h)/2] = 0
                self.correlograms[(cl0,cl1)] = h
            correlograms.append(self.correlograms[(cl0,cl1)])
                
        # populate SDH
        self.sdh.spiketimes = spiketimes
        self.sdh.correlograms = np.vstack(correlograms) * 1.
        
        self.update_signal()
    
    def update_signal(self):
        """Raise the update signal, specifying that the correlogram view
        needs to be updated."""
        signals.emit(self, 'CorrelogramsUpdated')
      

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
    

@qtjobqueue
class CorrelationMatrixQueue(object):
    def __init__(self, dh):
        self.dh = dh
        
    def process(self):
        self.dh.correlation_matrix = correlation_matrix(
            self.dh.features, self.dh.clusters, self.dh.masks_complete)
        # import time
        # time.sleep(5)
        signals.emit(self, 'CorrelationMatrixUpdated')
        
        
# Data holder
# -----------
class DataHolder(object):
    """This class holds all the data related to a spike sorting session.
    Some variables may not be in-memory arrays, but rather HDF5 proxies.
    Actual data from selected clusters can be obtained through
    SelectDataHolder.

    List of variables:
    
        freq: a float with the sampling frequency
        nchannels: number of channels in the probe
        nspikes: total number of spikes
        probe: a Probe dic
        fetdim: number of features per channel
        total_duration: total duration, in samples count, of the current dataset
        current_window: a tuple with the interval, in samples count, of the current window
        spiketimes: an array with the spike times of the spikes, in samples count
        waveforms: a nspikes*nsamples*nchannels array with all waveforms
        waveforms_info: a dict with the info about the waveforms
        clusters: an array with the cluster index for each spike
        clusters_info: a ClustersInfo dic
        correlograms: a (nclusters * (nclusters + 1) / 2) x nsamples_correlograms matrix
        correlograms_info: Info(nsamples_correlograms)
        correlationmatrix: a nclusters * nclusters matrix
        features: a nspikes*nchannels*fetdim array with the features of each spike, in each channel
        masks: a nspikes array with the mask for each spike, as a float in [0,1]
        raw_trace: a total_duration*nchannels array with the raw trace (or a HDF5 proxy with the same interface)
        filtered_trace: like raw trace, but with the filtered trace
        filter_info: a FilterInfo dic
    """
    def __init__(self):
        self.correlation_matrix = -np.ones((2, 2))
    
    def new_cluster(self):
        ids = self.clusters_info['clusters_info'].keys()
        return max(ids) + 1

class SelectDataHolder(object):
    """Provides access to the data related to selected clusters."""
    def __init__(self, dataholder):
        self.dataholder = dataholder
        self.override_color = False
        self.clustercache = ClusterCache(dataholder, self, impatient=True)
        self.spike_dependent_variables = [
            'spiketimes',
            'waveforms',
            'clusters',
            'features',
            'masks',
            ]
        self.select_clusters([])
        
    def invalidate(self, clusters):
        self.clustercache.invalidate(clusters)
        
    # @profile
    def select_clusters(self, clusters):
        """Provides the data related to the specified clusters."""
        
        # keep the order of the selected clusters in clusters_ordered
        clusters = np.array(clusters)
        clusters_ordered = list(clusters.copy())
        clusters.sort()
        
        select_mask = np.in1d(self.dataholder.clusters, clusters)
        # spike rel to spike abs
        self.spike_ids = np.nonzero(select_mask)[0]
        # spike abs to spike rel
        if len(self.spike_ids) > 0:
            self.spikes_rel = np.empty(self.spike_ids.max() + 1, dtype=np.int32)
            self.spikes_rel[self.spike_ids] = np.arange(len(self.spike_ids), dtype=np.int32)
        else:
            self.spikes_rel = np.array([], dtype=np.int32)
        
        # nspikes is the number of True elements in select_mask
        self.nspikes = select_mask.sum()
        
        # process correlograms, and a signal is emitted when they are ready
        self.clustercache.process(clusters)
        
        self.correlograms = np.array([[]])
        self.baselines = np.array([])
        
        # self.baselines = counts / float(self.dataholder.duration)
        
        self.nclusters = len(clusters)
        # cluster colors
        self.cluster_colors_original = np.array([self.dataholder.clusters_info['clusters_info'][cluster]['color'] for cluster in clusters])
        # list of group indices for each cluster
        groups = np.array([self.dataholder.clusters_info['clusters_info'][cluster]['groupidx'] for cluster in clusters])
        # cluster overriden colors: the color of the corresponding group
        self.cluster_overriden_colors = np.array([self.dataholder.clusters_info['groups_info'][groupidx]['color'] for groupidx in groups])
        # handle -1 color when 
        # self.cluster_overriden_colors[self.cluster_overriden_colors < 0] == 1
        # unique clusters
        self.clusters_unique = clusters
        
        # the clusters relative indices in the right order for the depth
        if self.nclusters > 0:
            # self.clusters_ordered = np.digitize(self.clusters_ordered, self.clusters_unique) - 1
            # self.clusters_ordered = np.array(self.nclusters, dtype=np.int32)
            self.clusters_ordered = np.array([clusters_ordered.index(cluster) for cluster in self.clusters_unique])
        else:
            self.clusters_ordered = np.array([])
        
        
        # override all spike dependent variables, where the first axis
        # is the spike index
        if select_mask.size > 0:
            for varname in self.spike_dependent_variables:
                if hasattr(self.dataholder, varname):
                    # val = getattr(self.dataholder, varname)[select_mask,...]
                    val = np.compress(select_mask, getattr(self.dataholder, varname), axis=0)
                    setattr(self, varname, val)
    
    @property
    def cluster_colors(self):
        if self.override_color:
            return self.cluster_overriden_colors
        else:
            return self.cluster_colors_original
        
    def get_clusters(self):
        return self.clusters_unique
        
    def __getattr__(self, name):
        """Fallback mechanism for selecting variables in data holder and that
        have not been overloaded in SelectDataHolder."""
        return getattr(self.dataholder, name)
        

# Data providers
# --------------
class DataProvider(object):
    """Provide import/export functions to load/save a DataHolder instance."""
    data_holder = None
    
    def load(self, filename):
        pass
        return self.data_holder
        
    def save(self, filename):
        pass


        
def get_actual_filename(filename, extension, fileindex=1):
    """Search the most plausible existing filename corresponding to the
    requested approximate filename, which has the required file index and
    extension."""
    dir = os.path.dirname(filename)
    filename = os.path.basename(filename)
    files = os.listdir(dir)
    prefix = filename
    if fileindex is None:
        suffix = '.{0:s}'.format(extension)
    else:
        suffix = '.{0:s}.{1:d}'.format(extension, fileindex)
    filtered = []
    # find the real filename with the longest path that fits the requested
    # filename
    while prefix and not filtered:
        filtered = filter(lambda file: (file.startswith(prefix) and 
            file.endswith(suffix)), files)
        prefix = prefix[:-1]
    # order by increasing length and return the shortest
    filtered = sorted(filtered, cmp=lambda k, v: len(k) - len(v))
    return os.path.join(dir, filtered[0])
    # print os.path.commonprefix(files + [filename])
        
        
class KlustersDataProvider(DataProvider):
    """Legacy Klusters data provider with the old format."""
    # def load_probe(self, filename):
        # pass
        
    def load(self, filename, fileindex=1, probefile=None):#, progressbar=None):
        
        # load XML
        self.holder = DataHolder()
        
        try:
            path = get_actual_filename(filename, 'xml', None)
            params = parse_xml(path, fileindex=fileindex)
        except Exception as e:
            raise Exception(("The XML file was not found and the data cannot "
                "be loaded."))
        
        
        # klusters tests
        nchannels = params['nchannels']
        nsamples = params['nsamples']
        fetdim = params['fetdim']
        freq = params['rate']
        
        self.filename = filename
        self.fileindex = fileindex
        
        # if filename.endswith('_spiky'):
            # filename = filename.replace('_spiky', '')
            # spiky = True
        # else:
            # spiky = False
        
        
        # CLUSTERS
        # -------------------------------------------------
        try:
            # if spiky:
                # path = filename + "_spiky.clu.%d" % fileindex
            # else:
                # path = filename + ".clu.%d" % fileindex
            path = get_actual_filename(filename, 'clu', fileindex)
            # clusters = load_text(path, np.int32)
            clusters = load_text_pandas(path, np.int32)
            nspikes = len(clusters) - 1
        except Exception as e:
            log_warn("CLU file '%s' not found" % filename)
            clusters = np.zeros(nspikes + 1, dtype=np.int32)
            clusters[0] = 1
        # nclusters = clusters[0]
        clusters = clusters[1:]
        # if progressbar:
            # progressbar.setValue(1)
        signals.emit(self, 'FileLoading', .2)
        
        
        # FEATURES
        # -------------------------------------------------
        # features = load_text_fast(filename + ".fet.%d" % fileindex, np.int32, skiprows=1)
        path = get_actual_filename(filename, 'fet', fileindex)
        features = load_text_pandas(path, np.int32, skiprows=1)
        features = np.array(features, dtype=np.float32)
        
        # HACK: there are either 1 or 5 dimensions more than fetdim*nchannels
        # we can't be sure so we first try 1, if it does not work we try 5
        try:
            features = features.reshape((-1, fetdim * nchannels + 1))
        except:
            log_debug("The number of columns is not fetdim (%d) x nchannels (%d) + 1." \
                % (fetdim, nchannels))
            try:
                features = features.reshape((-1, fetdim * nchannels + 5))
                
            except:
                log_debug("The number of columns is not fetdim (%d) x nchannels (%d) + 5, so I'm confused and I can't continue. Sorry :(" \
                    % (fetdim, nchannels))
            
        
        # get the spiketimes
        spiketimes = features[:,-1].copy()
        # remove the last column in features, containing the spiketimes
        # features = features[:,:nchannels * fetdim]
        nextrafet = features.shape[1] - nchannels * fetdim
        
        # normalize the data here
        m = features[:,:-nextrafet].min()
        M = features[:,:-nextrafet].max()
        # force symmetry
        vx = max(np.abs(m), np.abs(M))
        m, M = -vx, vx
        features[:,:-nextrafet] = -1+2*(features[:,:-nextrafet]-m)/(M-m)
        
        
        # normalize the data here
        m = features[:,-nextrafet:].min()
        M = features[:,-nextrafet:].max()
        # # force symmetry
        # vx = max(np.abs(m), np.abs(M))
        # m, M = -vx, vx
        features[:,-nextrafet:] = -1+2*(features[:,-nextrafet:]-m)/(M-m)
        
        # if progressbar:
            # progressbar.setValue(2)
        signals.emit(self, 'FileLoading', .4)
            
            
        
        # MASKS
        # -------------------------------------------------
        # first: try fmask
        try:
            # masks = load_text(filename + ".fmask.%d" % fileindex, np.float32, skiprows=1)
            path = get_actual_filename(filename, 'fmask', fileindex)
            masks = load_text_pandas(path, np.float32, skiprows=1)
            self.holder.masks_complete = masks
            masks = masks[:,:-1:fetdim]
            # masks = masks[::fetdim]
        except Exception as e:
            try:
                # otherwise, try mask
                # masks = load_text(filename + ".mask.%d" % fileindex, np.float32, skiprows=1)
                path = get_actual_filename(filename, 'mask', fileindex)
                masks = load_text_pandas(path, np.float32, skiprows=1)
                # masks = masks[:,:-1:fetdim]
                self.holder.masks_complete = masks
                masks = masks[:,:-1:fetdim]
                # masks = masks[::fetdim]
            except:
                # finally, warning and default masks (everything to 1)
                log_warn("MASK file '%s' not found" % filename)
                masks = np.ones((nspikes, nchannels))
                self.holder.masks_complete = np.ones(features.shape)
        
        # if progressbar:
            # progressbar.setValue(3)
        signals.emit(self, 'FileLoading', .6)
        
        
        
        # WAVEFORMS
        # -------------------------------------------------
        try:
            path = get_actual_filename(filename, 'spk', fileindex)
            waveforms = load_binary(path)
            waveforms = waveforms.reshape((nspikes, nsamples, nchannels))
        except IOError as e:
            log_warn("SPK file '%s' not found" % filename)
            waveforms = np.zeros((nspikes, nsamples, nchannels))
        
        # if progressbar:
            # progressbar.setValue(4)
        signals.emit(self, 'FileLoading', .8)
            
            
            
        
        self.holder.freq = freq
        
        
        self.holder.nspikes = nspikes
        self.holder.nchannels = nchannels
        self.holder.nextrafet = nextrafet
        
        # construct spike times from random interspike interval
        self.holder.spiketimes = spiketimes
        
        self.holder.duration = spiketimes[-1] / float(self.holder.freq)
    
        # normalize waveforms at once
        waveforms = (waveforms - waveforms.mean())
        waveforms = waveforms / np.abs(waveforms).max()
        
        self.holder.waveforms = waveforms
        self.holder.waveforms_info = dict(nsamples=nsamples)
        
        self.holder.fetdim = fetdim
        self.holder.features = features
        
        self.holder.masks = masks
        
        self.holder.clusters = clusters
        
        # create the groups info object
        # Default groups
        
        # GROUPS
        # --------------------------------------
        try:
            path = get_actual_filename(filename, 'groups', fileindex)
        
            info = load_pickle(path)
            clusters_info = info['clusters_info']
            groups_info = info['groups_info']
            
        except:
        
            groups_info = {
                0: dict(groupidx=0, name='Noise', color=0, spkcount=0),
                1: dict(groupidx=1, name='Multi-unit', color=1, spkcount=0),
                2: dict(groupidx=2, name='Good', color=2, spkcount=nspikes),
            }
            clusters_info = get_clusters_info(clusters, groupidx=2)
            
            
        nclusters = len(clusters_info)
        self.holder.nclusters = nclusters
        
        self.holder.clusters_info = dict(
            clusters_info=clusters_info,
            groups_info=groups_info,
            )

            
            
            
        # c = Counter(clusters)
        # self.holder.clusters_counter = c
        
            
        probe = None
        try:
            if probefile:
                probe = np.loadtxt(probefile)
        except Exception as e:
            print(str(e))
        self.holder.probe = dict(positions=probe)
        
        # cross correlograms
        nsamples_correlograms = 20
        self.holder.correlograms_info = dict(nsamples=nsamples_correlograms)
        
        # self.holder.correlation_matrix = rdn.rand(nclusters, nclusters)
        # self.holder.correlation_matrix = np.array([[]])
        # features = 
        # self.holder.correlation_matrix = correlation_matrix(features, clusters)
        self.holder.correlation_matrix_queue = CorrelationMatrixQueue(self.holder)
        self.holder.correlation_matrix_queue.process()
        
        return self.holder
        
    def save(self, filename=None):
        # if filename is None:
            # self.filename + "_spiky.clu.%d" % self.fileindex
            
        # add nclusters at the top of the clu file
        data = self.holder.clusters
        data = np.hstack((data.max(), data))
        # save the CLU file
        save_text(filename, data)
        
        # HACK: save the groups
        filename_groups = filename.replace('.clu.', '.groups.')
        save_pickle(filename_groups, self.holder.clusters_info)
        

class SpikeDetektH5DataProvider(DataProvider):
    """Legacy SpikeDetekt HDF5 data provider with the old format."""
    def load(self, filename):
        
        pass
        
    def save(self, filename):
        pass


class H5DataProvider(DataProvider):
    """Spiky HDF5 data provider with the future format."""
    def load(self, filename):
        pass
        
    def save(self, filename):
        pass


class MockDataProvider(DataProvider):
    """Mock data provider with totally random data."""
    def load(self, nspikes=100, nsamples=20, nclusters=5, nchannels=32):
        
        self.holder = DataHolder()
        
        self.holder.freq = 20000.
        
        self.holder.nspikes = nspikes
        self.holder.nclusters = nclusters
        self.holder.nchannels = nchannels
        self.holder.nextrafet = 0
        
        self.holder.duration = 100.
        
        # construct spike times from random interspike interval
        self.holder.spiketimes = np.cumsum(np.random.randint(size=nspikes,
            low=int(self.holder.freq*.005), high=int(self.holder.freq*10)))
        
        
        waveforms = rdn.randn(nspikes, nsamples, nchannels)
        if nspikes>0:
            waveforms = (waveforms - waveforms.mean())
            waveforms = waveforms / np.abs(waveforms).max()
        
        self.holder.waveforms = waveforms
        self.holder.waveforms_info = dict(nsamples=nsamples)
        
        
        fetdim = 3
        # TODO
        # self.holder.features = rdn.randn(nspikes, nchannels, fetdim)
        self.holder.fetdim = fetdim
        self.holder.features = rdn.randn(nspikes, nchannels * fetdim + 1)
        
        masksind = rdn.randint(size=(nspikes, nchannels), low=0, high=3)
        self.holder.masks = np.array([0., .5, 1.])[masksind]
        # self.holder.masks[self.holder.masks < .25] = 0

        # a list of dict with the info about each group
        groups_info = [dict(name='Group 0', groupidx=0, color=0),]
        if nspikes > 0:
            self.holder.clusters = rdn.randint(low=0, high=nclusters, size=nspikes)
        else:
            self.holder.clusters = np.zeros(0)
        
        # create the groups info object
        groups_info = {0: dict(groupidx=0, name='Group 0', color=0, spkcount=nspikes)}
        clusters_info = get_clusters_info(self.holder.clusters)
        
        self.holder.clusters_info = dict(
            clusters_info=clusters_info,
            groups_info=groups_info,
            )
            

        # try:
            # probe = np.loadtxt("data/buzsaki32.txt")
        # except Exception as e:
            # print(str(e))
            # probe = None
        self.holder.probe = dict(positions=None)
        
        # cross correlograms
        nsamples_correlograms = 20
        self.holder.correlograms_info = dict(nsamples=nsamples_correlograms)
        
        self.holder.correlation_matrix = -np.ones((nclusters, nclusters))
        
        
        return self.holder
        
    def save(self, filename):
        pass


if __name__ == '__main__':
    # provider = MockDataProvider()
    # dataholder = provider.load()
    
    filename = r"D:\Git\spiky\_test\data\subset41test"
    
    print get_actual_filename(filename, 'clu')
    
    