import collections
import threading
import numpy as np
import numpy.random as rdn
from copy import deepcopy
import h5py
from galry import *
from colors import COLORMAP
import signals
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




def load_text(file, dtype, skiprows=0):
    return np.loadtxt(file, dtype=dtype, skiprows=skiprows)
 
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

    
# Correlograms
# ------------
def brian(T1, T2, width=.02, bin=.001, T=None):
    '''
    Returns a cross-correlogram with lag in [-width,width] and given bin size.
    T is the total duration (optional) and should be greater than the duration of T1 and T2.
    The result is in Hz (rate of coincidences in each bin).

    N.B.: units are discarded.
    TODO: optimise?
    '''
    
    n = int(np.ceil(width / bin)) # Histogram length
    
    # print T1, T2, width, bin
    if (len(T1) == 0) or (len(T2) == 0): # empty spike train
        # return np.zeros(2 * n)
        return None
    # Remove units
    # width = float(width)
    # T1 = np.array(T1)
    # T2 = np.array(T2)
    i = 0
    j = 0
    # n = int(np.ceil(width / bin)) # Histogram length
    l = []
    for t in T1:
        while i < len(T2) and T2[i] < t - width: # other possibility use searchsorted
            i += 1
        while j < len(T2) and T2[j] < t + width:
            j += 1
        l.extend(T2[i:j] - t)
    H, _ = np.histogram(l, bins=np.arange(2 * n + 1) * bin - n * bin) #, new = True)

    # Divide by time to get rate
    if T is None:
        T = max(T1[-1], T2[-1]) - min(T1[0], T2[0])
        
    # Windowing function (triangle)
    W = np.zeros(2 * n)
    W[:n] = T - bin * np.arange(n - 1, -1, -1)
    W[n:] = T - bin * np.arange(n)

    return H / W

def get_correlogram(x, y, width=.021, bin=.001, duration=None):
    # TODO: this is highly unoptimized, optimize that
    corr = brian(x, y, width=width, bin=bin, T=duration)
    # print corr
    if corr is None:
        return np.zeros(2*int(np.ceil(width / bin)))
    corr[len(corr)/2] = 0
    return corr
    
    
class CorrelogramsThread(QtCore.QThread):
    def set_clustercache(self, clustercache):
        self.clustercache = clustercache
    
    def run(self):
        self.clustercache._get_correlograms()
    
    
class ClusterCache(object):
    
    # finished = QtCore.pyqtSignal()
    
    def __init__(self, dh, sdh, width=None, bin=None):
        self.dh = dh
        self.sdh = sdh
        
        if width is None:
            width = .02
        if bin is None:
            bin = .001
        
        self.width = width
        self.bin = bin
        self.histlen = 2 * int(np.ceil(width / bin))
        
        # cache correlograms, spikecounts, trains
        self.correlograms = {}
        self.spikecounts = {}
        self.spiketimes = {}
        
        self.th = CorrelogramsThread()
        self.th.set_clustercache(self)
        self.th.finished.connect(self.slotFinished)
        
        # self.is_finished = False
        self._correlograms =  np.array([[]])
        
    def reset(self):
        self.correlograms.clear()
        self.spikecounts.clear()
        self.spiketimes.clear()
    
    def invalidate(self, clusters):
        """Remove from the cache all correlograms related to the given
        clusters."""
        correlograms_new = {}
        spikecounts_new = {}
        spiketimes_new = {}
        # copy in the new dictionary all correlograms which do not refer
        # to clusters in the given list of invalidated clusters
        for (i, j), corr in self.correlograms.iteritems():
            if i not in clusters and j not in clusters:
                # update correlograms
                correlograms_new[(i, j)] = self.correlograms[(i, j)]
                # update spike counts too
                spikecounts_new[i] = self.spikecounts[i]
                spikecounts_new[j] = self.spikecounts[j]
                # update spike trains too
                spiketimes_new[i] = self.spiketimes[i]
                spiketimes_new[j] = self.spiketimes[j]
        self.correlograms = correlograms_new
        self.spikecounts = spikecounts_new
        self.spiketimes = spiketimes_new
    
    def compute(self, cluster0, cluster1=None):
        if cluster1 is None:
            cluster1 = cluster0

        # compute the spike times of the two clusters if needed
        if cluster0 not in self.spiketimes:
            self.spiketimes[cluster0] = self.dh.spiketimes[self.dh.clusters == cluster0]
        if cluster1 not in self.spiketimes:
            self.spiketimes[cluster1] = self.dh.spiketimes[self.dh.clusters == cluster1]
            
        x, y = self.spiketimes[cluster0], self.spiketimes[cluster1]
        
        # compute the spike counts of the two clusters if needed
        if cluster0 not in self.spikecounts:
            self.spikecounts[cluster0] = len(x)
        if cluster1 not in self.spikecounts:
            self.spikecounts[cluster1] = len(y)

        # compute correlograms of the two clusters if needed
        if (cluster0, cluster1) not in self.correlograms:
            # print cluster0, cluster1
            # convert spike train units from samples counts to seconds
            x = x * 1. / self.dh.freq
            y = y * 1. / self.dh.freq
            corr = get_correlogram(x, y, width=self.width, bin=self.bin,
                duration=self.dh.duration)
            self.correlograms[(cluster0, cluster1)] = corr

    def get_correlogram(self, cluster0, cluster1=None):
        if cluster1 is None:
            cluster1 = cluster0
        # if (cluster0, cluster1) not in self.correlograms:
        # make sure the requested information is available (the compute
        # function only computes something if necessary
        self.compute(cluster0, cluster1)
        return self.correlograms[(cluster0, cluster1)]
    
    def _get_correlograms(self):
        clusters = self._clusters
        # self.is_finished = False
        
        if len(clusters) == 0:
            return np.array([[]])
        correlograms = []
        for i in xrange(len(clusters)):
            for j in xrange(i, len(clusters)):
                self.compute(clusters[i], clusters[j])
                c = self.correlograms[(clusters[i], clusters[j])]
                correlograms.append(c)
                
        # self.is_finished = True
        self._correlograms = np.vstack(correlograms)
        
        return self._correlograms
        
    def get_correlograms(self, clusters):
        # return self._get_correlograms(clusters)
        
        self._clusters = clusters
        
        # start the thread only if it has not finished and it is not running
        # if not self.is_finished:
            
        if not self.th.isRunning():
            # print "start", clusters
            self.th.start()
        
            # return np.array([[]])
        
        # else:
            
        return self._correlograms
        
    def slotFinished(self):
        # print "finished"
        self.sdh.correlograms = self._correlograms
        signals.emit(self, 'CorrelogramsUpdated')
    
    def get_spikecounts(self, clusters):
        # make sure the requested information is available (the compute
        # function only computes something if necessary
        [self.compute(c) for c in clusters]
        return np.array([self.spikecounts[i] for i in clusters])
    
    def get_spiketimes(self, clusters):
        # make sure the requested information is available (the compute
        # function only computes something if necessary
        [self.compute(c) for c in clusters]
        return [self.spiketimes[i] for i in clusters]
    
    
    
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
    def new_cluster(self):
        ids = self.clusters_info['clusters_info'].keys()
        return max(ids) + 1

class SelectDataHolder(object):
    """Provides access to the data related to selected clusters."""
    def __init__(self, dataholder):
        self.dataholder = dataholder
        self.override_color = False
        self.clustercache = ClusterCache(dataholder, self)
        self.spike_dependent_variables = [
            'spiketimes',
            'waveforms',
            'clusters',
            'features',
            'masks',
            ]
        # DEBUG
        # self.select_clusters([0,1,2])
        self.select_clusters([])
        
    def _selector_ufunc(self, clusters=None):
        """Create a custom ufunc for cluster selection."""
        if clusters is None or len(clusters) == 0:
            return np.frompyfunc(lambda x: False, 1, 1)
        s = "lambda x: " + " | ".join(["(x == %d)" % c for c in clusters])
        f = eval(s)
        uf = np.frompyfunc(f, 1, 1)
        return uf
        
    # @property
    # def correlograms(self):
        # return self.clustercache.get_correlograms()
    
        
    # @profile
    def select_clusters(self, clusters):
        """Provides the data related to the specified clusters."""
        uf = self._selector_ufunc(clusters)
        select_mask = np.array(uf(self.dataholder.clusters), dtype=np.bool)
        # relative cluster indices
        # clusters_rel = [self.dataholder.clusters_info.cluster_indices[cl] for cl in clusters]
        
        self.spike_ids = np.nonzero(select_mask)[0]
        
        # nspikes is the number of True elements in select_mask
        self.nspikes = select_mask.sum()
        # TODO: move that outside dataio
        self.correlograms = self.clustercache.get_correlograms(clusters)
        
        counts = self.clustercache.get_spikecounts(clusters)
        counts = np.array([counts[i] for i in xrange(len(clusters)) 
                        for j in xrange(i, len(clusters))])
        
        # print counts
        # print self.correlograms
        
        self.baselines = counts / float(self.dataholder.duration)
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
        
        # override all spike dependent variables, where the first axis
        # is the spike index
        if select_mask.size > 0:
            for varname in self.spike_dependent_variables:
                if hasattr(self.dataholder, varname):
                    setattr(self, varname, getattr(self.dataholder, varname)[select_mask,...])
    
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


class KlustersDataProvider(DataProvider):
    """Legacy Klusters data provider with the old format."""
    def load_probe(self, filename):
        pass
        
    def load(self, filename, fileindex=1):
        # klusters tests
        nchannels = 32
        # nspikes = 10000
        nsamples = 20
        
        self.filename = filename
        self.fileindex = fileindex
        
        if filename.endswith('_spiky'):
            filename = filename.replace('_spiky', '')
            spiky = True
        else:
            spiky = False
        
        try:
            if spiky:
                path = filename + "_spiky.clu.%d" % fileindex
            else:
                path = filename + ".clu.%d" % fileindex
            clusters = load_text(path, np.int32)
            nspikes = len(clusters) - 1
        except Exception as e:
            log_warn("CLU file '%s' not found" % filename)
            clusters = np.zeros(nspikes + 1, dtype=np.int32)
            clusters[0] = 1
        # nclusters = clusters[0]
        clusters = clusters[1:]
        
        features = load_text(filename + ".fet.%d" % fileindex, np.int32, skiprows=1)
        features = np.array(features, dtype=np.float32)
        features = features.reshape((-1, 97))
        # get the spiketimes
        spiketimes = features[:,-1].copy()
        # remove the last column in features, containing the spiketimes
        features = features[:,:-1]
        # normalize the data here
        # dn = DataNormalizer(np.array(features, dtype=np.float32))
        # features = dn.normalize(symmetric=True)
        m = features.min()
        M = features.max()
        # force symmetry
        vx = max(np.abs(m), np.abs(M))
        m, M = -vx, vx
        features = -1+2*(features-m)/(M-m)
        
        # first: try fmask
        try:
            masks = load_text(filename + ".fmask.%d" % fileindex, np.float32, skiprows=1)
            masks = masks[:,:-1:3]
        except Exception as e:
            try:
                # otherwise, try mask
                masks = load_text(filename + ".mask.%d" % fileindex, np.float32, skiprows=1)
                masks = masks[:,:-1:3]
            except:
                # finally, warning and default masks (everything to 1)
                log_warn("MASK file '%s' not found" % filename)
                masks = np.ones((nspikes, nchannels))
        
        try:
            waveforms = load_binary(filename + ".spk.%d" % fileindex)
            waveforms = waveforms.reshape((nspikes, nsamples, nchannels))
        except IOError as e:
            log_warn("SPK file '%s' not found" % filename)
            waveforms = np.zeros((nspikes, nsamples, nchannels))
        
        self.holder = DataHolder()
        
        self.holder.freq = 20000.
        
        
        self.holder.nspikes = nspikes
        self.holder.nchannels = nchannels
        
        # construct spike times from random interspike interval
        self.holder.spiketimes = spiketimes
        
        # TODO
        self.holder.duration = spiketimes[-1] / float(self.holder.freq)
    
        # normalize waveforms at once
        waveforms = (waveforms - waveforms.mean())
        waveforms = waveforms / np.abs(waveforms).max()
        
        self.holder.waveforms = waveforms
        self.holder.waveforms_info = dict(nsamples=nsamples)
        
        fetdim = 3
        self.holder.fetdim = fetdim
        self.holder.features = features
        
        self.holder.masks = masks
        
        self.holder.clusters = clusters
        
        # create the groups info object
        # Default groups
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
            # groups=np.zeros(nclusters, dtype=np.int32),
            )

        try:
            probe = np.loadtxt("data/buzsaki32.txt")
        except Exception as e:
            print(str(e))
            probe = None
        self.holder.probe = dict(positions=probe)
        
        # cross correlograms
        nsamples_correlograms = 20
        self.holder.correlograms_info = dict(nsamples=nsamples_correlograms)
        
        self.holder.correlationmatrix = rdn.rand(nclusters, nclusters) ** 10
        
        return self.holder
        
    def save(self, filename=None):
        if filename is None:
            self.filename + "_spiky.clu.%d" % self.fileindex
        # add nclusters at the top of the clu file
        data = self.holder.clusters
        data = np.hstack((data.max(), data))
        # save the CLU file
        save_text(filename, data)
        

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
            

        try:
            probe = np.loadtxt("data/buzsaki32.txt")
        except Exception as e:
            print(str(e))
            probe = None
        self.holder.probe = dict(positions=probe)
        
        # cross correlograms
        nsamples_correlograms = 20
        self.holder.correlograms_info = dict(nsamples=nsamples_correlograms)
        
        self.holder.correlationmatrix = rdn.rand(nclusters, nclusters) ** 10
        
        
        return self.holder
        
    def save(self, filename):
        pass


if __name__ == '__main__':
    provider = MockDataProvider()
    dataholder = provider.load()
    
    
    
    
    