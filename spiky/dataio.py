import collections
import numpy as np
import numpy.random as rdn
import h5py
from galry import *
from colors import COLORMAP
from tools import Info

__all__ = [
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
        
def load_binary(file, dtype=None, count=None):
    if dtype is None:
        dtype = np.dtype(np.int16)
    if count is None:
        X = np.fromfile(file, dtype=dtype)
    else:
        X = np.fromfile(file, dtype=dtype, count=count)
    return X




    
# class Probe(object):
"""dict
nchannels: number of channels in the probe
positions: a nchannels*2 array with the coordinates of each channel
defaut_view: a dict with the info of the default view in waveformview
"""
    
# class ClustersInfo(object):
"""dict
nclusters: total number of clusters
groups: array of int with the group index for every cluster
group_info: a list of dict with the info for each group (name, etc)
colors: a nclusters*3 array with the color of each cluster
"""


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

    # print T
    # print T1
    # print T2
    # print H
    # print H / W
    # print
    
    return H / W


def get_correlogram(x, y, width=.021, bin=.001, duration=None):

    # return np.zeros(2*int(np.ceil(width / bin)))

    # TODO: this is highly unoptimized, optimize that
    corr = brian(x, y, width=width, bin=bin, T=duration)
    # print corr
    if corr is None:
        return np.zeros(2*int(np.ceil(width / bin)))
    corr[len(corr)/2] = 0
    return corr
    
    
class ClusterCache(object):
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
            # convert spike train units from samples counts to seconds
            x = x * 1. / self.dh.freq
            y = y * 1. / self.dh.freq
            
            corr = get_correlogram(x, y, width=self.width, bin=self.bin,
                duration=self.dh.duration)
            self.correlograms[(cluster0, cluster1)] = corr

        # return corr
        
    def get_correlogram(self, cluster0, cluster1=None):
        if cluster1 is None:
            cluster1 = cluster0
        # if (cluster0, cluster1) not in self.correlograms:
        # make sure the requested information is available (the compute
        # function only computes something if necessary
        self.compute(cluster0, cluster1)
        return self.correlograms[(cluster0, cluster1)]
    
    def get_correlograms(self, clusters):
        if len(clusters) == 0:
            return np.array([[]])
        # TODO: speed that up!
        correlograms = []
        for i in xrange(len(clusters)):
            for j in xrange(i, len(clusters)):
                # c = self.get_correlogram(clusters[i], clusters[j])
                self.compute(clusters[i], clusters[j])
                c = self.correlograms[(clusters[i], clusters[j])]
                # if c is None:
                    # c = np.zeros(self.histlen)
                correlograms.append(c)
        # print len(correlograms)
        # print [len(c) for c in correlograms]
        
        # return np.vstack([self.correlograms[(ci, cj)]
            # for (i, ci) in enumerate(clusters)
                # for cj in clusters[i:]])
                
        return np.vstack(correlograms)
    
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
        ids = self.clusters_info.names
        return ids.max() + 1

class SelectDataHolder(object):
    """Provides access to the data related to selected clusters."""
    def __init__(self, dataholder):
        self.dataholder = dataholder
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
        
    # def get_correlograms(self, clusters):
        # return self.clustercache.get_correlograms(clusters)
            
    # def get_spikecounts(self, clusters):
        # return self.clustercache.get_spikecounts(clusters)
            
    # def get_spiketimes(self, cluster):
        # # return self.dataholder.spiketimes[self.dataholder.clusters == cluster]
        # return self.clustercache.get_spiketimes([cluster])[0]
        
    def _selector_ufunc(self, clusters=None):
        """Create a custom ufunc for cluster selection."""
        if clusters is None or len(clusters) == 0:
            return np.frompyfunc(lambda x: False, 1, 1)
        s = "lambda x: " + " | ".join(["(x == %d)" % c for c in clusters])
        f = eval(s)
        uf = np.frompyfunc(f, 1, 1)
        return uf
        
    # @profile
    def select_clusters(self, clusters):
        """Provides the data related to the specified clusters."""
        uf = self._selector_ufunc(clusters)
        select_mask = np.array(uf(self.dataholder.clusters), dtype=np.bool)
        # relative cluster indices
        clusters_rel = [self.dataholder.clusters_info.cluster_indices[cl] for cl in clusters]
        
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
        self.cluster_colors = self.dataholder.clusters_info.colors[clusters_rel]
        # unique clusters
        self.clusters_unique = clusters
        
        # override all spike dependent variables, where the first axis
        # is the spike index
        if select_mask.size > 0:
            for varname in self.spike_dependent_variables:
                if hasattr(self.dataholder, varname):
                    setattr(self, varname, getattr(self.dataholder, varname)[select_mask,...])
        
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
    def load(self, filename):
        # klusters tests
        nchannels = 32
        # nspikes = 10000
        nsamples = 20
        
        try:
            clusters = load_text(filename + ".clu.1", np.int32)
            nspikes = len(clusters) - 1
        except Exception as e:
            log_warn("CLU file '%s' not found" % filename)
            clusters = np.zeros(nspikes + 1, dtype=np.int32)
            clusters[0] = 1
        nclusters = clusters[0]
        clusters = clusters[1:]
        
        features = load_text(filename + ".fet.1", np.int32, skiprows=1)
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
            masks = load_text(filename + ".fmask.1", np.float32, skiprows=1)
            masks = masks[:,:-1:3]
        except Exception as e:
            try:
                # otherwise, try mask
                masks = load_text(filename + ".mask.1", np.float32, skiprows=1)
                masks = masks[:,:-1:3]
            except:
                # finally, warning and default masks (everything to 1)
                log_warn("MASK file '%s' not found" % filename)
                masks = np.ones((nspikes, nchannels))
        
        try:
            waveforms = load_binary(filename + ".spk.1")
            waveforms = waveforms.reshape((nspikes, nsamples, nchannels))
        except IOError as e:
            log_warn("SPK file '%s' not found" % filename)
            waveforms = np.zeros((nspikes, nsamples, nchannels))
        
        self.holder = DataHolder()
        
        self.holder.freq = 20000.
        
        
        
        # find the number of spikes in each cluster
        # WARNING: the nclusters value in the .clu file may not correspond
        # to the number of different clusters, so we just get rid of it
        spkcounts = collections.Counter(clusters)
        cluster_keys = sorted(spkcounts.keys())
        spkcounts = np.array([spkcounts[key] for key in cluster_keys])
        cluster_names = np.array(cluster_keys)#map(str, cluster_keys)
        nclusters = len(cluster_names)
        # for each cluster absolute index, its relative index
        cluster_indices = dict([(key, i) for i, key in enumerate(cluster_keys)])
        
        
        self.holder.nspikes = nspikes
        self.holder.nclusters = nclusters
        self.holder.nchannels = nchannels
        
        # construct spike times from random interspike interval
        self.holder.spiketimes = spiketimes
        
        # TODO
        self.holder.duration = spiketimes[-1] / float(self.holder.freq)
    

        # NEW: normalize waveforms at once
        # waveforms = 2 * waveforms / np.abs(waveforms).max() - .5
        waveforms = (waveforms - waveforms.mean())
        waveforms = waveforms / np.abs(waveforms).max()
        
        self.holder.waveforms = waveforms
        self.holder.waveforms_info = Info(nsamples=nsamples)
        
        fetdim = 3
        self.holder.fetdim = fetdim
        self.holder.features = features
        
        self.holder.masks = masks
        
        # a list of dict with the info about each group
        groups_info = [dict(name='Group 0', groupidx=0)]
        self.holder.clusters = clusters
        self.holder.clusters_info = Info(
            colors=np.mod(np.arange(nclusters), len(COLORMAP)),
            names=cluster_names,
            spkcounts=spkcounts,
            groups_info=groups_info,
            cluster_indices=cluster_indices,
            groups=np.zeros(nclusters),
            )

        try:
            probe = np.loadtxt("data/buzsaki32.txt")
        except Exception as e:
            print(str(e))
            probe = None
        self.holder.probe = Info(positions=probe)
        
        # cross correlograms
        nsamples_correlograms = 20
        self.holder.correlograms_info = Info(nsamples=nsamples_correlograms)
        
        self.holder.correlationmatrix = rdn.rand(nclusters, nclusters) ** 10
        
        return self.holder
        
    def save(self, filename):
        pass

        
class SpikeDetektH5DataProvider(DataProvider):
    """Legacy SpikeDetekt HDF5 data provider with the old format."""
    def load(self, filename):
        
        pass
        # self.holder = DataHolder()
        
        # self.freq = 20000.
        
        # self.holder.nspikes = nspikes
        # self.holder.nclusters = nclusters
        # self.holder.nchannels = nchannels
        
        # # construct spike times from random interspike interval
        # self.holder.spiketimes = np.cumsum(np.random.randint(size=nspikes,
            # low=int(self.freq*.005), high=int(self.freq*10)))
        
        # self.holder.waveforms = rdn.randn(nspikes, nsamples, nchannels)
        # self.holder.waveforms_info = Info(nsamples=nsamples)
        
        # fetdim = 3
        # # TODO
        # # self.holder.features = rdn.randn(nspikes, nchannels, fetdim)
        # self.holder.fetdim = fetdim
        # self.holder.features = rdn.randn(nspikes, nchannels * fetdim + 1)
        
        # self.holder.masks = rdn.rand(nspikes, nchannels)
        # self.holder.masks[self.holder.masks < .25] = 0
        
        # # a list of dict with the info about each group
        # groups_info = [dict(name='Interneurons'),
                       # dict(name='MUA')]
        # self.holder.clusters = rdn.randint(low=0, high=nclusters, size=nspikes)
        # self.holder.clusters_info = Info(
            # colors=np.mod(np.arange(nclusters), len(COLORMAP)),
            # names=['%d' % i for i in xrange(nclusters)],
            # rates=rdn.rand(nclusters) * 20,
            # groups_info=groups_info,
            # groups=rdn.randint(low=0, high=len(groups_info), size=nclusters))

        # self.holder.probe = Info(positions=np.loadtxt("data/buzsaki32.txt"))
        
        # # cross correlograms
        # nsamples_correlograms = 20
        # # self.holder.correlograms = rdn.rand(nclusters * (nclusters + 1) / 2,
            # # nsamples_correlograms)
        # self.holder.correlograms_info = Info(nsamples=nsamples_correlograms)
        
        # self.holder.correlationmatrix = rdn.rand(nclusters, nclusters) ** 10
        
        
        # return self.holder
        
        
        
        
        
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
        
        # print waveforms.min(), waveforms.max()
        
        self.holder.waveforms = waveforms
        self.holder.waveforms_info = Info(nsamples=nsamples)
        
        
        
        fetdim = 3
        # TODO
        # self.holder.features = rdn.randn(nspikes, nchannels, fetdim)
        self.holder.fetdim = fetdim
        self.holder.features = rdn.randn(nspikes, nchannels * fetdim + 1)
        
        masksind = rdn.randint(size=(nspikes, nchannels), low=0, high=3)
        self.holder.masks = np.array([0., .5, 1.])[masksind]
        # self.holder.masks[self.holder.masks < .25] = 0


        
        # a list of dict with the info about each group
        # groups_info = [dict(name='Interneurons'),
                       # dict(name='MUA')]
        groups_info = [dict(name='Group 0', groupidx=0),]
        if nspikes > 0:
            self.holder.clusters = rdn.randint(low=0, high=nclusters, size=nspikes)
        else:
            self.holder.clusters = np.zeros(0)
        
        
        spkcounts = collections.Counter(self.holder.clusters)
        cluster_keys = sorted(spkcounts.keys())
        spkcounts = np.array([spkcounts[key] for key in cluster_keys])
        cluster_names = map(str, cluster_keys)
        nclusters = len(cluster_names)
        # for each cluster absolute index, its relative index
        cluster_indices = dict([(key, i) for i, key in enumerate(cluster_keys)])

        
        
        
        self.holder.clusters_info = Info(
            colors=np.mod(np.arange(nclusters), len(COLORMAP)),
            names=['%d' % i for i in xrange(nclusters)],
            spkcounts=rdn.rand(nclusters) * 20,
            groups_info=groups_info,
            groups=rdn.randint(low=0, high=len(groups_info), size=nclusters),
            cluster_indices=cluster_indices,)

        try:
            probe = np.loadtxt("data/buzsaki32.txt")
        except Exception as e:
            print(str(e))
            probe = None
        self.holder.probe = Info(positions=probe)
        
        # cross correlograms
        nsamples_correlograms = 20
        # self.holder.correlograms = rdn.rand(nclusters * (nclusters + 1) / 2,
            # nsamples_correlograms)
        self.holder.correlograms_info = Info(nsamples=nsamples_correlograms)
        
        self.holder.correlationmatrix = rdn.rand(nclusters, nclusters) ** 10
        
        
        return self.holder
        
    def save(self, filename):
        pass


if __name__ == '__main__':
    provider = MockDataProvider()
    dataholder = provider.load()
    
    
    
    
    