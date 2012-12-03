import numpy as np
import numpy.random as rdn
import h5py
from galry import *
from views.colors import COLORMAP
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
def get_correlogram(x, y, nbins=20, bin=.002):
    # TODO (this is the most efficient way of computing correlograms ever)
    return np.random.rand(nbins)
    
class CorrelogramManager(object):
    def __init__(self, dh, sdh):
        self.dh = dh
        self.sdh = sdh
        # cache correlograms
        self.correlograms = {}
        
    def reset(self):
        self.correlograms.clear()
    
    def invalidate(self, clusters):
        """Remove from the cache all correlograms related to the given
        clusters."""
        correlograms_new = {}
        # copy in the new dictionary all correlograms which do not refer
        # to clusters in the given list of invalidated clusters
        for (i, j), corr in self.correlograms.iteritems():
            if i not in clusters and j not in clusters:
                correlograms_new[(i, j)] = self.correlograms[(i, j)]
        self.correlograms = correlograms_new
    
    def compute(self, cluster0, cluster1=None):
        if cluster1 is None:
            cluster1 = cluster0
        x = self.sdh.get_spiketimes(cluster0)
        y = self.sdh.get_spiketimes(cluster1)
        corr = get_correlogram(x, y)
        self.correlograms[(cluster0, cluster1)] = corr
        return corr
        
    def get_correlogram(self, cluster0, cluster1=None):
        if cluster1 is None:
            cluster1 = cluster0
        if (cluster0, cluster1) not in self.correlograms:
            self.compute(cluster0, cluster1)
        return self.correlograms[(cluster0, cluster1)]
    
    def get_correlograms(self, clusters):
        if len(clusters) == 0:
            return np.array([[]])
        # TODO: speed that up!
        correlograms = []
        for i in xrange(len(clusters)):
            for j in xrange(i, len(clusters)):
                correlograms.append(self.get_correlogram(clusters[i], clusters[j]))
        return np.vstack(correlograms)
        
    
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

class SelectDataHolder(object):
    """Provides access to the data related to selected clusters."""
    def __init__(self, dataholder):
        self.dataholder = dataholder
        self.corr_manager = CorrelogramManager(dataholder, self)
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
        
    def get_correlograms(self, clusters):
        # nclusters = len(clusters)
        # TODO
        # return rdn.rand(nclusters * (nclusters + 1) / 2,
            # self.correlograms_info.nsamples) 
        return self.corr_manager.get_correlograms(clusters)
            
    def _selector_ufunc(self, clusters=None):
        """Create a custom ufunc for cluster selection."""
        if clusters is None or len(clusters) == 0:
            return np.frompyfunc(lambda x: False, 1, 1)
        s = "lambda x: " + " | ".join(["(x == %d)" % c for c in clusters])
        f = eval(s)
        uf = np.frompyfunc(f, 1, 1)
        return uf
        
    def select_clusters(self, clusters):
        """Provides the data related to the specified clusters."""
        uf = self._selector_ufunc(clusters)
        select_mask = np.array(uf(self.dataholder.clusters), dtype=np.bool)
        
        # nspikes is the number of True elements in select_mask
        self.nspikes = select_mask.sum()
        # TODO: move that outside dataio
        self.correlograms = self.get_correlograms(clusters)
        self.nclusters = len(clusters)
        self.cluster_colors = self.dataholder.clusters_info.colors[clusters]
        
        # override all spike dependent variables, where the first axis
        # is the spike index
        for varname in self.spike_dependent_variables:
            if hasattr(self.dataholder, varname):
                setattr(self, varname, getattr(self.dataholder, varname)[select_mask,...])
        
    def get_spiketimes(self, cluster):
        return self.dataholder.spiketimes[self.dataholder.clusters == cluster]
        
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
        nspikes = 10000
        nsamples = 20
        
        clusters = load_text(filename + ".clu.1", np.int32)
        nclusters = clusters[0]
        clusters = clusters[1:]
        
        features = load_text(filename + ".fet.1", np.int32, skiprows=1)
        features = features.reshape((-1, 97))
        spiketimes = features[:,-1]
        # features = features[:,:-1]
        
        masks = load_text(filename + ".mask.1", np.float32, skiprows=1)
        masks = masks[:,:-1:3]
        
        waveforms = load_binary(filename + ".spk.1")
        waveforms = waveforms.reshape((nspikes, nsamples, nchannels))
        
        self.holder = DataHolder()
        
        self.freq = 20000.
        
        self.holder.nspikes = nspikes
        self.holder.nclusters = nclusters
        self.holder.nchannels = nchannels
        
        # construct spike times from random interspike interval
        self.holder.spiketimes = spiketimes
        
        self.holder.waveforms = waveforms
        self.holder.waveforms_info = Info(nsamples=nsamples)
        
        fetdim = 3
        self.holder.fetdim = fetdim
        self.holder.features = features
        
        self.holder.masks = masks
        
        # a list of dict with the info about each group
        groups_info = [dict(name='Group')]
        self.holder.clusters = clusters
        self.holder.clusters_info = Info(
            colors=np.mod(np.arange(nclusters), len(COLORMAP)),
            names=['cluster%d' % i for i in xrange(nclusters)],
            rates=np.zeros(nclusters),
            groups_info=groups_info,
            groups=np.zeros(nclusters),
            )

        self.holder.probe = Info(positions=np.loadtxt("data/buzsaki32.txt"))
        
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
            # names=['cluster%d' % i for i in xrange(nclusters)],
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
        
        self.freq = 20000.
        
        self.holder.nspikes = nspikes
        self.holder.nclusters = nclusters
        self.holder.nchannels = nchannels
        
        # construct spike times from random interspike interval
        self.holder.spiketimes = np.cumsum(np.random.randint(size=nspikes,
            low=int(self.freq*.005), high=int(self.freq*10)))
        
        self.holder.waveforms = rdn.randn(nspikes, nsamples, nchannels)
        self.holder.waveforms_info = Info(nsamples=nsamples)
        
        fetdim = 3
        # TODO
        # self.holder.features = rdn.randn(nspikes, nchannels, fetdim)
        self.holder.fetdim = fetdim
        self.holder.features = rdn.randn(nspikes, nchannels * fetdim + 1)
        
        self.holder.masks = rdn.rand(nspikes, nchannels)
        self.holder.masks[self.holder.masks < .25] = 0
        
        # a list of dict with the info about each group
        groups_info = [dict(name='Interneurons'),
                       dict(name='MUA')]
        self.holder.clusters = rdn.randint(low=0, high=nclusters, size=nspikes)
        self.holder.clusters_info = Info(
            colors=np.mod(np.arange(nclusters), len(COLORMAP)),
            names=['cluster%d' % i for i in xrange(nclusters)],
            rates=rdn.rand(nclusters) * 20,
            groups_info=groups_info,
            groups=rdn.randint(low=0, high=len(groups_info), size=nclusters))

        self.holder.probe = Info(positions=np.loadtxt("data/buzsaki32.txt"))
        
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
    
    
    
    
    