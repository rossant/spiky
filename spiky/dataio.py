import collections
import threading
import numpy as np
import numpy.random as rdn
from copy import deepcopy as dcopy
import h5py
from galry import *
from colors import COLORMAP
import signals
from xmltools import parse_xml
from spiky.qtqueue import qtjobqueue
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
        
    return H * 1.
        
    # # Windowing function (triangle)
    # W = np.zeros(2 * n)
    # W[:n] = T - bin * np.arange(n - 1, -1, -1)
    # W[n:] = T - bin * np.arange(n)

    # return H / W

def get_correlogram(x, y, width=.021, bin=.001, duration=None):
    # TODO: this is highly unoptimized, optimize that
    corr = brian(x, y, width=width, bin=bin, T=duration)
    if corr is None:
        return np.zeros(2*int(np.ceil(width / bin)))
    corr[len(corr)/2] = 0
    return corr
  
  

@qtjobqueue
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
        
        if len(clusters) == 0:
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
            
        # process correlograms
        for i in xrange(len(clusters)):
            # first spike train
            spk0 = spiketimes[i] / float(self.dh.freq)
            for j in xrange(i, len(clusters)):
                # second spike train
                spk1 = spiketimes[j] / float(self.dh.freq)
                
                cl0 = clusters[i]
                cl1 = clusters[j]
                
                # compute correlogram
                if (cl0, cl1) not in self.correlograms:
                    corr = get_correlogram(spk0, spk1, width=self.width, bin=self.bin,
                        duration=self.dh.duration)
                    self.correlograms[(cl0, cl1)] = corr
                else:
                    corr = self.correlograms[(cl0, cl1)]
                correlograms.append(corr)
                
        # populate SDH
        self.sdh.spiketimes = spiketimes
        self.sdh.correlograms = np.vstack(correlograms)
        
        self.update_signal()
    
    def update_signal(self):
        """Raise the update signal, specifying that the correlogram view
        needs to be updated."""
        signals.emit(self, 'CorrelogramsUpdated')
        
        
    
    
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
        self.select_clusters([])
        
    def _selector_ufunc(self, clusters=None):
        """Create a custom ufunc for cluster selection."""
        if clusters is None or len(clusters) == 0:
            return np.frompyfunc(lambda x: False, 1, 1)
        s = "lambda x: " + " | ".join(["(x == %d)" % c for c in clusters])
        f = eval(s)
        uf = np.frompyfunc(f, 1, 1)
        return uf
    
    def invalidate(self, clusters):
        self.clustercache.invalidate(clusters)
        
    # @profile
    def select_clusters(self, clusters):
        """Provides the data related to the specified clusters."""
        uf = self._selector_ufunc(clusters)
        select_mask = np.array(uf(self.dataholder.clusters), dtype=np.bool)
        self.spike_ids = np.nonzero(select_mask)[0]
        
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
    # def load_probe(self, filename):
        # pass
        
    def load(self, filename, fileindex=1, probefile=None):
        
        # load XML
        
        try:
            params = parse_xml(filename + ".xml", fileindex=fileindex)
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
        
        # HACK: there are either 1 or 5 dimensions more than fetdim*nchannels
        # we can't be sure so we first try 1, if it does not work we try 5
        try:
            features = features.reshape((-1, fetdim * nchannels + 1))
        except:
            log_warn("The number of columns is not fetdim (%d) x nchannels (%d) + 1." \
                % (fetdim, nchannels))
            try:
                features = features.reshape((-1, fetdim * nchannels + 5))
                
            except:
                log_warn("The number of columns is not fetdim (%d) x nchannels (%d) + 5, so I'm confused and I can't continue. Sorry :(" \
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
        
        self.holder.freq = freq
        
        
        self.holder.nspikes = nspikes
        self.holder.nchannels = nchannels
        self.holder.nextrafet = nextrafet
        
        # construct spike times from random interspike interval
        self.holder.spiketimes = spiketimes
        
        # TODO
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
        
        # self.holder.correlationmatrix = rdn.rand(nclusters, nclusters) ** 10
        
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
        
        self.holder.correlationmatrix = rdn.rand(nclusters, nclusters) ** 10
        
        
        return self.holder
        
    def save(self, filename):
        pass


if __name__ == '__main__':
    provider = MockDataProvider()
    dataholder = provider.load()
    
    
    
    
    