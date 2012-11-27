import numpy as np
import numpy.random as rdn
from galry import *
import colors
from tools import Info




__all__ = [
    'DataHolder',
    'SelectDataHolder',
    'DataProvider',
    'H5DataProvider',
    'MockDataProvider',
    ]

    
    
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
        nclusters = len(clusters)
        # TODO
        return rdn.rand(nclusters * (nclusters + 1) / 2,
            self.correlograms_info.nsamples) 
            
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
        self.correlograms = self.get_correlograms(clusters)
        self.nclusters = len(clusters)
        self.cluster_colors = self.dataholder.clusters_info.colors[clusters]
        
        # override all spike dependent variables, where the first axis
        # is the spike index
        for varname in self.spike_dependent_variables:
            if hasattr(self.dataholder, varname):
                setattr(self, varname, getattr(self.dataholder, varname)[select_mask,...])
        
    def __getattr__(self, name):
        """Fallback mechanism for selecting variables in data holder and that
        have not been overloaded in SelectDataHolder."""
        return getattr(self.dataholder, name)
        

class DataProvider(object):
    """Provide import/export functions to load/save a DataHolder instance."""
    data_holder = None
    
    def load(self, filename):
        pass
        return self.data_holder
        
    def save(self, filename):
        pass


class H5DataProvider(DataProvider):
    def load(self, filename):
        pass
        
    def save(self, filename):
        pass


class MockDataProvider(DataProvider):
    def load(self, nspikes=100, nsamples=20, nclusters=5, nchannels=32):
        
        self.holder = DataHolder()
        
        self.holder.nspikes = nspikes
        self.holder.nclusters = nclusters
        self.holder.nchannels = nchannels
        
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
            colors=np.array(colors.generate_colors(nclusters),
                                    dtype=np.float32),
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
    
    
    
    
    