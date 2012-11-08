import numpy as np
import numpy.random as rdn

import colors
from tools import Info




__all__ = [
    'DataHolder',
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
    """
    freq: a float with the sampling frequency
    nchannels: number of channels in the probe
    nspikes: total number of spikes
    probe: a Probe dic
    total_duration: total duration, in samples count, of the current dataset
    current_window: a tuple with the interval, in samples cuont, of the current window
    spiketimes: an array with the spike times of the spikes, in samples count
    waveforms: a nspikes*nsamples*nchannels array with all waveforms
    waveforms_info: a dict with the info about the waveforms
    clusters: an array with the cluster index for each spike
    clusters_info: a ClustersInfo dic
    features: a nspikes*nchannels*fetdim array with the features of each spike, in each channel
    masks: a nspikes array with the mask for each spike, as a float in [0,1]
    raw_trace: a total_duration*nchannels array with the raw trace (or a HDF5 proxy with the same interface)
    filtered_trace: like raw trace, but with the filtered trace
    filter_info: a FilterInfo dic
    """



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
        
        self.holder.waveforms = rdn.randn(nspikes, nsamples, nchannels)
        self.holder.waveforms_info = Info(nsamples=nsamples)
        
        fetdim = 3
        # TODO
        # self.holder.features = rdn.randn(nspikes, nchannels, fetdim)
        self.holder.features = rdn.randn(nspikes, nchannels * fetdim + 1)
        
        self.holder.masks = rdn.rand(nspikes, nchannels)
        self.holder.masks[self.holder.masks < .25] = 0
        
        self.holder.clusters = rdn.randint(low=0, high=nclusters, size=nspikes)
        self.holder.clusters_info = Info(
            colors=np.array(colors.generate_colors(nclusters),
                                    dtype=np.float32))
                                    
        self.holder.probe = Info(positions=np.loadtxt("data/buzsaki32.txt"))
        
        # cross correlograms
        nsamples_correlograms = 20
        self.holder.correlograms = rdn.rand(nclusters * (nclusters + 1) / 2,
            nsamples_correlograms)
        self.holder.correlograms_info = Info(nsamples=nsamples_correlograms)
        
        self.holder.correlationmatrix = rdn.rand(nclusters, nclusters) ** 10
        
        
        return self.holder
        
        
    def save(self, filename):
        pass



if __name__ == '__main__':
    provider = MockDataProvider()
    dataholder = provider.load()
    
    
    
    
    