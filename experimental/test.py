from spiky import *
import numpy as np

# file = r"D:\Spike sorting\n6mab031109_buzsaki32\n6mab031109_buzsaki32.h5"

class TestSpiky(SpikyMainWindow):
    def initialize_data(self):
        filename = "data/test"
        
        provider = KlustersDataProvider()
        self.dh = provider.load(filename)
        
        # provider = MockDataProvider()
        # self.dh = provider.load()
        
        self.sdh = SelectDataHolder(self.dh)

        
        
        
if __name__ == '__main__':
    window = show_window(TestSpiky)

    # H5 tests
    # NOTE: use pytables instead of h5py for spikedetekt H5 file
    # import h5py
    # f = h5py.File(file)
    # nchannels = len(f['DatChannels'])
    # tab = f['SpikeTable_temp']
    # nspikes = tab.attrs['NROWS']
    
    # filename = "data/test"
    
    # # klusters tests
    # nchannels = 32
    # nspikes = 10000
    # nsamples = 20
    
    # clusters = load_text(filename + ".clu.1", np.int32)
    # nclusters = clusters[0]
    # clusters = clusters[1:]
    
    # features = load_text(filename + ".fet.1", np.int32, skiprows=1)
    # features = features.reshape((-1, 97))
    # spiketimes = features[:,-1]
    # # features = features[:,:-1]
    
    # masks = load_text(filename + ".fmask.1", np.float32, skiprows=1)
    # masks = masks[:,:-1:3]
    
    # waveforms = load_binary(filename + ".spk.1")
    # waveforms = waveforms.reshape((nspikes, nsamples, nchannels))
    
    # # print nclusters
    # # print clusters.shape
    # # print spiketimes.shape
    # # print features.shape
    # # print masks.shape
    # # print waveforms.shape
    
# # else:
    
    # self.holder = DataHolder()
    
    # self.freq = 20000.
    
    # self.holder.nspikes = nspikes
    # self.holder.nclusters = nclusters
    # self.holder.nchannels = nchannels
    
    # # construct spike times from random interspike interval
    # self.holder.spiketimes = spiketimes
    
    # self.holder.waveforms = waveforms
    # self.holder.waveforms_info = Info(nsamples=nsamples)
    
    # fetdim = 3
    # self.holder.fetdim = fetdim
    # self.holder.features = features
    
    # self.holder.masks = masks
    
    # # a list of dict with the info about each group
    # groups_info = [dict(name='Group')]
    # self.holder.clusters = clusters
    # self.holder.clusters_info = Info(
        # colors=np.mod(np.arange(nclusters), len(COLORMAP)),
        # names=['cluster%d' % i for i in xrange(nclusters)],
        # rates=np.zeros(nclusters),
        # groups_info=groups_info,
        # groups=np.zeros(nclusters),
        # )

    # self.holder.probe = Info(positions=np.loadtxt("data/buzsaki32.txt"))
    
    # # cross correlograms
    # nsamples_correlograms = 20
    # self.holder.correlograms_info = Info(nsamples=nsamples_correlograms)
    
    # self.holder.correlationmatrix = rdn.rand(nclusters, nclusters) ** 10
    
    
    # return self.holder
    
    
      # * channel_mask            int8            (Nchannels,)
      # * clu                     int32           ()
      # * fet                     float32         (Nchannels, fetdim)
      # * fet_mask                int8            (Nchannels x fetdim + 1,)
      # * float_channel_mask      float32         (Nchannels,)
      # * float_fet_mask          float32         (Nchannels x fetdim + 1,)
      # * time                    int32           ()
      # * unfiltered_wave         int32           (Nsamples, Nchannels)
      # * wave                    float32         (Nsamples, Nchannels)

    
    