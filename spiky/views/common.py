import numpy as np
import operator

from galry import *


__all__ = ['SpikeDataOrganizer', 'HighlightManager']


class SpikeDataOrganizer(object):
    def __init__(self, *args, **kwargs):
        # set data
        self.set_data(*args, **kwargs)
        # reorder data
        self.reorder()
        
    def set_data(self, data, clusters=None, cluster_colors=None, masks=None,
                             nchannels=None, spike_ids=None):
        """
        Arguments:
          * data: a Nspikes x ?? (x ??) array
          * clusters: a Nspikes array, dtype=int, absolute indices
          * cluster_colors: as a function of the RELATIVE index
        """
        # get the number of spikes from the first dimension of data
        self.nspikes = data.shape[0]
        self.ndim = data.ndim
        
        # check arguments
        if nchannels is None:
            raise TypeError("The number of channels should be specified.")
            
        # default arguments
        if clusters is None:
            clusters = np.zeros(self.nspikes, dtype=np.int)
        if masks is None:
            masks = np.ones((self.nspikes, self.nchannels))
        if spike_ids is None:
            spike_ids = np.arange(self.nspikes)
            
        self.data = enforce_dtype(data, np.float32)
        self.clusters = enforce_dtype(clusters, np.int32)
        self.masks = enforce_dtype(masks, np.float32)
        self.cluster_colors = enforce_dtype(cluster_colors, np.float32)
        
        # unique clusters
        self.clusters_unique = np.unique(clusters)
        self.clusters_unique.sort()
        self.nclusters = len(self.clusters_unique)
        
        if cluster_colors is None:
            cluster_colors = np.ones((self.nclusters, 3))
        self.cluster_colors = self.cluster_colors[:self.nclusters,:]
        
        # same as clusters, but with relative indexing instead of absolute
        clusters_rel = np.arange(self.clusters_unique.max() + 1)
        clusters_rel[self.clusters_unique] = np.arange(self.nclusters)
        self.clusters_rel = clusters_rel[self.clusters]
    
    def get_reordering(self):
        # regroup spikes from the same clusters, so that all data from
        # one cluster are contiguous in memory (better for OpenGL rendering)
        # permutation contains the spike indices in successive clusters
        self.permutation = []
        self.cluster_sizes_dict = {}
        self.cluster_sizes_cum = {}
        counter = 0
        for cluster in self.clusters_unique:
            # spike indices in the current cluster
            ids = np.nonzero(self.clusters == cluster)[0]
            # size of the current cluster
            size = len(ids)
            # record the size
            self.cluster_sizes_dict[cluster] = size
            # record the total number of spikes before the first spike in the
            # current cluster
            self.cluster_sizes_cum[cluster] = counter
            # create the spike permutation to regroup those in the same clusters
            self.permutation.append(ids)
            counter += size
        self.permutation = np.hstack(self.permutation)
        return self.permutation
        
    def reorder(self, permutation=None):
        if permutation is None:
            permutation = self.get_reordering()
        # reorder data
        if self.ndim == 1:
            self.data_reordered = self.data[permutation]
        elif self.ndim == 2:
            self.data_reordered = self.data[permutation,:]
        elif self.ndim == 3:
            self.data_reordered = self.data[permutation,:,:]
            
        # reorder masks
        self.masks = self.masks[permutation,:]
        self.clusters = self.clusters[permutation,:]
        self.clusters_rel = self.clusters_rel[permutation,:]
        
        # array of cluster sizes as a function of the relative index
        self.cluster_sizes = np.array(map(operator.itemgetter(1),
                                    sorted(self.cluster_sizes_dict.iteritems(),
                                            key=operator.itemgetter(0))))
        
        return self.data_reordered


class HighlightManager(object):
    
    highlight_rectangle_color = (0.75, 0.75, 1., .25)
    
    def initialize(self):
        self.highlight_box = None
        self.paint_manager.ds_highlight_rectangle = \
            self.paint_manager.create_dataset(RectanglesTemplate,
                coordinates=(0., 0., 0., 0.),
                color=self.highlight_rectangle_color,
                is_static=True,
                visible=False)
    
    def highlight(self, enclosing_box):
        # get the enclosing box in the window relative coordinates
        x0, y0, x1, y1 = enclosing_box
        
        # set the highlight box, in window relative coordinates, used
        # for displaying the selection rectangle on the screen
        self.highlight_box = (x0, y0, x1, y1)
        
        # paint highlight box
        self.paint_manager.set_data(visible=True,
            coordinates=self.highlight_box,
            dataset=self.paint_manager.ds_highlight_rectangle)
        
        # convert the box coordinates in the data coordinate system
        x0, y0 = self.interaction_manager.get_data_coordinates(x0, y0)
        x1, y1 = self.interaction_manager.get_data_coordinates(x1, y1)
        
        self.highlighted((x0, y0, x1, y1))
        
    def highlighted(self, box):
        pass

    def cancel_highlight(self):
        # self.set_highlighted_spikes([])
        if self.highlight_box is not None:
            self.paint_manager.set_data(visible=False,
                dataset=self.paint_manager.ds_highlight_rectangle)
            self.highlight_box = None
    

