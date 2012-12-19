import numpy as np
import operator
import collections

from galry import *


__all__ = ['SpikeDataOrganizer', 'HighlightManager', 'SpikyBindings',
           ]


class SpikeDataOrganizer(object):
    def __init__(self, *args, **kwargs):
        # set data
        self.set_data(*args, **kwargs)
        # reorder data
        # self.reorder()
        
    def set_data(self, data, clusters=None, cluster_colors=None, masks=None,
                             nchannels=None, spike_ids=None,
                             clusters_unique=None):
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
        
        # unique clusters
        if clusters_unique is None:
            self.clusters_unique = np.unique(clusters)
        else:
            self.clusters_unique = clusters_unique
        self.clusters_unique.sort()
        self.nclusters = len(self.clusters_unique)
        
        if cluster_colors is None:
            cluster_colors = np.ones(self.nclusters)
        cluster_colors = enforce_dtype(cluster_colors, np.int32)
        self.cluster_colors = cluster_colors[:self.nclusters,...]
        
        # same as clusters, but with relative indexing instead of absolute
        if self.nclusters > 0:
            clusters_rel = np.arange(self.clusters_unique.max() + 1)
            clusters_rel[self.clusters_unique] = np.arange(self.nclusters)
            self.clusters_rel = clusters_rel[self.clusters]
        else:
            self.clusters_rel = self.clusters

        # NEW: no more reordering
        self.data_reordered = self.data
        
        counter = collections.Counter(clusters)
        self.cluster_sizes_dict = counter
        self.cluster_sizes = np.array(map(operator.itemgetter(1),
                                    sorted(self.cluster_sizes_dict.iteritems(),
                                            key=operator.itemgetter(0))))
    
    def get_reordering(self):
        # regroup spikes from the same clusters, so that all data from
        # one cluster are contiguous in memory (better for OpenGL rendering)
        # permutation contains the spike indices in successive clusters
        permutation = []
        for cluster in self.clusters_unique:
            ids = np.nonzero(self.clusters == cluster)[0]
            size = len(ids)
            permutation.append(ids)
        if permutation:
            permutation = np.hstack(permutation)
        else:
            permutation = np.array([], dtype=np.int32)
        return permutation
        
    # def reorder(self, permutation=None):
        # if self.nclusters == 0 or self.nspikes == 0:
            # self.data_reordered = self.data
            # self.cluster_sizes = np.array([], dtype=np.int32)
            # permutation = self.get_reordering()
            # return self.data
            
        # if permutation is None:
            # permutation = self.get_reordering()
            
        # # reorder data
        # self.data_reordered = self.data[permutation,...]
            
        # # reorder masks
        # self.masks = self.masks[permutation,:]
        # self.clusters = self.clusters[permutation,:]
        # self.clusters_rel = self.clusters_rel[permutation,:]
        
        # # array of cluster sizes as a function of the relative index
        # self.cluster_sizes = np.array(map(operator.itemgetter(1),
                                    # sorted(self.cluster_sizes_dict.iteritems(),
                                            # key=operator.itemgetter(0))))
        
        # return self.data_reordered


class HighlightManager(Manager):
    
    highlight_rectangle_color = (0.75, 0.75, 1., .25)
    
    def initialize(self):
        self.highlight_box = None
        # self.paint_manager.ds_highlight_rectangle = \
        if not self.paint_manager.get_visual('highlight_rectangle'):
            self.paint_manager.add_visual(RectanglesVisual,
                                    coordinates=(0., 0., 0., 0.),
                                    color=self.highlight_rectangle_color,
                                    is_static=True,
                                    visible=False,
                                    name='highlight_rectangle')
    
    def highlight(self, enclosing_box):
        # get the enclosing box in the window relative coordinates
        x0, y0, x1, y1 = enclosing_box
        
        # set the highlight box, in window relative coordinates, used
        # for displaying the selection rectangle on the screen
        self.highlight_box = (x0, y0, x1, y1)
        
        # paint highlight box
        self.paint_manager.set_data(visible=True,
            coordinates=self.highlight_box,
            visual='highlight_rectangle')
        
        # convert the box coordinates in the data coordinate system
        x0, y0 = self.interaction_manager.get_processor('navigation').get_data_coordinates(x0, y0)
        x1, y1 = self.interaction_manager.get_processor('navigation').get_data_coordinates(x1, y1)
        
        self.highlighted((x0, y0, x1, y1))
        
    def highlighted(self, box):
        pass

    def cancel_highlight(self):
        # self.set_highlighted_spikes([])
        if self.highlight_box is not None:
            self.paint_manager.set_data(visible=False,
                visual='highlight_rectangle')
            self.highlight_box = None
    

class SpikyBindings(PlotBindings):
    def set_panning_keyboard(self):
        pass
        
    def set_zooming_keyboard(self):
        pass
    
    