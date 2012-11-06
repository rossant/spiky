import numpy as np
import numpy.random as rdn
import collections
import operator
import time

from galry import *
from common import *

# import colors
# from probes import Probe
# from waveformtest import *


VERTEX_SHADER = """
    // move the vertex to its position
    vec2 position = position0;
    
    // compute the color: cluster color and mask for the transparency
    varying_color.xyz = cluster_colors[int(cluster)];
    varying_color.w = mask;
    
    // highlighting: change color, not transparency
    if (highlight > 0)
        varying_color = vec4(1, 1, 1, varying_color.w);
        
    gl_PointSize = 3.;
"""
     
     
FRAGMENT_SHADER = """
    out_color = varying_color;
"""


class FeatureDataManager(object):
    # Initialization methods
    # ----------------------
    def set_data(self, features, fetdim=None, clusters=None, cluster_colors=None,
                 masks=None, spike_ids=None):
        
        assert fetdim is not None
        
        self.nspikes, self.ndim = features.shape
        self.fetdim = fetdim
        self.nchannels = (self.ndim - 1) // self.fetdim
        self.npoints = features.shape[0]
        self.features = features
        
        # data organizer: reorder data according to clusters
        self.data_organizer = SpikeDataOrganizer(features,
                                                clusters=clusters,
                                                cluster_colors=cluster_colors,
                                                masks=masks,
                                                nchannels=self.nchannels,
                                                spike_ids=spike_ids)
        
        # get reordered data
        self.permutation = self.data_organizer.permutation
        self.features_reordered = self.data_organizer.data_reordered
        self.nclusters = self.data_organizer.nclusters
        self.clusters = self.data_organizer.clusters
        self.masks = self.data_organizer.masks
        self.cluster_colors = self.data_organizer.cluster_colors
        self.clusters_unique = self.data_organizer.clusters_unique
        self.clusters_rel = self.data_organizer.clusters_rel
        self.cluster_sizes = self.data_organizer.cluster_sizes
        self.cluster_sizes_cum = self.data_organizer.cluster_sizes_cum
        self.cluster_sizes_dict = self.data_organizer.cluster_sizes_dict
        
        # self.full_clusters = self.clusters
        
        # prepare GPU data
        self.set_projection()
        
        # update the highlight manager
        self.highlight_manager.initialize()

    def set_projection(self, channel0=0, channel1=0, coord0=0, coord1=1):
        
        # in GPU memory, X coordinates are always between -1 and 1
        i0 = channel0 * self.fetdim + coord0
        i1 = channel1 * self.fetdim + coord1
        
        # copy each color as many times as there are spikes in each cluster
        self.colors = np.empty((self.nspikes, 4), dtype=np.float32)
        colors = np.repeat(self.cluster_colors, self.cluster_sizes, axis=0)
        self.colors[:,:3] = colors
        # add transparency: the max of transparency between channel0 and 1
        self.full_masks = np.max(self.masks[:,np.array([channel0, channel1])], 1)
        self.colors[:,3] = self.full_masks
        
        # feature data
        X = self.features_reordered[:,i0]
        Y = self.features_reordered[:,i1]
        
        # create a Nx2 array with all coordinates
        self.data = np.empty((X.size, 2), dtype=np.float32)
        self.data[:,0] = X.ravel()
        self.data[:,1] = Y.ravel()
        
        # initialize the normalizer
        self.data_normalizer = DataNormalizer(self.data)
        self.normalized_data = self.data_normalizer.normalize()
        
        
class FeatureTemplate(DefaultTemplate):
    def initialize(self, npoints=None, nclusters=None, **kwargs):
        self.primitive_type =PrimitiveType.Points
        self.size = npoints
        self.npoints = npoints
        self.nclusters = nclusters
        
        self.add_attribute("position0", vartype="float", ndim=2)
        self.add_attribute("mask", vartype="float", ndim=1)
        self.add_attribute("cluster", vartype="int", ndim=1)
        self.add_attribute("highlight", vartype="int", ndim=1)
        
        self.add_uniform("cluster_colors", vartype="float", ndim=3,
            size=self.nclusters)
        
        self.add_varying("varying_color", vartype="float", ndim=4)
        
        self.add_vertex_main(VERTEX_SHADER)
        self.add_fragment_main(FRAGMENT_SHADER)
        
        self.initialize_default(**kwargs)
        
        
class FeaturePaintManager(PaintManager):
    def initialize(self):
        self.ds = self.create_dataset(FeatureTemplate,
            npoints=self.data_manager.npoints,
            nclusters=self.data_manager.nclusters,
            position0=self.data_manager.normalized_data,
            mask=self.data_manager.full_masks,
            cluster=self.data_manager.clusters_rel,
            highlight=self.highlight_manager.highlight_mask,
            cluster_colors=self.data_manager.cluster_colors)
        
    def update_points(self):
        self.set_data(position0=self.data_manager.normalized_data,
            mask=self.data_manager.full_masks, dataset=self.ds)
        
        
class FeatureHighlightManager(HighlightManager):
    def initialize(self):
        super(FeatureHighlightManager, self).initialize()
        self.highlight_mask = np.zeros(self.data_manager.nspikes, dtype=np.int32)
        self.highlighted_spikes = []
        
    def find_enclosed_spikes(self, enclosing_box):
        x0, y0, x1, y1 = enclosing_box
        
        # press_position
        xp, yp = x0, y0

        # reorder
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)

        features = self.data_manager.normalized_data
        masks = self.data_manager.full_masks

        indices = ((masks > 0) & \
                  (features[:,0] >= xmin) & (features[:,0] <= xmax) & \
                  (features[:,1] >= ymin) & (features[:,1] <= ymax))
        # absolute indices in the data
        spkindices = np.nonzero(indices)[0]
        spkindices = np.unique(spkindices)
        return spkindices
        
    def set_highlighted_spikes(self, spikes):
        """Update spike colors to mark transiently selected spikes with
        a special color."""
        if len(spikes) == 0:
            # do update only if there were previously selected spikes
            do_update = len(self.highlighted_spikes) > 0
            self.highlight_mask[:] = 0
        else:
            do_update = True
            self.highlight_mask[:] = 0
            self.highlight_mask[spikes] = 1
        
        if do_update:
            self.paint_manager.set_data(
                highlight=self.highlight_mask, dataset=self.paint_manager.ds)
        
        self.highlighted_spikes = spikes
        
    def highlighted(self, box):
        spikes = self.find_enclosed_spikes(box)
        self.set_highlighted_spikes(spikes)
        
    def cancel_highlight(self):
        super(FeatureHighlightManager, self).cancel_highlight()
        self.set_highlighted_spikes([])
        
        
class FeatureInteractionManager(InteractionManager):
    def initialize(self):
        self.channel = 0
        self.coordorder = [(0,1),(0,2),(1,2)]
        self.icoord = 0
        self.constrain_navigation = False
        
    def process_none_event(self):
        super(FeatureInteractionManager, self).process_none_event()
        self.highlight_manager.cancel_highlight()
        
    def process_custom_event(self, event, parameter):
        if event == FeatureEventEnum.ChangeProjection:
            self.change_projection(parameter)
            
        # transient selection
        if event == FeatureEventEnum.HighlightSpikeEvent:
            self.highlight_manager.highlight(parameter)
            self.cursor = cursors.CrossCursor
          
    def change_projection(self, dir=1):
        self.icoord += dir
        nchannels = self.data_manager.nchannels
        if self.icoord == 3:
            self.icoord = 0
            self.channel = np.mod(self.channel + 1, nchannels)
        elif self.icoord == -1:
            self.icoord = 2
            self.channel = np.mod(self.channel - 1, nchannels)
        c0, c1 = self.coordorder[self.icoord]
        self.data_manager.set_projection(self.channel, self.channel,
                c0, c1)
        self.paint_manager.update_points()
        
        
FeatureEventEnum = enum(
    "ChangeProjection",
    "HighlightSpikeEvent",
    )
        
        
class FeaturesBindings(DefaultBindingSet):
    def set_highlight(self):
        # highlight
        self.set(UserActions.MiddleButtonMouseMoveAction,
                 FeatureEventEnum.HighlightSpikeEvent,
                 param_getter=lambda p: (p["mouse_press_position"][0],
                                         p["mouse_press_position"][1],
                                         p["mouse_position"][0],
                                         p["mouse_position"][1]))
        
        self.set(UserActions.LeftButtonMouseMoveAction,
                 FeatureEventEnum.HighlightSpikeEvent,
                 key_modifier=QtCore.Qt.Key_Control,
                 param_getter=lambda p: (p["mouse_press_position"][0],
                                         p["mouse_press_position"][1],
                                         p["mouse_position"][0],
                                         p["mouse_position"][1]))
      
    def extend(self):
        self.set_highlight()
        
        # change projection
        self.set(UserActions.KeyPressAction, FeatureEventEnum.ChangeProjection,
                 key=QtCore.Qt.Key_F, param_getter=lambda p: -1)
        self.set(UserActions.KeyPressAction, FeatureEventEnum.ChangeProjection,
                 key=QtCore.Qt.Key_G, param_getter=lambda p: 1)
     
     
class FeatureView(GalryWidget):
    def initialize(self):
        self.set_bindings(FeaturesBindings)
        self.set_companion_classes(
                paint_manager=FeaturePaintManager,
                data_manager=FeatureDataManager,
                highlight_manager=FeatureHighlightManager,
                interaction_manager=FeatureInteractionManager)
    
    def set_data(self, *args, **kwargs):
        self.data_manager.set_data(*args, **kwargs)
        
        

# if __name__ == '__main__':

    # spikes = 1000
    # nclusters = 5
    # nfet = 3 * 32 + 1
    
    # # features = np.load("data/fet%d.npy" % spikes)
    # # data = np.load("data/data%d.npz" % spikes)
    # # clusters = data["clusters"]
    # # masks = data["masks"]
    
    # # # select largest clusters
    # # c = collections.Counter(clusters)
    # # best_clusters = np.array(map(operator.itemgetter(0), c.most_common(nclusters)))
    # # indices = np.zeros(spikes, dtype=bool)
    # # for i in xrange(nclusters):
        # # indices = indices | (clusters == best_clusters[-i])
        
    # # for testing, we just use the first colors for our clusters
    # cluster_colors = np.array(colors.generate_colors(nclusters), dtype=np.float32)
        
    # # features = features[indices,:]
    # # clusters = clusters[indices]
    # # masks = masks[indices,:]
    
    
    