import numpy as np
import numpy.random as rdn
import collections
import operator
import time
from matplotlib.path import Path

from galry import *
from common import *
from signals import *
from colors import COLORMAP


__all__ = ['FeatureView', 'FeatureNavigationBindings',
           'FeatureSelectionBindings', 'FeatureEventEnum']


VERTEX_SHADER = """
    // move the vertex to its position
    vec2 position = position0;
    
    vhighlight = highlight;
    cmap_vindex = cmap_index;
    vmask = mask;
    
    // selection
    if (selection > 0)
        gl_PointSize = 6.;
    else
        gl_PointSize = 3.;
"""
     
     
FRAGMENT_SHADER = """
    float index = %CMAP_OFFSET% + cmap_vindex * %CMAP_STEP%;
    out_color = texture1D(cmap, index);
        
    // toggle mask and masked points
    if ((vmask < 1) && (toggle_mask > 0)) {
        out_color.xyz = vec3(.5, .5, .5);
        // mask only for masked points in mask activated mode
        out_color.w = .5 + .5 * vmask;
    }
    else {
        
    }
    
    // highlight
    if (vhighlight > 0) {
        out_color.xyz = out_color.xyz + vec3(.5, .5, .5);
    }
"""

def polygon_contains_points(polygon, points):
    """Returns the points within a polygon.
    
    Arguments:
      * polygon: a Nx2 array with the coordinates of the polygon vertices.
      * points: a Nx2 array with the coordinates of the points.

    Returns:
      * arr: a Nx2 array of booleans with the belonging of every point to
        the inside of the polygon.
      
    """
    p = Path(polygon)
    if hasattr(p, 'contains_points'):
        return p.contains_points(points)
    else:
        import matplotlib.nxutils
        return matplotlib.nxutils.points_inside_poly(points, polygon)


class FeatureDataManager(Manager):
    projection = [None, None]
    
    # Initialization methods
    # ----------------------
    def set_data(self, features=None, fetdim=None, clusters=None,
                cluster_colors=None, clusters_unique=None,
                 masks=None, spike_ids=None):
        
        assert fetdim is not None
        
        self.nspikes, self.ndim = features.shape
        self.fetdim = fetdim
        self.nchannels = (self.ndim - 1) // self.fetdim
        self.npoints = features.shape[0]
        self.features = features
        # self.colormap = colormap
        
        # data organizer: reorder data according to clusters
        self.data_organizer = SpikeDataOrganizer(features,
                                                clusters=clusters,
                                                cluster_colors=cluster_colors,
                                                clusters_unique=clusters_unique,
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
        self.data = np.empty((self.nspikes, 2), dtype=np.float32)
        
        if self.projection[0] is None or self.projection[1] is None:
            self.set_projection(0, 0, 0, False)
            self.set_projection(1, 0, 1)
        else:
            self.set_projection(0, self.projection[0][0], self.projection[0][1], False)
            self.set_projection(1, self.projection[1][0], self.projection[1][1])
            
        # update the highlight manager
        self.highlight_manager.initialize()
        self.selection_manager.initialize()

    def set_projection(self, coord, channel, feature, do_update=True):
        """Set the projection axes."""
        i = channel * self.fetdim + feature
        self.full_masks = self.masks[:,channel]
        self.data[:, coord] = self.features_reordered[:, i].ravel()
        
        if do_update:
            self.data_normalizer = DataNormalizer(self.data)
            self.normalized_data = self.data_normalizer.normalize()
            self.projection[coord] = (channel, feature)
            # show the selection polygon only if the projection axes correspond
            self.selection_manager.set_selection_polygon_visibility(
              (self.projection[0] == self.selection_manager.projection[0]) & \
               (self.projection[1] == self.selection_manager.projection[1]))
        
    def automatic_projection(self):
        """Set the best projections depending on the selected clusters."""
        # TODO
        log_info("TODO: automatic projection")
        
        
class FeatureVisual(Visual):
    def initialize(self, npoints=None, #nclusters=None, 
                    position0=None,
                    mask=None,
                    cluster=None,
                    highlight=None,
                    selection=None,
                    cluster_colors=None,
                    # colormap=None
                    ):
        
        self.primitive_type = 'POINTS'
        self.size = npoints
        
        self.add_attribute("position0", vartype="float", ndim=2, data=position0)
        
        self.add_attribute("mask", vartype="float", ndim=1, data=mask)
        self.add_varying("vmask", vartype="float", ndim=1)
        
        
        # self.add_attribute("cluster", vartype="int", ndim=1, data=cluster)
        
        self.add_attribute("highlight", vartype="int", ndim=1, data=highlight)
        self.add_varying("vhighlight", vartype="int", ndim=1)
        
        self.add_uniform("toggle_mask", vartype="int", ndim=1, data=0)
        
        self.add_attribute("selection", vartype="int", ndim=1, data=selection)
        # color map for cluster colors, each spike has an index of the color
        # in the color map
        ncolors = COLORMAP.shape[0]
        ncomponents = COLORMAP.shape[1]
        
        # associate the cluster color to each spike
        # give the correct shape to cmap
        colormap = COLORMAP.reshape((1, ncolors, ncomponents))
        
        cmap_index = cluster_colors[cluster]
        
        self.add_texture('cmap', ncomponents=ncomponents, ndim=1, data=colormap)
        self.add_attribute('cmap_index', ndim=1, vartype='int', data=cmap_index)
        self.add_varying('cmap_vindex', vartype='int', ndim=1)
        
        dx = 1. / ncolors
        offset = dx / 2.
        
        global FRAGMENT_SHADER
        FRAGMENT_SHADER = FRAGMENT_SHADER.replace('%CMAP_OFFSET%', "%.5f" % offset)
        FRAGMENT_SHADER = FRAGMENT_SHADER.replace('%CMAP_STEP%', "%.5f" % dx)
        
        self.add_vertex_main(VERTEX_SHADER)
        self.add_fragment_main(FRAGMENT_SHADER)
        
        
class FeaturePaintManager(PaintManager):
    def update_points(self):
        self.set_data(position0=self.data_manager.normalized_data,
            mask=self.data_manager.full_masks, visual='features')
        
    def initialize(self):
        self.toggle_mask_value = False
        
        self.add_visual(FeatureVisual, name='features',
            npoints=self.data_manager.npoints,
            position0=self.data_manager.normalized_data,
            mask=self.data_manager.full_masks,
            cluster=self.data_manager.clusters_rel,
            highlight=self.highlight_manager.highlight_mask,
            selection=self.selection_manager.selection_mask,
            cluster_colors=self.data_manager.cluster_colors,
            # colormap=self.data_manager.colormap,
            )
        
    def update(self):

        cluster = self.data_manager.clusters_rel
        cluster_colors = self.data_manager.cluster_colors
        cmap_index = cluster_colors[cluster]
    
        self.set_data(visual='features', 
            size=self.data_manager.npoints,
            position0=self.data_manager.normalized_data,
            mask=self.data_manager.full_masks,
            # cluster=self.data_manager.clusters_rel,
            highlight=self.highlight_manager.highlight_mask,
            selection=self.selection_manager.selection_mask,
            cmap_index=cmap_index
            )

    def toggle_mask(self):
        self.toggle_mask_value = 1 - self.toggle_mask_value
        self.set_data(visual='features', toggle_mask=self.toggle_mask_value)
            

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

        indices = (
                  # (masks > 0) & \
                  (features[:,0] >= xmin) & (features[:,0] <= xmax) & \
                  (features[:,1] >= ymin) & (features[:,1] <= ymax))
        # absolute indices in the data
        spkindices = np.nonzero(indices)[0]
        spkindices = np.unique(spkindices)
        return spkindices
        
    def set_highlighted_spikes(self, spikes, do_emit=True):
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
            
            # emit the HighlightSpikes signal
            if do_emit:
                emit(self.parent, 'HighlightSpikes', spikes)
            
            self.paint_manager.set_data(
                highlight=self.highlight_mask, visual='features')
        
        # self.HighlightSpikes = QtCore.pyqtSignal(np.ndarray)
        
        # self.parent.emit(SpikySignals.HighlightSpikes, spikes)
        
        self.highlighted_spikes = spikes
        
    def highlighted(self, box):
        spikes = self.find_enclosed_spikes(box)
        self.set_highlighted_spikes(spikes)
        
    def cancel_highlight(self):
        super(FeatureHighlightManager, self).cancel_highlight()
        self.set_highlighted_spikes(np.array([]))


class FeatureSelectionManager(Manager):
    
    selection_polygon_color = (1., 1., 1., .5)
    points = np.zeros((100, 2))
    npoints = 0
    is_selection_pending = False
    projection = [None, None]
    
    def polygon(self):
        return self.points[:self.npoints + 2,:]
    
    def initialize(self):
        if not self.paint_manager.get_visual('selection_polygon'):
            self.paint_manager.add_visual(PlotVisual,
                                    position=self.points,
                                    color=self.selection_polygon_color,
                                    primitive_type='LINE_LOOP',
                                    visible=False,
                                    name='selection_polygon')
                
        self.selection_mask = np.zeros(self.data_manager.nspikes, dtype=np.int32)
        self.selected_spikes = []
        
    def set_selected_spikes(self, spikes, do_emit=True):
        """Update spike colors to mark transiently selected spikes with
        a special color."""
        if len(spikes) == 0:
            # do update only if there were previously selected spikes
            do_update = len(self.selected_spikes) > 0
            self.selection_mask[:] = 0
        else:
            do_update = True
            self.selection_mask[:] = 0
            self.selection_mask[spikes] = 1
        
        if do_update:
            
            # TODO
            # emit the SelectionSpikes signal
            # if do_emit:
                # emit(self.parent, 'SelectionSpikes', spikes)
            
            self.paint_manager.set_data(
                selection=self.selection_mask, visual='features')
        
        self.selected_spikes = spikes
    
    def find_enclosed_spikes(self, polygon=None):
        """Find the indices of the spikes inside the polygon (in 
        transformed coordinates)."""
        if polygon is None:
            polygon = self.polygon()
        features = self.data_manager.normalized_data
        masks = self.data_manager.full_masks
        # indices = (masks > 0) & polygon_contains_points(polygon, features)
        indices = polygon_contains_points(polygon, features)
        spkindices = np.nonzero(indices)[0]
        spkindices = np.unique(spkindices)
        return spkindices
   
    def select_spikes(self, polygon=None):
        """Select spikes enclosed in the selection polygon."""
        spikes = self.find_enclosed_spikes(polygon)
        self.set_selected_spikes(spikes)
   
    def add_point(self, point):
        """Add a point in the selection polygon."""
        point = self.interaction_manager.get_data_coordinates(*point)
        if not self.is_selection_pending:
            self.points = np.tile(point, (100, 1))
            self.paint_manager.set_data(
                    visible=True,
                    position=self.points,
                    visual='selection_polygon')
        self.is_selection_pending = True
        self.npoints += 1
        self.points[self.npoints,:] = point
        
    def point_pending(self, point):
        """A point is currently being positioned by the user. The polygon
        is updated in real time."""
        point = self.interaction_manager.get_data_coordinates(*point)
        if self.is_selection_pending:
            self.points[self.npoints + 1,:] = point
            self.paint_manager.set_data(
                    position=self.points,
                    visual='selection_polygon')
            # select spikes
            self.select_spikes()
        
    def set_selection_polygon_visibility(self, visible):
        if hasattr(self.paint_manager, 'ds_selection_polygon'):
            self.paint_manager.set_data(
                    visible=visible,
                    visual='selection_polygon')
        
    def end_point(self, point):
        """Terminate selection polygon."""
        point = self.interaction_manager.get_data_coordinates(*point)
        self.points[self.npoints + 1,:] = self.points[0,:]
        self.paint_manager.set_data(
                position=self.points,
                visual='selection_polygon')
        self.select_spikes()
        # record the projection axes corresponding to the current selection
        self.projection = list(self.data_manager.projection)
        self.is_selection_pending = False
        
    def cancel_selection(self):
        """Remove the selection polygon."""
        # hide the selection polygon
        if self.paint_manager.get_visual('selection_polygon').get('visible', None):
            self.paint_manager.set_data(visible=False,
                visual='selection_polygon')
        self.set_selected_spikes(np.array([]))
        self.is_selection_pending = False


class FeatureInteractionManager(InteractionManager):
    def initialize(self):
        # self.channel = 0
        # self.coordorder = [(0,1),(0,2),(1,2)]
        # self.icoord = 0
        self.constrain_navigation = False
        
    def process_none_event(self):
        super(FeatureInteractionManager, self).process_none_event()
        self.highlight_manager.cancel_highlight()
        
    def process_custom_event(self, event, parameter):
        # highlight
        if event == FeatureEventEnum.HighlightSpikeEvent:
            self.highlight_manager.highlight(parameter)
            self.cursor = cursors.CrossCursor
            
        # selection
        if event == FeatureEventEnum.SelectionPointPendingEvent:
            self.selection_manager.point_pending(parameter)
        if event == FeatureEventEnum.AddSelectionPointEvent:
            self.selection_manager.add_point(parameter)
        if event == FeatureEventEnum.EndSelectionPointEvent:
            self.selection_manager.end_point(parameter)
        if event == FeatureEventEnum.CancelSelectionPointEvent:
            self.selection_manager.cancel_selection()
          
        # select projection
        if event == FeatureEventEnum.SelectProjectionEvent:
            self.select_projection(parameter)
            
        # automatic projection
        if event == FeatureEventEnum.AutomaticProjectionEvent:
            self.data_manager.automatic_projection()
            
        # toggle mask
        if event == FeatureEventEnum.ToggleMaskEvent:
            self.paint_manager.toggle_mask()
            
        # select neighbor channel
        if event == FeatureEventEnum.SelectNeighborChannelEvent:
            # print self.data_manager.projection
            coord, channel_dir = parameter
            # current channel and feature in the given coordinate
            proj = self.data_manager.projection[coord]
            if proj is None:
                proj = (0, coord)
            channel, feature = proj
            # next or previous channel
            channel = np.mod(channel + channel_dir, self.data_manager.nchannels)
            # select projection
            # self.select_projection((coord, channel, feature))
            emit(self.parent, 'ProjectionToChange', coord, channel, feature)
            
        # select neighbor feature
        if event == FeatureEventEnum.SelectNeighborFeatureEvent:
            # print self.data_manager.projection
            coord, feature_dir = parameter
            # current channel and feature in the given coordinate
            proj = self.data_manager.projection[coord]
            if proj is None:
                proj = (0, coord)
            channel, feature = proj
            # next or previous feature
            feature = np.mod(feature + feature_dir, self.data_manager.fetdim)
            # select projection
            # self.select_projection((coord, channel, feature))
            emit(self.parent, 'ProjectionToChange', coord, channel, feature)
            
            
    def select_projection(self, parameter):
        """Select a projection for the given coordinate."""
        self.data_manager.set_projection(*parameter)  # coord, channel, feature
        self.paint_manager.update_points()
        self.paint_manager.updateGL()


FeatureEventEnum = enum(
    "HighlightSpikeEvent",
    
    "AddSelectionPointEvent",
    "SelectionPointPendingEvent",
    "EndSelectionPointEvent",
    "CancelSelectionPointEvent",
    
    "ToggleMaskEvent",
    
    "SelectProjectionEvent",
    "AutomaticProjectionEvent",
    
    "SelectNeighborChannelEvent",
    "SelectNeighborFeatureEvent",
    )
        
        
# Bindings
# --------
class FeatureBindings(SpikyDefaultBindingSet):
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
        
    def set_toggle_mask(self):
        self.set(UserActions.KeyPressAction,
                 FeatureEventEnum.ToggleMaskEvent,
                 key=QtCore.Qt.Key_T)
        
    def set_neighbor_channel(self):
        # select previous/next channel for coordinate 0
        self.set(UserActions.KeyPressAction, FeatureEventEnum.SelectNeighborChannelEvent,
                 key=QtCore.Qt.Key_Up, key_modifier=QtCore.Qt.Key_Control,
                 param_getter=lambda p: (0, -1))
        self.set(UserActions.KeyPressAction, FeatureEventEnum.SelectNeighborChannelEvent,
                 key=QtCore.Qt.Key_Down, key_modifier=QtCore.Qt.Key_Control,
                 param_getter=lambda p: (0, 1))
                 
        # select previous/next channel for coordinate 1
        self.set(UserActions.KeyPressAction, FeatureEventEnum.SelectNeighborChannelEvent,
                 key=QtCore.Qt.Key_Up, key_modifier=QtCore.Qt.Key_Shift,
                 param_getter=lambda p: (1, -1))
        self.set(UserActions.KeyPressAction, FeatureEventEnum.SelectNeighborChannelEvent,
                 key=QtCore.Qt.Key_Down, key_modifier=QtCore.Qt.Key_Shift,
                 param_getter=lambda p: (1, 1))
        
    def set_neighbor_feature(self):
        # select previous/next feature for coordinate 0
        self.set(UserActions.KeyPressAction, FeatureEventEnum.SelectNeighborFeatureEvent,
                 key=QtCore.Qt.Key_Left, key_modifier=QtCore.Qt.Key_Control,
                 param_getter=lambda p: (0, -1))
        self.set(UserActions.KeyPressAction, FeatureEventEnum.SelectNeighborFeatureEvent,
                 key=QtCore.Qt.Key_Right, key_modifier=QtCore.Qt.Key_Control,
                 param_getter=lambda p: (0, 1))
                 
        # select previous/next feature for coordinate 1
        self.set(UserActions.KeyPressAction, FeatureEventEnum.SelectNeighborFeatureEvent,
                 key=QtCore.Qt.Key_Left, key_modifier=QtCore.Qt.Key_Shift,
                 param_getter=lambda p: (1, -1))
        self.set(UserActions.KeyPressAction, FeatureEventEnum.SelectNeighborFeatureEvent,
                 key=QtCore.Qt.Key_Right, key_modifier=QtCore.Qt.Key_Shift,
                 param_getter=lambda p: (1, 1))
        
        
class FeatureNavigationBindings(FeatureBindings):
    def extend(self):
        self.set_highlight()
        self.set_toggle_mask()
        self.set_neighbor_channel()
        self.set_neighbor_feature()


class FeatureSelectionBindings(FeatureBindings):
    def set_selection(self):
        # selection
        self.set(UserActions.MouseMoveAction,
                 FeatureEventEnum.SelectionPointPendingEvent,
                 param_getter=lambda p: (p["mouse_position"][0],
                                         p["mouse_position"][1],))
        self.set(UserActions.LeftButtonClickAction,
                 FeatureEventEnum.AddSelectionPointEvent,
                 param_getter=lambda p: (p["mouse_press_position"][0],
                                         p["mouse_press_position"][1],))
        self.set(UserActions.RightButtonClickAction,
                 FeatureEventEnum.EndSelectionPointEvent,
                 param_getter=lambda p: (p["mouse_press_position"][0],
                                         p["mouse_press_position"][1],))
        self.set(UserActions.DoubleClickAction,
                 FeatureEventEnum.CancelSelectionPointEvent,
                 param_getter=lambda p: (p["mouse_press_position"][0],
                                         p["mouse_press_position"][1],))
    
    def extend(self):
        self.set_highlight()
        self.set_toggle_mask()
        self.set_neighbor_channel()
        self.set_neighbor_feature()
        
        self.set_base_cursor(cursors.CrossCursor)
        self.set_selection()
     
     
class FeatureView(GalryWidget):
    def initialize(self):
        self.set_bindings(FeatureNavigationBindings, FeatureSelectionBindings)
        self.set_companion_classes(
                paint_manager=FeaturePaintManager,
                data_manager=FeatureDataManager,
                highlight_manager=FeatureHighlightManager,
                selection_manager=FeatureSelectionManager,
                interaction_manager=FeatureInteractionManager)
        # connect the AutomaticProjection signal to the
        # AutomaticProjectionEvent
        self.connect_events(SIGNALS.AutomaticProjection,
                            FeatureEventEnum.AutomaticProjectionEvent)
    
    def set_data(self, *args, **kwargs):
        self.data_manager.set_data(*args, **kwargs)
        if self.initialized:
            log_debug("Updating data for features")
            self.paint_manager.update()
            self.updateGL()
        else:
            log_debug("Initializing data for features")
        
    # Signals-related methods
    # -----------------------
    def highlight_spikes(self, spikes):
        self.highlight_manager.set_highlighted_spikes(spikes, False)
        self.updateGL()

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
    
    
    