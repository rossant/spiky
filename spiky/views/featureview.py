import numpy as np
import numpy.random as rdn
import collections
import operator
import time
from matplotlib.path import Path

from galry import *
from common import HighlightManager, SpikyBindings, SpikeDataOrganizer
from widgets import VisualizationWidget
import spiky.colors as scolors
import spiky
import spiky.tools as stools
import spiky.signals as ssignals


__all__ = ['FeatureView', 'FeatureNavigationBindings',
           'FeatureSelectionBindings', # 'FeatureEnum'
           'FeatureWidget',
           ]


VERTEX_SHADER = """
    // move the vertex to its position
    vec2 position = position0;
    
    vhighlight = highlight;
    cmap_vindex = cmap_index;
    vmask = mask;
    vselection = selection;
        
    if ((highlight > 0) || (selection > 0))
        gl_PointSize = 5.;
    else
        gl_PointSize = 3.;
"""
     
     
FRAGMENT_SHADER = """
    float index = %CMAP_OFFSET% + cmap_vindex * %CMAP_STEP%;
    if ((vhighlight > 0) || (vselection > 0)) {
        out_color = texture1D(hcmap, index);
    }
    else {
        out_color = texture1D(cmap, index);
    }
    out_color.w = .75;
        
    // toggle mask and masked points
    if ((vmask == 0) && (toggle_mask > 0)) {
        if (vhighlight > 0) {
            out_color.xyz = vec3(.95, .95, .95);
        }
        else {
            out_color.xyz = vec3(.75, .75, .75);
        }
        
        // mask only for masked points in mask activated mode
        out_color.w = .5 + .25 * vmask;
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
                nchannels=None, nextrafet=None,
                cluster_colors=None, clusters_unique=None,
                 masks=None, spike_ids=None):
        
        assert fetdim is not None
        
        self.nspikes, self.ndim = features.shape
        self.fetdim = fetdim
        self.nchannels = nchannels
        self.nextrafet = nextrafet
        self.npoints = features.shape[0]
        self.features = features
        self.spike_ids = spike_ids
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
        # self.permutation = self.data_organizer.permutation
        self.features_reordered = self.data_organizer.data_reordered
        self.nclusters = self.data_organizer.nclusters
        self.clusters = self.data_organizer.clusters
        self.masks = self.data_organizer.masks
        self.cluster_colors = self.data_organizer.cluster_colors
        self.clusters_unique = self.data_organizer.clusters_unique
        self.clusters_rel = self.data_organizer.clusters_rel
        self.cluster_sizes = self.data_organizer.cluster_sizes
        # self.cluster_sizes_cum = self.data_organizer.cluster_sizes_cum
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
        if channel < self.nchannels:
            i = channel * self.fetdim + feature
            self.full_masks = self.masks[:,channel]
        # handle extra feature, with channel being directly the feature index
        else:
            # print self.nchannels * self.fetdim + self.nextrafet - 1, channel
            i = min(self.nchannels * self.fetdim + self.nextrafet - 1,
                    channel - self.nchannels + self.nchannels * self.fetdim)
            # print channel, i
            # i = channel
        self.data[:, coord] = self.features_reordered[:, i].ravel()
        
        if do_update:
            # self.data_normalizer = DataNormalizer(self.data)
            # self.normalized_data = self.data_normalizer.normalize(symmetric=True)
            self.normalized_data = self.data
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
        self.add_varying("vselection", vartype="int", ndim=1)
        
        # color map for cluster colors, each spike has an index of the color
        # in the color map
        ncolors = scolors.COLORMAP.shape[0]
        ncomponents = scolors.COLORMAP.shape[1]
        
        # associate the cluster color to each spike
        # give the correct shape to cmap
        colormap = scolors.COLORMAP.reshape((1, ncolors, ncomponents))
        hcolormap = scolors.HIGHLIGHT_COLORMAP.reshape((1, ncolors, ncomponents))
        
        cmap_index = cluster_colors[cluster]
        
        self.add_texture('cmap', ncomponents=ncomponents, ndim=1, data=colormap)
        self.add_texture('hcmap', ncomponents=ncomponents, ndim=1, data=hcolormap)
        
        self.add_attribute('cmap_index', ndim=1, vartype='int', data=cmap_index)
        self.add_varying('cmap_vindex', vartype='int', ndim=1)
        
        dx = 1. / ncolors
        offset = dx / 2.
        
        global FRAGMENT_SHADER
        FRAGMENT_SHADER = FRAGMENT_SHADER.replace('%CMAP_OFFSET%', "%.5f" % offset)
        FRAGMENT_SHADER = FRAGMENT_SHADER.replace('%CMAP_STEP%', "%.5f" % dx)
        
        # self.add_fragment_header(FSH)

        self.add_vertex_main(VERTEX_SHADER)
        self.add_fragment_main(FRAGMENT_SHADER)
        
        
class FeaturePaintManager(PlotPaintManager):
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
        
        self.add_visual(AxesVisual, name='grid')
        
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
        self.spike_ids = self.data_manager.spike_ids
        
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
        # return self.spike_ids[spkindices]
        
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
            # from absolue indices to relative indices
            spikes_rel = np.digitize(spikes, self.spike_ids) - 1
            self.highlight_mask[spikes_rel] = 1
        
        if do_update:
            
            # emit the HighlightSpikes signal
            if do_emit:
                ssignals.emit(self.parent, 'HighlightSpikes', spikes)
                    # self.spike_ids[np.array(spikes, dtype=np.int32)])
            
            self.paint_manager.set_data(
                highlight=self.highlight_mask, visual='features')
        
        # self.HighlightSpikes = QtCore.pyqtSignal(np.ndarray)
        
        # self.parent.emit(SpikySignals.HighlightSpikes, spikes)
        
        self.highlighted_spikes = spikes
        
    def highlighted(self, box):
        spikes = self.find_enclosed_spikes(box)
        # from relative indices to absolute indices
        self.set_highlighted_spikes(self.spike_ids[spikes])
        
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
            
            # emit the SelectionSpikes signal
            if do_emit:
                if len(spikes) > 0:
                    aspikes = self.data_manager.spike_ids[spikes]
                else:
                    aspikes = spikes
                ssignals.emit(self.parent, 'SelectSpikes', aspikes)
            
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
        point = self.interaction_manager.get_processor('navigation').get_data_coordinates(*point)
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
        point = self.interaction_manager.get_processor('navigation').get_data_coordinates(*point)
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
        point = self.interaction_manager.get_processor('navigation').get_data_coordinates(*point)
        # record the last point in the selection polygon
        self.points[self.npoints + 1,:] = point
        self.points[self.npoints + 2,:] = self.points[0,:]
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


class FeatureInteractionManager(PlotInteractionManager):
    def initialize(self):
        self.constrain_navigation = False
        
    def cancel_highlight(self, parameter):
        self.highlight_manager.cancel_highlight()
        
    def highlight_spike(self, parameter):
        self.highlight_manager.highlight(parameter)
        self.cursor = 'CrossCursor'
        
    def selection_point_pending(self, parameter):
        self.selection_manager.point_pending(parameter)
        
    def selection_add_point(self, parameter):
        self.selection_manager.add_point(parameter)
        
    def selection_end_point(self, parameter):
        self.selection_manager.end_point(parameter)
        
    def selection_cancel(self, parameter):
        self.selection_manager.cancel_selection()

    def automatic_projection(self, parameter):
        self.data_manager.automatic_projection()

    def toggle_mask(self, parameter):
        self.paint_manager.toggle_mask()
        
    def initialize(self):
    
        self.register(None, self.cancel_highlight)
        self.register('HighlightSpike', self.highlight_spike)
        self.register('SelectionPointPending', self.selection_point_pending)
        
        self.register('AddSelectionPoint', self.selection_add_point)
        self.register('EndSelectionPoint', self.selection_end_point)
        self.register('CancelSelectionPoint', self.selection_cancel)
        
        self.register('SelectProjection', self.select_projection)
        self.register('AutomaticProjection', self.automatic_projection)

        self.register('ToggleMask', self.toggle_mask)

        self.register('SelectNeighborChannel', self.select_neighbor_channel)
        self.register('SelectNeighborFeature', self.select_neighbor_feature)
        
    def select_neighbor_channel(self, parameter):
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
        ssignals.emit(self.parent, 'ProjectionToChange', coord, channel, feature)
            
    def select_neighbor_feature(self, parameter):
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
        ssignals.emit(self.parent, 'ProjectionToChange', coord, channel, feature)
            
    def select_projection(self, parameter):
        """Select a projection for the given coordinate."""
        self.data_manager.set_projection(*parameter)  # coord, channel, feature
        self.paint_manager.update_points()
        self.paint_manager.updateGL()


# Bindings
# --------
class FeatureBindings(SpikyBindings):
    def set_zoombox_keyboard(self):
        """Set zoombox bindings with the keyboard."""
        self.set('MiddleClickMove', 'ZoomBox',
                    # key_modifier='Shift',
                    param_getter=lambda p: (p["mouse_press_position"][0],
                                            p["mouse_press_position"][1],
                                            p["mouse_position"][0],
                                            p["mouse_position"][1]))
                                            
    def set_highlight(self):
        # highlight
        # self.set('MiddleClickMove',
                 # 'HighlightSpike',
                 # param_getter=lambda p: (p["mouse_press_position"][0],
                                         # p["mouse_press_position"][1],
                                         # p["mouse_position"][0],
                                         # p["mouse_position"][1]))
        
        self.set('LeftClickMove',
                 'HighlightSpike',
                 key_modifier='Control',
                 param_getter=lambda p: (p["mouse_press_position"][0],
                                         p["mouse_press_position"][1],
                                         p["mouse_position"][0],
                                         p["mouse_position"][1]))
        
    def set_toggle_mask(self):
        self.set('KeyPress',
                 'ToggleMask',
                 key='T')
        
    def set_neighbor_channel(self):
        # select previous/next channel for coordinate 0
        self.set('KeyPress', 'SelectNeighborChannel',
                 key='Up', description='X-', key_modifier='Control',
                 param_getter=lambda p: (0, -1))
        self.set('KeyPress', 'SelectNeighborChannel',
                 key='Down', description='X+', key_modifier='Control',
                 param_getter=lambda p: (0, 1))
                 
        # select previous/next channel for coordinate 1
        self.set('KeyPress', 'SelectNeighborChannel',
                 key='Up', description='Y-', key_modifier='Shift',
                 param_getter=lambda p: (1, -1))
        self.set('KeyPress', 'SelectNeighborChannel',
                 key='Down', description='Y+', key_modifier='Shift',
                 param_getter=lambda p: (1, 1))
                 
        # The same, but with the mosue wheel
        # select previous/next channel for coordinate 0
        self.set('Wheel', 'SelectNeighborChannel',
                 key_modifier='Control',
                 param_getter=lambda p: (0, -int(np.sign(p['wheel']))))
        # self.set('Wheel', 'SelectNeighborChannel',
                 # key_modifier='Control',
                 # param_getter=lambda p: (0, 1))
                 
        # select previous/next channel for coordinate 1
        self.set('Wheel', 'SelectNeighborChannel',
                 key_modifier='Shift',
                 param_getter=lambda p: (1, -int(np.sign(p['wheel']))))
        # self.set('Wheel', 'SelectNeighborChannel',
                 # key_modifier='Shift',
                 # param_getter=lambda p: (1, 1))
        
    def set_neighbor_feature(self):
        # select previous/next feature for coordinate 0
        self.set('KeyPress', 'SelectNeighborFeature',
                 key='Left', description='X-', key_modifier='Control',
                 param_getter=lambda p: (0, -1))
        self.set('KeyPress', 'SelectNeighborFeature',
                 key='Right', description='X+', key_modifier='Control',
                 param_getter=lambda p: (0, 1))
                 
        # select previous/next feature for coordinate 1
        self.set('KeyPress', 'SelectNeighborFeature',
                 key='Left', description='Y-', key_modifier='Shift',
                 param_getter=lambda p: (1, -1))
        self.set('KeyPress', 'SelectNeighborFeature',
                 key='Right', description='Y+', key_modifier='Shift',
                 param_getter=lambda p: (1, 1))
        
    def initialize(self):
        super(FeatureBindings, self).initialize()
        self.set('KeyPress', 'SwitchInteractionMode', key='N')
        
        
class FeatureNavigationBindings(FeatureBindings):
    def initialize(self):
        self.set_highlight()
        self.set_toggle_mask()
        self.set_neighbor_channel()
        self.set_neighbor_feature()


class FeatureSelectionBindings(FeatureBindings):
    def set_selection(self):
        # selection
        self.set('Move',
                 'SelectionPointPending',
                 param_getter=lambda p: (p["mouse_position"][0],
                                         p["mouse_position"][1],))
        self.set('LeftClick',
                 'AddSelectionPoint',
                 param_getter=lambda p: (p["mouse_press_position"][0],
                                         p["mouse_press_position"][1],))
        self.set('RightClick',
                 'EndSelectionPoint',
                 param_getter=lambda p: (p["mouse_press_position"][0],
                                         p["mouse_press_position"][1],))
        self.set('DoubleClick',
                 'CancelSelectionPoint',
                 param_getter=lambda p: (p["mouse_press_position"][0],
                                         p["mouse_press_position"][1],))
    
    def initialize(self):
        self.set_highlight()
        self.set_toggle_mask()
        self.set_neighbor_channel()
        self.set_neighbor_feature()
        
        self.set_base_cursor('CrossCursor')
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
        # AutomaticProjection
        self.connect_events(ssignals.SIGNALS.AutomaticProjection,
                            'AutomaticProjection')
    
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

        
class FeatureWidget(VisualizationWidget):
    def create_view(self, dh):
        self.dh = dh
        self.view = FeatureView(getfocus=False)
        self.view.set_data(fetdim=self.dh.fetdim,
                      features=self.dh.features,
                      clusters=self.dh.clusters,
                      nchannels=self.dh.nchannels,
                      nextrafet=self.dh.nextrafet,
                      # colormap=self.dh.colormap,
                      cluster_colors=self.dh.cluster_colors,
                      masks=self.dh.masks,
                      spike_ids=self.dh.spike_ids)
        return self.view
        
    def update_view(self, dh=None):
        if dh is not None:
            self.dh = dh
        self.view.set_data(fetdim=self.dh.fetdim,
                      features=self.dh.features,
                      clusters=self.dh.clusters,
                      nchannels=self.dh.nchannels,
                      nextrafet=self.dh.nextrafet,
                      cluster_colors=self.dh.cluster_colors,
                      clusters_unique=self.dh.clusters_unique,
                      masks=self.dh.masks,
                      spike_ids=self.dh.spike_ids)
        self.update_nspikes_viewer(self.dh.nspikes, 0)
        self.update_feature_widget()

    def create_toolbar(self):
        toolbar = QtGui.QToolBar(self)
        toolbar.setObjectName("toolbar")
        toolbar.setIconSize(QtCore.QSize(32, 32))
        
        # navigation toolbar
        toolbar.addAction(spiky.get_icon('hand'), "Move (press I to switch)",
            self.set_navigation)
        toolbar.addAction(spiky.get_icon('selection'), "Selection (press I to switch)",
            self.set_selection)
            
        # toolbar.addSeparator()
            
        # autoprojection
        # toolbar.addAction(spiky.get_icon('hand'), "Move (press I to switch)",
            # self.main_window.autoproj_action)
        
        toolbar.addSeparator()
        
        return toolbar
        
    def initialize_connections(self):
        ssignals.SIGNALS.ProjectionChanged.connect(self.slotProjectionChanged, QtCore.Qt.UniqueConnection)
        ssignals.SIGNALS.ClusterSelectionChanged.connect(self.slotClusterSelectionChanged, QtCore.Qt.UniqueConnection)
        ssignals.SIGNALS.HighlightSpikes.connect(self.slotHighlightSpikes, QtCore.Qt.UniqueConnection)
        
    def slotHighlightSpikes(self, sender, spikes):
        self.update_nspikes_viewer(self.dh.nspikes, len(spikes))
        if sender != self.view:
            self.view.highlight_spikes(spikes)
        
    def slotClusterSelectionChanged(self, sender, clusters):
        self.update_view()
        
    def slotProjectionChanged(self, sender, coord, channel, feature):
        """Process the ProjectionChanged signal."""
        
        # feature == -1 means that it should be automatically selected as
        # a function of the current projection
        if feature < 0:
            # current channel and feature in the other coordinate
            other_channel, other_feature = self.view.data_manager.projection[1 - coord]
            fetdim = self.dh.fetdim
            # first dimension: we force to 0
            if coord == 0:
                feature = 0
            # other dimension: 0 if different channel, or next feature if the same
            # channel
            else:
                # same channel case
                if channel == other_channel:
                    feature = np.mod(other_feature + 1, fetdim)
                # different channel case
                else:
                    feature = 0
        
        # print sender
        log_debug("Projection changed in coord %s, channel=%d, feature=%d" \
            % (('X', 'Y')[coord], channel, feature))
        # record the new projection
        self.projection[coord] = (channel, feature)
        
        # prevent the channelbox to raise signals when we change its state
        # programmatically
        self.channel_box[coord].blockSignals(True)
        # update the channel box
        self.set_channel_box(coord, channel)
        # update the feature button
        self.set_feature_button(coord, feature)
        # reactive signals for the channel box
        self.channel_box[coord].blockSignals(False)
        
        # update the view
        self.view.process_interaction('SelectProjection', 
                                      (coord, channel, feature))
        
    def set_channel_box(self, coord, channel):
        """Select the adequate line in the channel selection combo box."""
        self.channel_box[coord].setCurrentIndex(channel)
        
    def set_feature_button(self, coord, feature):
        """Push the corresponding button."""
        if feature < len(self.feature_buttons[coord]):
            self.feature_buttons[coord][feature].setChecked(True)
        
    def select_feature(self, coord, fet=0):
        """Select channel coord, feature fet."""
        # raise the ProjectionToChange signal, and keep the previously
        # selected channel
        ssignals.emit(self, "ProjectionToChange", coord, self.projection[coord][0], fet)
        
    def select_channel(self, channel, coord=0):
        """Raise the ProjectionToChange signal when the channel is changed."""
        # print type(channel)
        # return
        # if isinstance(channel, basestring):
        if channel.startswith('Extra'):
            channel = channel[6:]
            extra = True
        else:
            extra = False
        # try:
        channel = int(channel)
        if extra:
            channel += self.dh.nchannels #* self.dh.fetdim
        ssignals.emit(self, "ProjectionToChange", coord, channel,
                 self.projection[coord][1])
        # except:
            # log_warn("'%s' is not a valid channel." % channel)
        
    def _select_feature_getter(self, coord, fet):
        """Return the callback function for the feature selection."""
        return lambda *args: self.select_feature(coord, fet)
        
    def _select_channel_getter(self, coord):
        """Return the callback function for the channel selection."""
        return lambda channel: self.select_channel(channel, coord)
        
    def create_feature_widget(self, coord=0):
        # coord => (channel, feature)
        self.projection = [(0, 0), (0, 1)]
        
        gridLayout = QtGui.QGridLayout()
        gridLayout.setSpacing(0)
        # HACK: pyside does not have this function
        if hasattr(gridLayout, 'setMargin'):
            gridLayout.setMargin(0)
        
        # channel selection
        comboBox = QtGui.QComboBox(self)
        comboBox.setEditable(True)
        comboBox.setInsertPolicy(QtGui.QComboBox.NoInsert)
        comboBox.addItems(["%d" % i for i in xrange(self.dh.nchannels)])
        comboBox.addItems(["Extra %d" % i for i in xrange(self.dh.nextrafet)])
        comboBox.editTextChanged.connect(self._select_channel_getter(coord), QtCore.Qt.UniqueConnection)
        # comboBox.currentIndexChanged.connect(self._select_channel_getter(coord), QtCore.Qt.UniqueConnection)
        # comboBox.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.channel_box[coord] = comboBox
        gridLayout.addWidget(comboBox, 0, 0, 1, self.dh.fetdim)
        
        # create 3 buttons for selecting the feature
        widths = [30] * self.dh.fetdim
        labels = ['PC%d' % i for i in xrange(1, self.dh.fetdim + 1)]
        
        # ensure exclusivity of the group of buttons
        pushButtonGroup = QtGui.QButtonGroup(self)
        for i in xrange(len(labels)):
            # selecting feature i
            pushButton = QtGui.QPushButton(labels[i], self)
            pushButton.setCheckable(True)
            if coord == i:
                pushButton.setChecked(True)
            pushButton.setMaximumSize(QtCore.QSize(widths[i], 20))
            pushButton.clicked.connect(self._select_feature_getter(coord, i), QtCore.Qt.UniqueConnection)
            pushButtonGroup.addButton(pushButton, i)
            self.feature_buttons[coord][i] = pushButton
            gridLayout.addWidget(pushButton, 1, i)
        
        return gridLayout
        
    def create_nspikes_viewer(self):
        self.nspikes_viewer = QtGui.QLabel("", self)
        return self.nspikes_viewer
        
    def get_nspikes_text(self, nspikes, nspikes_highlighted):
        return "Spikes: %d. Highlighted: %d." % (nspikes, nspikes_highlighted)
        
    def update_nspikes_viewer(self, nspikes, nspikes_highlighted):
        text = self.get_nspikes_text(nspikes, nspikes_highlighted)
        self.nspikes_viewer.setText(text)
        
    def update_feature_widget(self):
        for coord in [0, 1]:
            comboBox = self.channel_box[coord]
            # update the channels/features list only if necessary
            if comboBox.count() != self.dh.nchannels + self.dh.nextrafet:
                comboBox.blockSignals(True)
                comboBox.clear()
                comboBox.addItems(["%d" % i for i in xrange(self.dh.nchannels)])
                comboBox.addItems(["Extra %d" % i for i in xrange(self.dh.nextrafet)])
                comboBox.blockSignals(False)
        
    def create_controller(self):
        box = super(FeatureWidget, self).create_controller()
        
        # coord => channel combo box
        self.channel_box = [None, None]
        # coord => (butA, butB, butC)
        self.feature_buttons = [[None] * self.dh.fetdim, [None] * self.dh.fetdim]
        
        # add navigation toolbar
        self.toolbar = self.create_toolbar()
        box.addWidget(self.toolbar)

        # add number of spikes
        self.nspikes_viewer = self.create_nspikes_viewer()
        box.addWidget(self.nspikes_viewer)
        
        # add feature widget
        self.feature_widget1 = self.create_feature_widget(0)
        box.addLayout(self.feature_widget1)
        
        # add feature widget
        self.feature_widget2 = self.create_feature_widget(1)
        box.addLayout(self.feature_widget2)
        
        self.setTabOrder(self.channel_box[0], self.channel_box[1])
        
        return box
    
    def set_navigation(self):
        self.view.set_interaction_mode(FeatureNavigationBindings)
    
    def set_selection(self):
        self.view.set_interaction_mode(FeatureSelectionBindings)
    
    
        