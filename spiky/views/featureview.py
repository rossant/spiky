"""Feature View: show spikes as 2D points in feature space."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import Counter
import operator
import time

import numpy as np
import numpy.random as rdn
from matplotlib.path import Path

from galry import (Manager, PlotPaintManager, PlotInteractionManager, Visual,
    GalryWidget, QtGui, QtCore, show_window, enforce_dtype, RectanglesVisual,
    TextVisual, PlotVisual, AxesVisual)
from spiky.io.selection import get_indices
from spiky.io.tools import get_array
from spiky.views.common import HighlightManager, SpikyBindings
from spiky.views.widgets import VisualizationWidget
from spiky.utils.colors import COLORMAP, HIGHLIGHT_COLORMAP
import spiky.utils.logger as log
import spiky


# -----------------------------------------------------------------------------
# Shaders
# -----------------------------------------------------------------------------
VERTEX_SHADER = """
    // move the vertex to its position
    vec3 position = vec3(0, 0, 0);
    position.xy = position0;
    
    vhighlight = highlight;
    cmap_vindex = cmap_index;
    vmask = mask;
    vselection = selection;
        
        
    // compute the depth: put masked spikes on the background, unmasked ones
    // on the foreground on a different layer for each cluster
    float depth = 0.;
    //if (mask == 1.)
    depth = -(cluster_depth + 1) / (nclusters + 10);
    position.z = depth;
        
    if ((highlight > 0) || (selection > 0))
        gl_PointSize = 5.;
    else
        gl_PointSize = 3.;
        
    // DEBUG
    //gl_PointSize = 20;
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


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Data manager
# -----------------------------------------------------------------------------
class FeatureDataManager(Manager):
    # Initialization methods
    # ----------------------
    def set_data(self,
                 features=None,
                 masks=None,
                 clusters=None,
                 clusters_selected=None,
                 cluster_colors=None,
                 fetdim=None,
                 nchannels=None,
                 nextrafet=None,
                 ):
        
        if features is None:
            features = np.zeros((0, 2))
            masks = np.zeros((0, 1))
            clusters = np.zeros(0, dtype=np.int32)
            clusters_selected = []
            cluster_colors = np.zeros(0, dtype=np.int32)
            fetdim = 2
            nchannels = 1
            nextrafet = 0
        
        assert fetdim is not None
        
        self.nspikes, self.ndim = features.shape
        self.fetdim = fetdim
        self.nchannels = nchannels
        self.nextrafet = nextrafet
        self.npoints = features.shape[0]
        self.features = features
        self.masks = masks
        self.clusters = clusters
        self.feature_indices = get_indices(self.features)
        self.feature_indices_array = get_array(self.feature_indices)
        
        self.features_array = get_array(self.features)
        self.masks_array = get_array(self.masks)
        self.clusters_array = get_array(self.clusters)
        self.cluster_colors = get_array(cluster_colors)
        
        # Relative indexing.
        if len(clusters_selected) > 0:
            self.clusters_rel = np.digitize(self.clusters_array, sorted(clusters_selected)) - 1
            self.clusters_rel_ordered = np.argsort(clusters_selected)[self.clusters_rel]
        else:
            self.clusters_rel = np.zeros(0, dtype=np.int32)
            self.clusters_rel_ordered = np.zeros(0, dtype=np.int32)
        
        self.clusters_unique = sorted(clusters_selected)
        self.nclusters = len(Counter(clusters))
        self.masks_full = self.masks_array.T.ravel()
        self.clusters_full_depth = self.clusters_rel_ordered
        self.clusters_full = self.clusters_rel
        
        # prepare GPU data
        self.data = np.empty((self.nspikes, 2), dtype=np.float32)
        
        # set initial projection
        self.projection_manager.set_data()
        self.projection_manager.reset_projection()
        
        # update the highlight manager
        self.highlight_manager.initialize()
        self.selection_manager.initialize()


# -----------------------------------------------------------------------------
# Visuals
# -----------------------------------------------------------------------------
class FeatureVisual(Visual):
    def initialize(self, npoints=None, 
                    nclusters=None,
                    cluster_depth=None,
                    position0=None,
                    mask=None,
                    cluster=None,
                    highlight=None,
                    selection=None,
                    cluster_colors=None,
                    ):
        
        self.primitive_type = 'POINTS'
        self.size = npoints
        
        self.add_attribute("position0", vartype="float", ndim=2, data=position0)
        
        self.add_attribute("mask", vartype="float", ndim=1, data=mask)
        self.add_varying("vmask", vartype="float", ndim=1)
        
        self.add_attribute("highlight", vartype="int", ndim=1, data=highlight)
        self.add_varying("vhighlight", vartype="int", ndim=1)
        
        self.add_uniform("toggle_mask", vartype="int", ndim=1, data=0)
        
        self.add_attribute("selection", vartype="int", ndim=1, data=selection)
        self.add_varying("vselection", vartype="int", ndim=1)
        
        self.add_uniform("nclusters", vartype="int", ndim=1, data=nclusters)
        
        self.add_attribute("cluster_depth", vartype="int", ndim=1, data=cluster_depth)
        
        # color map for cluster colors, each spike has an index of the color
        # in the color map
        ncolors = COLORMAP.shape[0]
        ncomponents = COLORMAP.shape[1]
        
        # associate the cluster color to each spike
        # give the correct shape to cmap
        colormap = COLORMAP.reshape((1, ncolors, ncomponents))
        hcolormap = HIGHLIGHT_COLORMAP.reshape((1, ncolors, ncomponents))
        
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
        
        # necessary so that the navigation shader code is updated
        self.is_position_3D = True
        
        self.add_vertex_main(VERTEX_SHADER)
        self.add_fragment_main(FRAGMENT_SHADER)


class FeaturePaintManager(PlotPaintManager):
    def update_points(self):
        self.set_data(position0=self.data_manager.data,
            mask=self.data_manager.masks_full, visual='features')
        
    def initialize(self):
        self.toggle_mask_value = False
        
        self.add_visual(FeatureVisual, name='features',
            npoints=self.data_manager.npoints,
            position0=self.data_manager.data,
            mask=self.data_manager.masks_full,
            cluster=self.data_manager.clusters_rel,
            highlight=self.highlight_manager.highlight_mask,
            selection=self.selection_manager.selection_mask,
            cluster_colors=self.data_manager.cluster_colors,
            nclusters=self.data_manager.nclusters,
            cluster_depth=self.data_manager.clusters_full_depth,
            )
        
        self.add_visual(AxesVisual, name='grid')
        
        self.add_visual(TextVisual, text='0', name='clusterinfo', fontsize=16,
            background_transparent=False,
            posoffset=(.05, -.05),
            letter_spacing=350.,
            depth=-1,
            visible=False)
        
    def update(self):

        cluster = self.data_manager.clusters_rel
        cluster_colors = self.data_manager.cluster_colors
        cmap_index = cluster_colors[cluster]
    
        self.set_data(visual='features', 
            size=self.data_manager.npoints,
            position0=self.data_manager.data,
            mask=self.data_manager.masks_full,
            highlight=self.highlight_manager.highlight_mask,
            selection=self.selection_manager.selection_mask,
            nclusters=self.data_manager.nclusters,
            cluster_depth=self.data_manager.clusters_full_depth,
            cmap_index=cmap_index
            )

    def toggle_mask(self):
        self.toggle_mask_value = 1 - self.toggle_mask_value
        self.set_data(visual='features', toggle_mask=self.toggle_mask_value)


# -----------------------------------------------------------------------------
# Highlight/Selection manager
# -----------------------------------------------------------------------------
class FeatureHighlightManager(HighlightManager):
    def initialize(self):
        super(FeatureHighlightManager, self).initialize()
        self.feature_indices = self.data_manager.feature_indices
        self.feature_indices_array = self.data_manager.feature_indices_array
        self.highlight_mask = np.zeros(self.data_manager.nspikes,
            dtype=np.int32)
        self.highlighted_spikes = []
        
    def find_enclosed_spikes(self, enclosing_box):
        x0, y0, x1, y1 = enclosing_box
        
        # press_position
        xp, yp = x0, y0

        # reorder
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)

        features = self.data_manager.data
        masks = self.data_manager.masks_full

        indices = (
                  # (masks > 0) & \
                  (features[:,0] >= xmin) & (features[:,0] <= xmax) & \
                  (features[:,1] >= ymin) & (features[:,1] <= ymax))
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
            if len(spikes) > 0:
                self.highlight_mask[spikes] = 1
        
        if do_update:
            self.paint_manager.set_data(
                highlight=self.highlight_mask, visual='features')
        
        self.highlighted_spikes = spikes
        
    def highlighted(self, box):
        # Get selected spikes (relative indices).
        spikes = self.find_enclosed_spikes(box)
        # Set highlighted spikes.
        self.set_highlighted_spikes(spikes)
        # Emit the HighlightSpikes signal.
        self.emit(spikes)
        
    def highlight_spikes(self, spikes):
        """spikes in absolute indices."""
        spikes = np.intersect1d(self.data_manager.feature_indices_array, 
            spikes)
        if len(spikes) > 0:
            spikes_rel = np.digitize(spikes, 
                self.data_manager.feature_indices_array) - 1
        else:
            spikes_rel = []
        self.set_highlighted_spikes(spikes_rel)
        
    def cancel_highlight(self):
        super(FeatureHighlightManager, self).cancel_highlight()
        self.set_highlighted_spikes(np.array([]))
        self.emit([])

    def emit(self, spikes):
        spikes = np.array(spikes, dtype=np.int32)
        spikes_abs = self.feature_indices[spikes]
        # emit signal
        # log.debug("Highlight {0:d} spikes.".format(len(spikes_abs)))
        self.parent.spikesHighlighted.emit(spikes_abs)
        

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
        self.feature_indices = self.data_manager.feature_indices
        self.selection_mask = np.zeros(self.data_manager.nspikes, dtype=np.int32)
        self.selected_spikes = []
        
    def set_selected_spikes(self, spikes):
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
            self.paint_manager.set_data(
                selection=self.selection_mask, visual='features')
        
        self.selected_spikes = spikes
    
    def emit(self, spikes):
        spikes = np.array(spikes, dtype=np.int32)
        spikes_abs = self.feature_indices[spikes]
        # emit signal
        # log.debug("Select {0:d} spikes.".format(len(spikes_abs)))
        self.parent.spikesSelected.emit(spikes_abs)
        
    def find_enclosed_spikes(self, polygon=None):
        """Find the indices of the spikes inside the polygon (in 
        transformed coordinates)."""
        if polygon is None:
            polygon = self.polygon()
        features = self.data_manager.data
        masks = self.data_manager.masks_full
        # indices = (masks > 0) & polygon_contains_points(polygon, features)
        indices = polygon_contains_points(polygon, features)
        spkindices = np.nonzero(indices)[0]
        spkindices = np.unique(spkindices)
        return spkindices
   
    def select_spikes(self, polygon=None):
        """Select spikes enclosed in the selection polygon."""
        spikes = self.find_enclosed_spikes(polygon)
        self.set_selected_spikes(spikes)
        self.emit(spikes)
   
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
        self.projection = list(self.projection_manager.projection)
        self.is_selection_pending = False
        
    def cancel_selection(self):
        """Remove the selection polygon."""
        # hide the selection polygon
        if self.paint_manager.get_visual('selection_polygon').get('visible', None):
            self.paint_manager.set_data(visible=False,
                visual='selection_polygon')
        self.set_selected_spikes(np.array([]))
        self.is_selection_pending = False


# -----------------------------------------------------------------------------
# Projection
# -----------------------------------------------------------------------------
class FeatureProjectionManager(Manager):
    def set_data(self):
        if not hasattr(self, 'projection'):
            self.projection = [None, None]
        self.nchannels = self.data_manager.nchannels
        self.fetdim = self.data_manager.fetdim
        self.nextrafet = self.data_manager.nextrafet
        self.nchannels = self.data_manager.nchannels
        
    def set_projection(self, coord, channel, feature, do_update=True):
        """Set the projection axes."""
        if channel < self.nchannels:
            i = channel * self.fetdim + feature
            self.data_manager.masks_full = self.data_manager.masks_array[:,channel]
        # handle extra feature, with channel being directly the feature index
        else:
            i = min(self.nchannels * self.fetdim + self.nextrafet - 1,
                    channel - self.nchannels + self.nchannels * self.fetdim)
        self.data_manager.data[:, coord] = self.data_manager.features_array[:, i].ravel()
        
        if do_update:
            self.projection[coord] = (channel, feature)
            # show the selection polygon only if the projection axes correspond
            self.selection_manager.set_selection_polygon_visibility(
              (self.projection[0] == self.selection_manager.projection[0]) & \
              (self.projection[1] == self.selection_manager.projection[1]))
        
    def reset_projection(self):
        if self.projection[0] is None or self.projection[1] is None:
            self.set_projection(0, 0, 0, False)
            self.set_projection(1, 0, 1)
        else:
            self.set_projection(0, self.projection[0][0], self.projection[0][1], False)
            self.set_projection(1, self.projection[1][0], self.projection[1][1])

    def select_neighbor_channel(self, coord, channel_dir):
        # current channel and feature in the given coordinate
        proj = self.projection[coord]
        if proj is None:
            proj = (0, coord)
        channel, feature = proj
        # next or previous channel
        channel = np.mod(channel + channel_dir, self.data_manager.nchannels + 
            self.data_manager.nextrafet)
        self.set_projection(coord, channel, feature, do_update=True)
        
    def select_neighbor_feature(self, parameter):
        coord, feature_dir = parameter
        # current channel and feature in the given coordinate
        proj = self.projection_manager.projection[coord]
        if proj is None:
            proj = (0, coord)
        channel, feature = proj
        # next or previous feature
        feature = np.mod(feature + feature_dir, self.data_manager.fetdim)
        self.set_projection(coord, channel, feature, do_update=True)
            
    def get_projection(self, coord):
        return self.projection[coord]
        
        
# -----------------------------------------------------------------------------
# Interaction
# -----------------------------------------------------------------------------
class FeatureInfoManager(Manager):
    def show_closest_cluster(self, xd, yd):
        # find closest spike
        dist = (self.data_manager.data[:, 0] - xd) ** 2 + \
                (self.data_manager.data[:, 1] - yd) ** 2
        ispk = dist.argmin()
        cluster = self.data_manager.clusters_rel[ispk]
        
        # color = self.data_manager.cluster_colors[cluster]
        # r, g, b = COLORMAP[color,:]
        # color = (r, g, b, .75)
        
        text = "cluster {0:d}".format(self.data_manager.clusters_unique[cluster])
        
        self.paint_manager.set_data(coordinates=(xd, yd), #color=color,
            text=text,
            visible=True,
            visual='clusterinfo')
        

class FeatureInteractionManager(PlotInteractionManager):
    def initialize(self):
        self.constrain_navigation = False
    
        self.register(None, self.cancel_highlight)
        self.register('HighlightSpike', self.highlight_spike)
        self.register('SelectionPointPending', self.selection_point_pending)
        
        self.register('AddSelectionPoint', self.selection_add_point)
        self.register('EndSelectionPoint', self.selection_end_point)
        self.register('CancelSelectionPoint', self.selection_cancel)
        
        self.register('SelectProjection', self.select_projection)

        self.register('ToggleMask', self.toggle_mask)

        self.register('SelectNeighborChannel', self.select_neighbor_channel)
        self.register('SelectNeighborFeature', self.select_neighbor_feature)
        
        self.register('ShowClosestCluster', self.show_closest_cluster)
    
    
    # Highlighting
    # ------------
    def cancel_highlight(self, parameter):
        self.highlight_manager.cancel_highlight()
        self.paint_manager.set_data(visible=False, visual='clusterinfo')
        
    def highlight_spike(self, parameter):
        self.highlight_manager.highlight(parameter)
        self.cursor = 'CrossCursor'
        
        
    # Selection
    # ---------
    def selection_point_pending(self, parameter):
        self.selection_manager.point_pending(parameter)
        
    def selection_add_point(self, parameter):
        self.selection_manager.add_point(parameter)
        
    def selection_end_point(self, parameter):
        self.selection_manager.end_point(parameter)
        
    def selection_cancel(self, parameter):
        self.selection_manager.cancel_selection()

        
    # Projection
    # ----------
    def select_neighbor_channel(self, parameter):
        coord, channel_dir = parameter
        
        self.projection_manager.select_neighbor_channel(coord, channel_dir)
        channel, feature = self.projection_manager.get_projection(coord)
        
        log.debug(("Projection changed to channel {0:d} and "
                   "feature {1:d} on axis {2:s}.").format(
                        channel, feature, 'xy'[coord]))
        self.parent.projectionChanged.emit(coord, channel, feature)
        
        self.paint_manager.update_points()
        self.paint_manager.updateGL()
        
    def select_neighbor_feature(self, parameter):
        coord, feature_dir = parameter
        
        self.projection_manager.select_neighbor_channel(coord, channel_dir)
        channel, feature = self.projection_manager.get_projection(coord)
        
        log.debug(("Projection changed to channel {0:d} and "
                   "feature {1:d} on axis {2:s}.").format(
                        channel, feature, 'xy'[coord]))
        self.parent.projectionChanged.emit(coord, channel, feature)
        
        self.paint_manager.update_points()
        self.paint_manager.updateGL()
            
    def select_projection(self, parameter):
        """Select a projection for the given coordinate."""
        self.projection_manager.set_projection(*parameter)  # coord, channel, feature
        self.paint_manager.update_points()
        self.paint_manager.updateGL()
    
    
    # Misc
    # ----
    def toggle_mask(self, parameter):
        self.paint_manager.toggle_mask()
        
    def show_closest_cluster(self, parameter):
        
        self.cursor = None
        
        nav = self.get_processor('navigation')
        # print "hey"
        # window coordinates
        x, y = parameter
        # data coordinates
        xd, yd = nav.get_data_coordinates(x, y)
        
        # print self.data_manager.data
        if self.data_manager.data.size == 0:
            return
            
        self.info_manager.show_closest_cluster(xd, yd)
    
    
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
        self.set('Wheel', 'SelectNeighborChannel',
                 key_modifier='Control',
                 param_getter=lambda p: (0, -int(np.sign(p['wheel']))))
                 
        # select previous/next channel for coordinate 1
        self.set('Wheel', 'SelectNeighborChannel',
                 key_modifier='Shift',
                 param_getter=lambda p: (1, -int(np.sign(p['wheel']))))
        
    # def set_neighbor_feature(self):
        # # select previous/next feature for coordinate 0
        # self.set('KeyPress', 'SelectNeighborFeature',
                 # key='Left', description='X-', key_modifier='Control',
                 # param_getter=lambda p: (0, -1))
        # self.set('KeyPress', 'SelectNeighborFeature',
                 # key='Right', description='X+', key_modifier='Control',
                 # param_getter=lambda p: (0, 1))
                 
        # # select previous/next feature for coordinate 1
        # self.set('KeyPress', 'SelectNeighborFeature',
                 # key='Left', description='Y-', key_modifier='Shift',
                 # param_getter=lambda p: (1, -1))
        # self.set('KeyPress', 'SelectNeighborFeature',
                 # key='Right', description='Y+', key_modifier='Shift',
                 # param_getter=lambda p: (1, 1))
        
    def set_clusterinfo(self):
        self.set('Move', 'ShowClosestCluster', key_modifier='Shift',
            param_getter=lambda p:
            (p['mouse_position'][0], p['mouse_position'][1]))
        
    def set_switch_mode(self):
        self.set('KeyPress', 'SwitchInteractionMode', key='N')
        
    def set_selection(self):
        # selection
        self.set('Move',
                 'SelectionPointPending',
                 key_modifier=QtCore.Qt.Key_Control,
                 param_getter=lambda p: (p["mouse_position"][0],
                                         p["mouse_position"][1],))
        self.set('LeftClick',
                 'AddSelectionPoint',
                 key_modifier=QtCore.Qt.Key_Control,
                 param_getter=lambda p: (p["mouse_press_position"][0],
                                         p["mouse_press_position"][1],))
        self.set('RightClick',
                 'EndSelectionPoint',
                 key_modifier=QtCore.Qt.Key_Control,
                 param_getter=lambda p: (p["mouse_press_position"][0],
                                         p["mouse_press_position"][1],))
        self.set('DoubleClick',
                 'CancelSelectionPoint',
                 key_modifier=QtCore.Qt.Key_Control,
                 param_getter=lambda p: (p["mouse_press_position"][0],
                                         p["mouse_press_position"][1],))
    
    def initialize(self):
        self.set_highlight()
        self.set_toggle_mask()
        self.set_neighbor_channel()
        # self.set_neighbor_feature()
        self.set_switch_mode()
        self.set_clusterinfo()
        self.set_selection()


# -----------------------------------------------------------------------------
# Top-level widget
# -----------------------------------------------------------------------------
class FeatureView(GalryWidget):
    # Raise the list of highlighted spike absolute indices.
    spikesHighlighted = QtCore.pyqtSignal(np.ndarray)
    spikesSelected = QtCore.pyqtSignal(np.ndarray)
    projectionChanged = QtCore.pyqtSignal(int, int, int)
    
    # Initialization
    # --------------
    def initialize(self):
        self.activate3D = True
        self.set_bindings(FeatureBindings)
        self.set_companion_classes(
                paint_manager=FeaturePaintManager,
                data_manager=FeatureDataManager,
                projection_manager=FeatureProjectionManager,
                info_manager=FeatureInfoManager,
                highlight_manager=FeatureHighlightManager,
                selection_manager=FeatureSelectionManager,
                interaction_manager=FeatureInteractionManager)
    
    def set_data(self, *args, **kwargs):
        # if kwargs.get('clusters_selected', None) is None:
            # return
        self.data_manager.set_data(*args, **kwargs)
        
        # update?
        if self.initialized:
            self.paint_manager.update()
            self.updateGL()

    
    # Public methods
    # --------------
    def highlight_spikes(self, spikes):
        self.highlight_manager.highlight_spikes(spikes)
        self.updateGL()
    
    def select_spikes(self, spikes):
        pass
    
    def toggle_mask(self):
        pass
        
    def set_projection(self, coord, channel, feature):
        pass
        
        
        
    def sizeHint(self):
        return QtCore.QSize(600, 600)
    
        