"""Correlation matrix View: show correlation matrix."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import numpy.random as rdn
from galry import (Manager, PaintManager, PlotInteractionManager, Visual,
    GalryWidget, QtGui, QtCore, QtOpenGL, enforce_dtype, RectanglesVisual,
    TextVisual, TextureVisual)
from matplotlib.colors import hsv_to_rgb
    
from spiky.io.selection import get_indices
from spiky.io.tools import get_array
from spiky.utils.colors import COLORMAP
import spiky.utils.logger as log
from spiky.views.common import HighlightManager, SpikyBindings
from spiky.views.widgets import VisualizationWidget


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def colormap(x, col0=None, col1=None):
    """Colorize a 2D grayscale array.
    
    Arguments: 
      * x:an NxM array with values in [0,1].
      * col0=None: a tuple (H, S, V) corresponding to color 0. By default, a
        rainbow color gradient is used.
      * col1=None: a tuple (H, S, V) corresponding to color 1.
    
    Returns:
      * y: an NxMx3 array with a rainbow color palette.
    
    """
    # record values to be removed
    removed = x == -1
    
    x[np.isnan(x)] = 0.
    # TODO: proper scaling
    x -= x.min()
    x *= (1. / x.max())
    x = np.clip(x, 0., 1.)
    
    shape = x.shape
    
    if col0 is None:
        col0 = (.67, .91, .65)
    if col1 is None:
        col1 = (0., 1., 1.)
    
    col0 = np.array(col0).reshape((1, 1, -1))
    col1 = np.array(col1).reshape((1, 1, -1))
    
    col0 = np.tile(col0, x.shape + (1,))
    col1 = np.tile(col1, x.shape + (1,))
    
    x = np.tile(x.reshape(shape + (1,)), (1, 1, 3))
    
    y = hsv_to_rgb(col0 + (col1 - col0) * x)
    
    # value of -1 = black
    y[removed,:] = 0
    
    return y
    

# -----------------------------------------------------------------------------
# Data manager
# -----------------------------------------------------------------------------
class CorrelationMatrixDataManager(Manager):
    def set_data(self, correlation_matrix=None,
        cluster_colors_full=None
        # clusters_info=None,
        ):
        
        if correlation_matrix is None:
            correlation_matrix = np.zeros(0)
            cluster_colors_full = np.zeros(0)
        
        if correlation_matrix.size == 0:
            correlation_matrix = -np.ones((2, 2))
        elif correlation_matrix.shape[0] == 1:
            correlation_matrix = -np.ones((2, 2))
        n = correlation_matrix.shape[0]
        # remove diagonal
        # correlation_matrix[range(n), range(n)] = -1
        
        self.texture = colormap(correlation_matrix)[::-1, :, :]
        self.correlation_matrix = correlation_matrix
        
        
        # Hide some clusters.
        # if n >= 3:
            # tex0 = self.texture.copy()
            # for clu in clusters_hidden:
                # self.texture[clu, :, :] = tex0[clu, :, :] * .25
                # self.texture[:, clu, :] = tex0[:, clu, :] * .25
        
        
        self.clusters_unique = get_indices(cluster_colors_full)
        self.cluster_colors = cluster_colors_full
        self.nclusters = len(self.clusters_unique)
    
    
# -----------------------------------------------------------------------------
# Visuals
# -----------------------------------------------------------------------------
class CorrelationMatrixPaintManager(PaintManager):
    def initialize(self):
        self.add_visual(TextureVisual,
            texture=self.data_manager.texture, 
            name='correlation_matrix')

        self.add_visual(TextVisual, text='0', name='clusterinfo', fontsize=16,
            background_transparent=False,
            posoffset=(.12, -.28),
            letter_spacing=350.,
            depth=-1,
            visible=False)
        
    def update(self):
        self.set_data(
            texture=self.data_manager.texture, visual='correlation_matrix')
        

# -----------------------------------------------------------------------------
# Interaction
# -----------------------------------------------------------------------------
class CorrelationMatrixInfoManager(Manager):
    def initialize(self):
        pass
        
    def get_closest_cluster(self, xd, yd):
        nclu = self.data_manager.nclusters
        
        cy = int((xd + 1) / 2. * nclu)
        cx = int((yd + 1) / 2. * nclu)
        
        cx_rel = np.clip(cx, 0, nclu - 1)
        cy_rel = np.clip(cy, 0, nclu - 1)
        
        return cx_rel, cy_rel
        
    def show_closest_cluster(self, xd, yd):
        
        cx_rel, cy_rel = self.get_closest_cluster(xd, yd)
        
        color1 = self.data_manager.cluster_colors[cy_rel]
        r, g, b = COLORMAP[color1,:]
        color1 = (r, g, b, .75)
        
        cx = self.data_manager.clusters_unique[cx_rel]
        cy = self.data_manager.clusters_unique[cy_rel]
        
        if ((cx_rel >= self.data_manager.correlation_matrix.shape[0]) or
            (cy_rel >= self.data_manager.correlation_matrix.shape[1])):
            return
            
        val = self.data_manager.correlation_matrix[cx_rel, cy_rel]
        
        text = "%d / %d : %.3f" % (cx, cy, val)
        
        self.paint_manager.set_data(coordinates=(xd, yd), 
            text=text,
            visible=True,
            visual='clusterinfo')
        
    
class CorrelationMatrixInteractionManager(PlotInteractionManager):
    def initialize(self):
        self.register('ShowClosestCluster', self.show_closest_cluster)
        self.register('SelectPair', self.select_pair)
        self.register(None, self.hide_closest_cluster)
            
    def hide_closest_cluster(self, parameter):
        self.paint_manager.set_data(visible=False, visual='clusterinfo')
        self.cursor = None
        
    def select_pair(self, parameter):
        
        self.cursor = 'ArrowCursor'
        
        nav = self.get_processor('navigation')
        
        # window coordinates
        x, y = parameter
        # data coordinates
        xd, yd = nav.get_data_coordinates(x, y)
        
        cx_rel, cy_rel = self.info_manager.get_closest_cluster(xd, yd)
        
        cx = self.data_manager.clusters_unique[cx_rel]
        cy = self.data_manager.clusters_unique[cy_rel]
        
        # Emit signal.
        log.debug("Selected clusters {0:d} and {1:d}.".format(cx, cy))
        self.parent.pairSelected.emit(cx, cy)
        
    def show_closest_cluster(self, parameter):
        nclu = self.data_manager.nclusters
        
        if nclu == 0:
            return
            
        self.cursor = 'ArrowCursor'
        nav = self.get_processor('navigation')
        
        # window coordinates
        x, y = parameter
        # data coordinates
        xd, yd = nav.get_data_coordinates(x, y)
        
        self.info_manager.show_closest_cluster(xd, yd)
        
        
class CorrelationMatrixBindings(SpikyBindings):
    # def set_zoombox_keyboard(self):
        # """Set zoombox bindings with the keyboard."""
        # self.set('LeftClickMove', 'ZoomBox',
                    # key_modifier='Shift',
                    # param_getter=lambda p: (p["mouse_press_position"][0],
                                            # p["mouse_press_position"][1],
                                            # p["mouse_position"][0],
                                            # p["mouse_position"][1]))

    def set_clusterinfo(self):
        self.set('Move', 'ShowClosestCluster', key_modifier='Shift',
            param_getter=lambda p:
            (p['mouse_position'][0], p['mouse_position'][1]))
    
    def set_selectcluster(self):
        self.set('LeftClick', 'SelectPair', key_modifier='Shift',
            param_getter=lambda p:
            (p['mouse_position'][0], p['mouse_position'][1]))
    
    def initialize(self):
        super(CorrelationMatrixBindings, self).initialize()
        self.set_clusterinfo()
        self.set_selectcluster()
    

# -----------------------------------------------------------------------------
# Top-level module
# -----------------------------------------------------------------------------
class CorrelationMatrixView(GalryWidget):
    
    # Raise the list of highlighted spike absolute indices.
    pairSelected = QtCore.pyqtSignal(int, int)
    
    def initialize(self):
        self.set_bindings(CorrelationMatrixBindings)
        self.set_companion_classes(
            paint_manager=CorrelationMatrixPaintManager,
            info_manager=CorrelationMatrixInfoManager,
            interaction_manager=CorrelationMatrixInteractionManager,
            data_manager=CorrelationMatrixDataManager,)
    
    def set_data(self, *args, **kwargs):
        # if kwargs.get('correlation_matrix', None) is None:
            # return
        self.data_manager.set_data(*args, **kwargs)
        
        # update?
        if self.initialized:
            self.paint_manager.update()
            self.updateGL()

    
    
    def sizeHint(self):
        return QtCore.QSize(300, 400)
        