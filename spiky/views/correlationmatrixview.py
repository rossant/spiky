from galry import *
from common import *
import numpy.random as rdn
from matplotlib.colors import hsv_to_rgb
from widgets import VisualizationWidget






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
    
    return hsv_to_rgb(col0 + (col1 - col0) * x)
    
    
class CorrelationMatrixDataManager(Manager):
    def set_data(self, matrix=None):
        if matrix.size == 0:
            matrix = np.zeros((2, 2))
        elif matrix.shape[0] == 1:
            matrix = np.zeros((2, 2))
        self.texture = colormap(matrix)
        
    
    
class CorrelationMatrixPaintManager(PaintManager):
    def initialize(self):
        self.add_visual(TextureVisual,
            texture=self.data_manager.texture, name='correlation_matrix')

    def update(self):
        self.set_data(
            texture=self.data_manager.texture, visual='correlation_matrix')
        
        
class CorrelationMatrixBindings(SpikyBindings):
    pass
    
        
class CorrelationMatrixInteractionManager(PlotInteractionManager):
    pass
        
        
class CorrelationMatrixView(GalryWidget):
    def initialize(self):
        self.set_bindings(CorrelationMatrixBindings)
        self.constrain_ratio = True
        self.constrain_navigation = True
        self.set_companion_classes(
            paint_manager=CorrelationMatrixPaintManager,
            interaction_manager=CorrelationMatrixInteractionManager,
            data_manager=CorrelationMatrixDataManager,)
    
    def set_data(self, *args, **kwargs):
        self.data_manager.set_data(*args, **kwargs)
    
        if self.initialized:
            log_debug("Updating data for correlograms")
            self.paint_manager.update()
            self.updateGL()
        else:
            log_debug("Initializing data for correlograms")
    
    
class CorrelationMatrixWidget(VisualizationWidget):
    def create_view(self, dh):
        self.dh = dh
        view = CorrelationMatrixView(getfocus=False)
        view.set_data(
                      matrix=dh.correlation_matrix,
                      # clusters_unique=self.dh.clusters_unique,
                      # cluster_colors=dh.cluster_colors
                      )
        return view
        
    def update_view(self, dh=None):
        if dh is not None:
            self.dh = dh
        self.view.set_data(
                      matrix=dh.correlation_matrix,
                      # clusters_unique=self.dh.clusters_unique,
                      # cluster_colors=dh.cluster_colors
                      )

