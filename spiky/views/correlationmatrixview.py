from galry import *
import numpy.random as rdn
from matplotlib.colors import hsv_to_rgb


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
    
    
class CorrelationMatrixPaintManager(PaintManager):
    def load_data(self, data):
        self.texture = colormap(data)
    
    def initialize(self):
        self.create_dataset(TextureTemplate, texture=self.texture)

        
class CorrelationMatrixView(GalryWidget):
    def initialize(self, **kwargs):
        self.constrain_ratio = True
        self.constrain_navigation = True
        self.set_companion_classes(paint_manager=CorrelationMatrixPaintManager)
    
    def set_data(self, data):
        self.paint_manager.load_data(data)

        