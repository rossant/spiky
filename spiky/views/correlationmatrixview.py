from galry import *
from common import *
import numpy.random as rdn
from matplotlib.colors import hsv_to_rgb
from widgets import VisualizationWidget
import spiky.colors as scolors
import spiky.signals as ssignals


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
    
    
class CorrelationMatrixDataManager(Manager):
    def set_data(self, matrix=None, 
        # clusters_unique=None, cluster_colors=None
        clusters_info=None,
        ):
        if matrix.size == 0:
            matrix = -np.ones((2, 2))
        elif matrix.shape[0] == 1:
            matrix = -np.ones((2, 2))
        n = matrix.shape[0]
        # remove diagonal
        # matrix[range(n), range(n)] = -1
        
        self.texture = colormap(matrix)[::-1, :, :]
        self.matrix = matrix
        
        clusters_info = clusters_info['clusters_info']
        clusters_unique = sorted(clusters_info.keys())
        cluster_colors = [clusters_info[c]['color'] for c in clusters_unique]
        
        self.clusters_unique = clusters_unique
        self.cluster_colors = cluster_colors
        self.nclusters = len(clusters_unique)
    
    
class CorrelationMatrixPaintManager(PaintManager):
    def initialize(self):
        self.add_visual(TextureVisual,
            texture=self.data_manager.texture, name='correlation_matrix')

        # self.add_visual(RectanglesVisual, coordinates=(0.,0.,0.,0.),
            # color=(0.,0.,0.,1.), name='clusterinfo_bg', visible=False,
            # depth=-.99,
            # is_static=True
            # )
        
        self.add_visual(TextVisual, text='0', name='clusterinfo', fontsize=16,
            # background=(0., 0., 0., 1.),
            # color
            posoffset=(.12, -.28),
            letter_spacing=250.,
            depth=-1,
            visible=False)
        
    def update(self):
        self.set_data(
            texture=self.data_manager.texture, visual='correlation_matrix')
        
        
class CorrelationMatrixInteractionManager(PlotInteractionManager):
    def initialize(self):
        self.register('ShowClosestCluster', self.show_closest_cluster)
        self.register('SelectPair', self.select_pair)
        self.register(None, self.hide_closest_cluster)
            
    def get_closest_cluster(self, parameter):
        nclu = self.data_manager.nclusters
        
        nav = self.get_processor('navigation')
        
        # window coordinates
        x, y = parameter
        # data coordinates
        xd, yd = nav.get_data_coordinates(x, y)
        
        cy = int((xd + 1) / 2. * nclu)
        cx = int((yd + 1) / 2. * nclu)
        
        cx_rel = np.clip(cx, 0, nclu - 1)
        cy_rel = np.clip(cy, 0, nclu - 1)
        
        return cx_rel, cy_rel
            
    def hide_closest_cluster(self, parameter):
        self.paint_manager.set_data(visible=False, visual='clusterinfo')
        # self.paint_manager.set_data(visible=False, visual='clusterinfo_bg')
        self.cursor = None
        
    def select_pair(self, parameter):
        
        self.cursor = 'ArrowCursor'
        
        cx_rel, cy_rel = self.get_closest_cluster(parameter)
        
        cx = self.data_manager.clusters_unique[cx_rel]
        cy = self.data_manager.clusters_unique[cy_rel]
        
        pair = np.unique(np.array([cx, cy]))
        
        ssignals.emit(self, 'ClusterSelectionToChange', pair)
        
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
        
        cx_rel, cy_rel = self.get_closest_cluster(parameter)
        
        color1 = self.data_manager.cluster_colors[cy_rel]
        r, g, b = scolors.COLORMAP[color1,:]
        color1 = (r, g, b, .75)
        
        cx = self.data_manager.clusters_unique[cx_rel]
        cy = self.data_manager.clusters_unique[cy_rel]
        
        if ((cx_rel >= self.data_manager.matrix.shape[0]) or
            (cy_rel >= self.data_manager.matrix.shape[1])):
            return
            
        val = self.data_manager.matrix[cx_rel, cy_rel]
        # val2 = self.data_manager.texture[cx_rel, cy_rel, :]
        # print type(val), val
        # if type(val) is float:
            # val = "%.3f" % float
        # else:
            # val = str(val)
        text = "%d / %d : %.3f" % (cx, cy, val)
        
        # print cx, cy, val
        # print val, val2
        
        # text = "%d / %d" % (cx, cy)
        
        # update clusterinfo visual
        # rect = (x-.24, y-.04, x+.44, y-.21)
        # self.paint_manager.set_data(coordinates=rect, 
            # visible=True,
            # visual='clusterinfo_bg')
            
        self.paint_manager.set_data(coordinates=(xd, yd), #color=color1,
            text=text,
            visible=True,
            visual='clusterinfo')
        
        
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
    
    
class CorrelationMatrixView(GalryWidget):
    def initialize(self):
        self.set_bindings(CorrelationMatrixBindings)
        # self.constrain_ratio = True
        # self.constrain_navigation = True
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
                      clusters_info=self.dh.clusters_info,
                      # clusters_unique=self.dh.clusters_unique,
                      # cluster_colors=dh.cluster_colors
                      )
        return view
        
    def update_view(self, dh=None):
        if dh is not None:
            self.dh = dh
        self.view.set_data(
                      matrix=dh.correlation_matrix,
                      clusters_info=self.dh.clusters_info,
                      # clusters_unique=self.dh.clusters_unique,
                      # cluster_colors=dh.cluster_colors
                      )

