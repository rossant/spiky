import numpy.random as rdn
from galry import *
from common import HighlightManager, SpikyBindings
from widgets import VisualizationWidget
import spiky.colors as scolors
import spiky.tools as stools
import spiky.signals as ssignals

VERTEX_SHADER = """
    //vec3 color = vec3(1, 1, 1);

    float margin = 0.05;
    float a = 1.0 / (nclusters * (1 + 2 * margin));
    
    vec2 box_position = vec2(0, 0);
    box_position.x = -1 + a * (1 + 2 * margin) * (2 * cluster.x + 1);
    box_position.y = -1 + a * (1 + 2 * margin) * (2 * cluster.y + 1);
    
    vec2 transformed_position = position;
    transformed_position.y = 2 * transformed_position.y - 1;
    transformed_position = box_position + a * transformed_position;
"""


def get_histogram_points(hist):
    """Tesselates histograms.
    
    Arguments:
      * hist: a N x Nsamples array, where each line contains an histogram.
      
    Returns:
      * X, Y: two N x (5*Nsamples+1) arrays with the coordinates of the
        histograms, a
      
    """
    if hist.size == 0:
        return np.array([[]]), np.array([[]])
    n, nsamples = hist.shape
    dx = 2. / nsamples
    
    x0 = -1 + dx * np.arange(nsamples)
    
    x = np.zeros((n, 5 * nsamples + 1))
    # y = -np.ones((n, 5 * nsamples + 1))
    y = np.zeros((n, 5 * nsamples + 1))
    
    x[:,0:-1:5] = x0
    x[:,1::5] = x0
    x[:,2::5] = x0 + dx
    x[:,3::5] = x0
    x[:,4::5] = x0 + dx
    x[:,-1] = 1
    
    y[:,1::5] = hist
    y[:,2::5] = hist
    
    return x, y


class CorrelogramsDataManager(Manager):
    def set_data(self, histograms=None, cluster_colors=None, baselines=None,
        clusters_unique=None):
        
        self.histograms = histograms
        self.nhistograms, self.nbins = histograms.shape
        
        self.clusters_unique = clusters_unique
        
        # HACK: if histograms is empty, nhistograms == 1 here!
        if histograms.size == 0:
            self.nhistograms = 0
        
        # add sub-diagonal
        self.nclusters = int((-1 + np.sqrt(1 + 8 * self.nhistograms)) / 2.)
        if self.nclusters:
            cl = np.array([(i,j) for i in xrange(self.nclusters) for j in xrange(self.nclusters) if j >= i])
            nonid = cl[:,0] != cl[:,1]
            self.histograms = np.vstack((self.histograms, self.histograms[nonid,::-1]))
            self.nhistograms, self.nbins = self.histograms.shape
        
        # cluster i and j for each histogram in the view
        clusters = [(i,j) for i in xrange(self.nclusters) for j in xrange(self.nclusters) if j >= i]
        clusters += [(j,i) for i in xrange(self.nclusters) for j in xrange(self.nclusters) if j > i]
        self.clusters = np.array(clusters, dtype=np.int32)
        
        # normalization
        for j in xrange(self.nclusters):
            # histograms in a given row
            ind = self.clusters[:,1] == j
            # index of the (i,j) histogram
            i0 = np.nonzero((cl[:,0] == cl[:,1]) & (cl[:,0] == j))[0][0]
            # divide all histograms in the row by the max of this i0 histogram
            m = self.histograms[i0,:].max()
            if m > 0:
                self.histograms[ind,:] /= m
            # normalize all histograms in the row so that they all fit in the 
            # window
            m = self.histograms[ind,:].max()
            if m > 0:
                self.histograms[ind,:] /= m
        
        self.nprimitives = self.nhistograms
        # index 0 = heterogeneous clusters, index>0 ==> cluster index + 1
        self.cluster_colors = cluster_colors
        
        # get the vertex positions
        X, Y = get_histogram_points(self.histograms)
        n = X.size
        self.nsamples = X.shape[1]
        
        # fill the data array
        self.position = np.empty((n, 2), dtype=np.float32)
        self.position[:,0] = X.ravel()
        self.position[:,1] = Y.ravel()
        
        # baselines of the correlograms
        self.baselines = baselines
        
        # indices of histograms on the diagonal
        if self.nclusters:
            identity = self.clusters[:,0] == self.clusters[:,1]
        else:
            identity = []
            
        color_array_index = np.zeros(self.nhistograms, dtype=np.int32)
        
        color_array_index[identity] = np.array(cluster_colors + 1, dtype=np.int32)
        # very first color in color map = white (cross-correlograms)
        self.color = np.vstack((np.ones((1, 3)), scolors.COLORMAP))
        self.color_array_index = color_array_index
        
        self.clusters = np.repeat(self.clusters, self.nsamples, axis=0)
        self.color_array_index = np.repeat(self.color_array_index, self.nsamples, axis=0)
        
        
class CorrelogramsVisual(PlotVisual):
    def initialize(self, nclusters=None, nhistograms=None, #nsamples=None,
        position=None, color=None, color_array_index=None, clusters=None):
        
        self.position_attribute_name = "transformed_position"
        
        super(CorrelogramsVisual, self).initialize(
            position=position,
            nprimitives=nhistograms,
            color=color,
            color_array_index=color_array_index,
            )
            
        self.primitive_type = 'TRIANGLE_STRIP'
        # self.primitive_type = 'LINE_STRIP'
        
        # print position.shape
        # print nhistograms
        # print nclusters
        # print clusters.shape
        # print
        
        self.add_attribute("cluster", vartype="int", ndim=2, data=clusters)
        self.add_uniform("nclusters", vartype="int", ndim=1, data=nclusters)
        
        self.add_vertex_main(VERTEX_SHADER)

        
class CorrelogramsBaselineVisual(PlotVisual):
    def initialize(self, nclusters=None, baselines=None, clusters=None):
        
        self.position_attribute_name = "transformed_position"
        
        # texture = np.ones((10, 10, 3))
        
        n = len(baselines)
        position = np.zeros((2 * n, 2))
        position[:,0] = np.tile(np.array([-1., 1.]), n)
        position[:,1] = np.repeat(baselines, 2)
        # position[n:,1] = baselines
        
        clusters = np.repeat(clusters, 2, axis=0)
        
        self.primitive_type = 'LINES'
        
        super(CorrelogramsBaselineVisual, self).initialize(
            position=position,
            # texture=texture
            )
            
        self.add_attribute("cluster", vartype="int", ndim=2, data=clusters)
        self.add_uniform("nclusters", vartype="int", ndim=1, data=nclusters)
        
        self.add_vertex_main(VERTEX_SHADER)
        
        
class CorrelogramsPaintManager(PlotPaintManager):
    def initialize(self, **kwargs):
        self.add_visual(CorrelogramsVisual,
            nclusters=self.data_manager.nclusters,
            # nsamples=self.data_manager.nsamples,
            nhistograms=self.data_manager.nhistograms,
            position=self.data_manager.position,
            color=self.data_manager.color,
            color_array_index=self.data_manager.color_array_index,
            clusters=self.data_manager.clusters,
            name='correlograms')
            
        self.add_visual(RectanglesVisual, coordinates=(0.,0.,0.,0.),
            color=(0.,0.,0.,1.), name='clusterinfo_bg', visible=False,
            depth=-.99, is_static=True)
        
        self.add_visual(TextVisual, text='0', name='clusterinfo', fontsize=16,
            posoffset=(.12, -.12),
            letter_spacing=200.,
            depth=-1,
            visible=False)
        
    def update(self):
        self.reinitialize_visual(
            size=self.data_manager.position.shape[0],
            nclusters=self.data_manager.nclusters,
            # nsamples=self.data_manager.nsamples,
            nhistograms=self.data_manager.nhistograms,
            position=self.data_manager.position,
            color=self.data_manager.color,
            color_array_index=self.data_manager.color_array_index,
            clusters=self.data_manager.clusters,
            visual='correlograms')
            

class CorrelogramsInteractionManager(PlotInteractionManager):
    def initialize(self):
        self.register('ShowClosestCluster', self.show_closest_cluster)
        self.register(None, self.hide_closest_cluster)
            
    def hide_closest_cluster(self, parameter):
        self.paint_manager.set_data(visible=False, visual='clusterinfo')
        self.paint_manager.set_data(visible=False, visual='clusterinfo_bg')
        
    def show_closest_cluster(self, parameter):
        
        if self.data_manager.nclusters == 0:
            return
            
        self.cursor = None
        
        nav = self.get_processor('navigation')
        
        # window coordinates
        x, y = parameter
        # data coordinates
        xd, yd = nav.get_data_coordinates(x, y)
        
        

        margin = 0.05
        a = 1.0 / (self.data_manager.nclusters * (1 + 2 * margin))
        
        cx = int(((xd + 1) / (a * (1 + 2 * margin)) - 1) / 2 + .5)
        cy = int(((yd + 1) / (a * (1 + 2 * margin)) - 1) / 2 + .5)
        
        cx_rel = np.clip(cx, 0, self.data_manager.nclusters - 1)
        cy_rel = np.clip(cy, 0, self.data_manager.nclusters - 1)
        
        
        # color0 = self.data_manager.cluster_colors[cx_rel]
        # r, g, b = scolors.COLORMAP[color0,:]
        # color0 = (r, g, b, .75)
        
        color1 = self.data_manager.cluster_colors[cy_rel]
        r, g, b = scolors.COLORMAP[color1,:]
        color1 = (r, g, b, .75)
        
        cx = self.data_manager.clusters_unique[cx_rel]
        cy = self.data_manager.clusters_unique[cy_rel]
        
        
        text = "%d / %d" % (cx, cy)
        
        # update clusterinfo visual
        rect = (x-.06, y-.05, x+.24, y-.2)
        self.paint_manager.set_data(coordinates=rect, 
            visible=True,
            visual='clusterinfo_bg')
            
        self.paint_manager.set_data(coordinates=(xd, yd), #color=color1,
            text=text,
            visible=True,
            visual='clusterinfo')
        
            
class CorrelogramsBindings(SpikyBindings):
    def set_zoombox_keyboard(self):
        """Set zoombox bindings with the keyboard."""
        self.set('LeftClickMove', 'ZoomBox',
                    key_modifier='Shift',
                    param_getter=lambda p: (p["mouse_press_position"][0],
                                            p["mouse_press_position"][1],
                                            p["mouse_position"][0],
                                            p["mouse_position"][1]))

    def set_clusterinfo(self):
        self.set('Move', 'ShowClosestCluster', key_modifier='Shift',
            param_getter=lambda p:
            (p['mouse_position'][0], p['mouse_position'][1]))
    
    def initialize(self):
        super(CorrelogramsBindings, self).initialize()
        self.set_clusterinfo()
    
    
class CorrelogramsView(GalryWidget):
    def __init__(self, *args, **kwargs):
        format = QtOpenGL.QGLFormat()
        format.setSampleBuffers(True)
        kwargs['format'] = format
        super(CorrelogramsView, self).__init__(*args, **kwargs)

    def initialize(self):
        self.set_bindings(CorrelogramsBindings)
        self.set_companion_classes(paint_manager=CorrelogramsPaintManager,
            interaction_manager=CorrelogramsInteractionManager,
            data_manager=CorrelogramsDataManager,)
    
    def set_data(self, *args, **kwargs):
        self.data_manager.set_data(*args, **kwargs)
    
        if self.initialized:
            log_debug("Updating data for correlograms")
            self.paint_manager.update()
            self.updateGL()
        else:
            log_debug("Initializing data for correlograms")
    
    
class CorrelogramsWidget(VisualizationWidget):
    def create_view(self, dh):
        self.dh = dh
        view = CorrelogramsView(getfocus=False)
        view.set_data(histograms=dh.correlograms,
                      clusters_unique=self.dh.clusters_unique,
                      baselines=dh.baselines,
                      cluster_colors=dh.cluster_colors)
        return view
        
    def update_view(self, dh=None):
        if dh is not None:
            self.dh = dh
        self.view.set_data(histograms=self.dh.correlograms,
                      clusters_unique=self.dh.clusters_unique,
                      baselines=self.dh.baselines,
                      cluster_colors=self.dh.cluster_colors)

    def initialize_connections(self):
        ssignals.SIGNALS.ClusterSelectionChanged.connect(self.slotClusterSelectionChanged, QtCore.Qt.UniqueConnection)
    
    def slotClusterSelectionChanged(self, sender, clusters):
        self.update_view()
        
        