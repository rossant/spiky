from galry import *
from common import *
from colors import COLORMAP
import numpy.random as rdn

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


class HistogramDataManager(Manager):
    def set_data(self, histograms=None, #nclusters=None, 
        cluster_colors=None):
        
        # self.clusters = clusters
        # self.nclusters = nclusters
        self.histograms = histograms
        self.nhistograms, self.nbins = histograms.shape
        self.nprimitives = self.nhistograms
        # index 0 = heterogeneous clusters, index>0 ==> cluster index + 1
        # self.cluster_colors = np.vstack((np.array([1.,1.,1.]), cluster_colors))
        self.cluster_colors = cluster_colors
        
        # print histograms
        # print histograms.shape
        # print cluster_colors
        
        # one histogram per cluster pair (i,j) 
        # assert self.nhistograms == self.nclusters * (self.nclusters + 1) / 2
        # deduce the number of clusters from the size of the histogram
        self.nclusters = int((-1 + np.sqrt(1 + 8 * self.nhistograms)) / 2.)
    
        # get the vertex positions
        X, Y = get_histogram_points(self.histograms)
        n = X.size
        self.nsamples = X.shape[1]
        
        # fill the data array
        self.position = np.empty((n, 2), dtype=np.float32)
        self.position[:,0] = X.ravel()
        self.position[:,1] = Y.ravel()
        
        # cluster i and j for each histogram in the view
        clusters = [(i,j) for i in xrange(self.nclusters) for j in xrange(self.nclusters) if j >= i]
        self.clusters = np.array(clusters, dtype=np.int32)
        # print clusters
        
        
        color_array_index = np.zeros(self.nhistograms, dtype=np.int32)
        
        # indices of histograms on the diagonal
        if self.nclusters:
            identity = self.clusters[:,0] == self.clusters[:,1]
        else:
            identity = []
        
        color_array_index[identity] = cluster_colors + 1
        
        # color_array_size = nclusters + 1
        self.color = np.vstack((np.ones((1, 3)), COLORMAP))
        
        # colormap, color_array_index
        
        self.color_array_index = color_array_index
        
        # print identity
        # print color_array_index
        
        self.clusters = np.repeat(self.clusters, self.nsamples, axis=0)
        self.color_array_index = np.repeat(self.color_array_index, self.nsamples, axis=0)
        
        
class HistogramVisual(PlotVisual):
    def initialize(self, nclusters=None, nhistograms=None, #nsamples=None,
        position=None, color=None, color_array_index=None, clusters=None):
        
        self.position_attribute_name = "transformed_position"
        
        super(HistogramVisual, self).initialize(
            position=position,
            nprimitives=nhistograms,
            color=color,
            color_array_index=color_array_index,
            )
            
        self.primitive_type = 'TRIANGLE_STRIP'
        # self.primitive_type = 'LINE_STRIP'
        
        self.add_attribute("cluster", vartype="int", ndim=2, data=clusters)
        self.add_uniform("nclusters", vartype="int", ndim=1, data=nclusters)
        
        self.add_vertex_main(VERTEX_SHADER)
        
        
class HistogramPaintManager(PaintManager):
    def initialize(self, **kwargs):
        
        # nclusters=None, nhistograms=None, nsamples=None,
        # position=None, color=None, color_array_index=None, clusters=None
        
        # create dataset
        self.add_visual(HistogramVisual,
            nclusters=self.data_manager.nclusters,
            # nsamples=self.data_manager.nsamples,
            nhistograms=self.data_manager.nhistograms,
            position=self.data_manager.position,
            color=self.data_manager.color,
            color_array_index=self.data_manager.color_array_index,
            clusters=self.data_manager.clusters,
            name='correlograms')
        
        
    def update(self):
        # color = self.data_manager.color
        # ncolors = color.shape[0]
        # ncomponents = color.shape[1]
        # color = color.reshape((1, ncolors, ncomponents))
        
        # size = self.data_manager.position.shape[0]
        # nsamples = size // self.data_manager.nprimitives
        # bounds = np.arange(0, size + 1, size // nsamples)
            
        # self.set_data(visual='correlograms', 
            # size=self.data_manager.position.shape[0],
            # nclusters=self.data_manager.nclusters,
            # # nsamples=self.data_manager.nsamples,
            # # nprimitives=self.data_manager.nhistograms,
            # position=self.data_manager.position,
            # colormap=color,
            # index=self.data_manager.color_array_index,
            # cluster=self.data_manager.clusters,
            # )
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


class HistogramBindings(SpikyDefaultBindingSet):
    pass


class CorrelogramsView(GalryWidget):
    def initialize(self):
        self.constrain_ratio = True
        self.constrain_navigation = True
        self.set_bindings(HistogramBindings)
        self.set_companion_classes(paint_manager=HistogramPaintManager,
            data_manager=HistogramDataManager,)
    
    def set_data(self, *args, **kwargs):
        self.data_manager.set_data(*args, **kwargs)
    
        if self.initialized:
            log_debug("Updating data for correlograms")
            self.paint_manager.update()
            self.updateGL()
        else:
            log_debug("Initializing data for correlograms")
    
    