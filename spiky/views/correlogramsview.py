from galry import *
import numpy.random as rdn

VERTEX_SHADER = """
    //vec3 color = vec3(1, 1, 1);

    float margin = 0.05;
    float a = 1.0 / (nclusters * (1 + 2 * margin));
    
    vec2 box_position = vec2(0, 0);
    box_position.x = -1 + a * (1 + 2 * margin) * (2 * cluster.y + 1);
    box_position.y = 1 - a * (1 + 2 * margin) * (2 * cluster.x + 1);
    
    vec2 transformed_position = box_position + a * position;
"""


def get_histogram_points(hist):
    """Tesselates histograms.
    
    Arguments:
      * hist: a N x Nsamples array, where each line contains an histogram.
      
    Returns:
      * X, Y: two N x (5*Nsamples+1) arrays with the coordinates of the
        histograms, a
      
    """
    n, nsamples = hist.shape
    dx = 2. / nsamples
    
    x0 = -1 + dx * np.arange(nsamples)
    
    x = np.zeros((n, 5 * nsamples + 1))
    y = -np.ones((n, 5 * nsamples + 1))
    
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
    def set_data(self, histograms=None, nclusters=None, cluster_colors=None):
        # self.clusters = clusters
        # self.nclusters = nclusters
        self.histograms = histograms
        self.nhistograms, self.nbins = histograms.shape
        self.nprimitives = self.nhistograms
        # index 0 = heterogeneous clusters, index>0 ==> cluster index + 1
        self.cluster_colors = np.vstack((np.array([1.,1.,1.]), cluster_colors))
        
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
        
        
class HistogramVisual(PlotVisual):
    def initialize(self, nclusters=None, nhistograms=None, nsamples=None,
        position=None, color=None):
        
        self.size = position.shape[0]
        self.primitive_type = 'TRIANGLE_STRIP'
        
        self.position_attribute_name = "transformed_position"
        
        # get the cluster indices
        # clusters = np.zeros((nhistograms * nsamples, 2), dtype=np.float32)
        clusters = [(i,j) for i in xrange(nclusters) for j in xrange(nclusters) if j <= i]
        clusters = np.array(clusters, dtype=np.int32)
        # indices of histograms on the diagonal
        clusters = np.repeat(clusters, nsamples, axis=0)
        if nclusters:
            identity = clusters[:,0] == clusters[:,1]
        else:
            identity = []
        # for each histogram, index of the color
        color_array_index = np.zeros(self.size, dtype=np.int32)
        color_array_index[identity] = np.array(np.repeat(np.arange(1, nclusters + 1),
            nsamples), np.int32)
        # size of the array with all colors: one color per cluster + one for
        # the off-diagonal
        color_array_size = nclusters + 1
        
        
        super(HistogramVisual, self).initialize(position=position, color=color,
            nprimitives=nhistograms, color_array_index=color_array_index,
            # position_attribute_name="transformed_position"
            )
        
        self.add_attribute("cluster", vartype="int", ndim=2, data=clusters)
        self.add_uniform("nclusters", vartype="int", ndim=1, data=nclusters)
        
        # call the parent initialize with the right parameters
        # kwargs.update(nprimitives=nhistograms, nsamples=nsamples,
            # single_color=False, colors_ndim=3,
            # use_color_array=True,
            # color_array_index=color_array_index,
            # color_array_size=color_array_size,
            # )
        # super(HistogramVisual, self).initialize(position=position, color=color)
    
        self.add_vertex_main(VERTEX_SHADER)
        
class HistogramPaintManager(PaintManager):
    def initialize(self, **kwargs):
        
        # create dataset
        self.add_visual(HistogramVisual,
            nclusters=self.data_manager.nclusters,
            nsamples=self.data_manager.nsamples,
            nhistograms=self.data_manager.nhistograms,
            color=self.data_manager.cluster_colors,
            # cluster=self.data_manager.clusters,
            position=self.data_manager.position,
            name='histogram')
        

class CorrelogramsView(GalryWidget):
    def initialize(self):
        self.constrain_ratio = True
        self.constrain_navigation = True
        self.set_companion_classes(paint_manager=HistogramPaintManager,
            data_manager=HistogramDataManager)
    
    def set_data(self, *args, **kwargs):
        self.data_manager.set_data(*args, **kwargs)
    
    
    