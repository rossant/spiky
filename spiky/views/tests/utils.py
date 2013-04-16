"""Utils for unit tests for views package."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

from galry import QtGui, QtCore, show_window

from spiky.io.loader import KlustersLoader


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def get_data():
    """Return a dictionary with data variables, after the fixture setup
    has been called."""
    # Mock data folder.
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '../../io/tests/mockdata')
    
    # Load data files.
    xmlfile = os.path.join(dir, 'test.xml')
    l = KlustersLoader(xmlfile)
    
    # Get full data sets.
    clusters_selected = [1, 3, 10]
    l.select(clusters=clusters_selected)
    
    data = dict(
        clusters_selected=clusters_selected,
        features=l.get_features(),
        masks=l.get_masks(),
        waveforms=l.get_waveforms(),
        correlograms=l.get_correlograms(),
        correlation_matrix=l.get_correlation_matrix(),
        clusters=l.get_clusters(),
        cluster_colors=l.get_cluster_colors(),
        cluster_colors_full=l.get_cluster_colors('all'),
        spiketimes=l.get_spiketimes(),
        geometrical_positions=l.get_probe(),
        nchannels=l.nchannels,
        nsamples=l.nsamples,
        fetdim=l.fetdim,
        nextrafet=l.nextrafet,
        ncorrbins=l.ncorrbins,
    )
    
    return data

def show_view(view_class, **kwargs):
    # Display a view.
    class TestWindow(QtGui.QMainWindow):
        def __init__(self):
            super(TestWindow, self).__init__()
            self.setFocusPolicy(QtCore.Qt.WheelFocus)
            self.setMouseTracking(True)
            self.view = view_class(self, getfocus=False)
            self.view.set_data(**kwargs)
            self.setCentralWidget(self.view)
            self.show()
            
        def keyPressEvent(self, e):
            super(TestWindow, self).keyPressEvent(e)
            self.view.keyPressEvent(e)
            if e.key() == QtCore.Qt.Key_Q:
                self.close()
            
        def keyReleaseEvent(self, e):
            super(TestWindow, self).keyReleaseEvent(e)
            self.view.keyReleaseEvent(e)
                
    show_window(TestWindow)
    