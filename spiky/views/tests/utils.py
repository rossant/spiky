"""Utils for unit tests for views package."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import threading
import time

from galry import QtGui, QtCore, show_window

from spiky.io.loader import KlustersLoader
from spiky.utils.userpref import USERPREF


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
        clusters=l.get_clusters(),
        
        cluster_colors=l.get_cluster_colors(),
        cluster_colors_full=l.get_cluster_colors('all'),
        
        cluster_groups=l.get_cluster_groups('all'),
        group_colors=l.get_group_colors('all'),
        group_names=l.get_group_names('all'),
        cluster_sizes=l.get_cluster_sizes('all'),
        
        spiketimes=l.get_spiketimes(),
        geometrical_positions=l.get_probe(),
        
        correlation_matrix=l.get_correlation_matrix(),
        
        nchannels=l.nchannels,
        nsamples=l.nsamples,
        fetdim=l.fetdim,
        nextrafet=l.nextrafet,
        ncorrbins=l.ncorrbins,
    )
    
    return data

def show_view(view_class, **kwargs):
    
    operators = kwargs.pop('operators', None)
    
    # Display a view.
    class TestWindow(QtGui.QMainWindow):
        operatorStarted = QtCore.pyqtSignal(int)
        
        def __init__(self):
            super(TestWindow, self).__init__()
            self.setFocusPolicy(QtCore.Qt.WheelFocus)
            self.setMouseTracking(True)
            self.view = view_class(self, getfocus=False)
            self.view.set_data(**kwargs)
            self.setCentralWidget(self.view)
            self.show()
            
            # Start "operator" asynchronously in the main thread.
            if operators:
                self.operator_list = operators
                self.operatorStarted.connect(self.operator)
                self._thread = threading.Thread(target=self._run_operator)
                self._thread.start()
            
        def _run_operator(self):
            for i in xrange(len(self.operator_list)):
                # Call asynchronously operation #i, after a given delay.
                if type(self.operator_list[i]) == tuple:
                    dt = self.operator_list[i][1]
                else:
                    # Default delay.
                    dt = USERPREF['test_operator_delay'] or .5
                time.sleep(dt)
                self.operatorStarted.emit(i)
            
        def operator(self, i):
            # Execute operation #i.
            if type(self.operator_list[i]) == tuple:
                fun = self.operator_list[i][0]
            else:
                fun = self.operator_list[i]
            fun(self)
            
        def keyPressEvent(self, e):
            super(TestWindow, self).keyPressEvent(e)
            self.view.keyPressEvent(e)
            if e.key() == QtCore.Qt.Key_Q:
                self.close()
            
        def keyReleaseEvent(self, e):
            super(TestWindow, self).keyReleaseEvent(e)
            self.view.keyReleaseEvent(e)
                
        def closeEvent(self, e):
            if operators:
                self._thread.join()
            return super(TestWindow, self).closeEvent(e)
                
    show_window(TestWindow)
    