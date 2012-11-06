from galry import *
from views import *
from dataio import MockDataProvider


class VisualizationWidget(QtGui.QWidget):
    def __init__(self, dataholder):
        super(VisualizationWidget, self).__init__()
        self.dataholder = dataholder
        self.view = self.create_view(dataholder)
        self.controller = self.create_controller()
        self.initialize()

    def create_view(self, dataholder):
        """Create the view and return it.
        The view must be an instance of a class deriving from `QWidget`.
        
        To be overriden."""
        return None

    def create_controller(self):
        """Create the controller and return it.
        The controller must be an instance of a class deriving from `QLayout`.
        
        To be overriden."""
        hbox = QtGui.QHBoxLayout()
        b = QtGui.QPushButton("OK")
        c = QtGui.QPushButton("Cancel")
        hbox.addWidget(b)
        hbox.addWidget(c)
        return hbox
        
    def initialize(self):
        """Initialize the user interface.
        
        By default, add the controller at the top, and the view at the bottom.
        
        To be overriden."""
        # put the controller and the view vertically
        vbox = QtGui.QVBoxLayout()
        # add the controller (which must be a layout)
        vbox.addLayout(self.controller)
        # add the view (which must be a widget, typically deriving from
        # GalryWidget)
        vbox.addWidget(self.view)
        # set the VBox as layout of the widget
        self.setLayout(vbox)
        

        
        
class WaveformWidget(VisualizationWidget):
    def create_view(self, dh):
        view = WaveformView()
        view.set_data(dh.waveforms,
                      clusters=dh.clusters,
                      cluster_colors=dh.clusters_info.colors,
                      geometrical_positions=dh.probe.positions,
                      masks=dh.masks)
        return view

    
class FeatureWidget(VisualizationWidget):
    def create_view(self, dh):
        view = FeatureView()
        view.set_data(dh.features, clusters=dh.clusters,
                      fetdim=3,
                      cluster_colors=dh.clusters_info.colors,
                      masks=dh.masks)
        return view

    
class CorrelogramsWidget(VisualizationWidget):
    def create_view(self, dh):
        view = CorrelogramsView()
        view.set_data(histograms=dh.correlograms,
                        # nclusters=dh.nclusters,
                        cluster_colors=dh.clusters_info.colors)
        return view

    
class CorrelationMatrixWidget(VisualizationWidget):
    def create_view(self, dh):
        view = CorrelationMatrixView()
        view.set_data(dh.correlationmatrix)
        return view




class SpikyMainWindow(QtGui.QMainWindow):
    
    def add_dock(self, widget_class, position, name=None):
        if name is None:
            name = widget_class.__name__
        dockwidget = QtGui.QDockWidget(name)
        dockwidget.setWidget(widget_class(self.dh))
        self.addDockWidget(position, dockwidget)
        
    def __init__(self):
        super(SpikyMainWindow, self).__init__()
        
        # load mock data
        provider = MockDataProvider()
        self.dh = provider.load(nspikes=1000)
        
        self.add_dock(WaveformWidget, QtCore.Qt.LeftDockWidgetArea)
        # self.add_dock(CorrelogramsWidget, QtCore.Qt.LeftDockWidgetArea)
        # self.add_dock(CorrelationMatrixWidget, QtCore.Qt.RightDockWidgetArea)
        self.add_dock(FeatureWidget, QtCore.Qt.RightDockWidgetArea)

        
        
        self.show()




if __name__ == '__main__':
    window = show_window(SpikyMainWindow)


