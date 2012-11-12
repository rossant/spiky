from galry import *
from views import *
import tools
import numpy as np
from dataio import MockDataProvider
from tools import Info
from collections import OrderedDict
from widgets import *

SETTINGS = tools.init_settings()

__all__ = ['SpikyMainWindow']

class SpikyMainWindow(QtGui.QMainWindow):
    window_title = "Spiky"
    
    def __init__(self):
        super(SpikyMainWindow, self).__init__()
        # parameters related to docking
        self.setAnimated(False)
        self.setTabPosition(
            QtCore.Qt.LeftDockWidgetArea |
            QtCore.Qt.RightDockWidgetArea |
            QtCore.Qt.TopDockWidgetArea |
            QtCore.Qt.BottomDockWidgetArea,
            QtGui.QTabWidget.North)
        self.setDockNestingEnabled(True)
        self.setWindowTitle(self.window_title)
        # make the UI initialization
        self.initialize()
        self.restore_geometry()
        self.show()
        
    def initialize(self):
        """Make the UI initialization."""
        
        # load mock data
        provider = MockDataProvider()
        self.dh = provider.load(nspikes=100)
        
        # central window, the dockable widgets are arranged around it
        self.add_central(FeatureWidget)
        self.add_dock(WaveformWidget, QtCore.Qt.RightDockWidgetArea)        
        self.add_dock(CorrelogramsWidget, QtCore.Qt.RightDockWidgetArea)
        self.add_dock(CorrelationMatrixWidget, QtCore.Qt.RightDockWidgetArea)
        self.add_dock(ClusterWidget, QtCore.Qt.RightDockWidgetArea)

    def add_dock(self, widget_class, position, name=None, minsize=None):
        """Add a dockable widget"""
        if name is None:
            name = widget_class.__name__
        widget = widget_class(self.dh)
        if minsize is not None:
            widget.setMinimumSize(*minsize)
        dockwidget = QtGui.QDockWidget(name)
        dockwidget.setObjectName(name)
        dockwidget.setWidget(widget)
        dockwidget.setFeatures(QtGui.QDockWidget.DockWidgetFloatable | \
            QtGui.QDockWidget.DockWidgetMovable)
        self.addDockWidget(position, dockwidget)
        
    def add_central(self, widget_class, name=None, minsize=None):
        """Add a central widget in the main window."""
        if name is None:
            name = widget_class.__name__
        widget = widget_class(self.dh)
        widget.setObjectName(name)
        if minsize is not None:
            widget.setMinimumSize(*minsize)
        self.setCentralWidget(widget)
        
    def save_geometry(self):
        """Save the arrangement of the whole window into a INI file."""
        SETTINGS.set("mainWindow/geometry", self.saveGeometry())
        SETTINGS.set("mainWindow/windowState", self.saveState())
        
    def restore_geometry(self):
        """Restore the arrangement of the whole window from a INI file."""
        g = SETTINGS.get("mainWindow/geometry")
        w = SETTINGS.get("mainWindow/windowState")
        if g:
            self.restoreGeometry(g)
        if w:
            self.restoreState(w)
        
    def closeEvent(self, e):
        """Automatically save the arrangement of the window when closing
        the window."""
        self.save_geometry()
        super(SpikyMainWindow, self).closeEvent(e)


if __name__ == '__main__':
    window = show_window(SpikyMainWindow)


