"""Main window."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pprint
import inspect
from collections import OrderedDict

import pandas as pd
import numpy as np
import numpy.random as rnd
from galry import QtGui, QtCore

import spiky.views as vw
from spiky.io.loader import KlustersLoader
import spiky.utils.logger as log
from spiky.utils.settings import SETTINGS
from spiky.utils.globalpaths import APPNAME


# -----------------------------------------------------------------------------
# Main Window
# -----------------------------------------------------------------------------
class MainWindow(QtGui.QMainWindow):
    
    def __init__(self):
        super(MainWindow, self).__init__()
        
        # Main window options.
        self.move(50, 50)
        self.setWindowTitle(APPNAME.title())
        
        # Focus options.
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.setMouseTracking(True)
        
        # Dock widgets options.
        self.setDockNestingEnabled(True)
        self.setAnimated(False)
        
        # Create the views.
        self.create_views()
        
        self.show()
        
    def create_views(self):
        """Create all views at initialization."""
        
        # Create the default layout.
        self.views = {}
        self.views['ClusterView'] = self.add_view(vw.ClusterView,
            position=QtCore.Qt.LeftDockWidgetArea,)
        self.views['CorrelationMatrixView'] = self.add_view(vw.CorrelationMatrixView,
            position=QtCore.Qt.LeftDockWidgetArea,)
            
        self.splitDockWidget(
            self.views['ClusterView'], 
            self.views['CorrelationMatrixView'], 
            QtCore.Qt.Vertical
            )
            
        self.views['WaveformView'] = self.add_view(vw.WaveformView,
            position=QtCore.Qt.RightDockWidgetArea,)
        
        self.views['FeatureView'] = self.add_view(vw.FeatureView,
            position=QtCore.Qt.RightDockWidgetArea,)
            
        self.splitDockWidget(
            self.views['WaveformView'], 
            self.views['FeatureView'], 
            QtCore.Qt.Horizontal
            )
            
        self.views['CorrelogramsView'] = self.add_view(vw.CorrelogramsView,
            position=QtCore.Qt.RightDockWidgetArea,)
            
        self.splitDockWidget(
            self.views['FeatureView'], 
            self.views['CorrelogramsView'], 
            QtCore.Qt.Vertical
            )
        
            
    def add_view(self, view_class, position=None, **kwargs):
        """Add a widget to the main window."""
        view = view_class(self, getfocus=False)
        view.set_data(**kwargs)
        if not position:
            position = QtCore.Qt.LeftDockWidgetArea
            
        # Create the dock widget.
        dockwidget = QtGui.QDockWidget(view_class.__name__)
        dockwidget.setObjectName(view_class.__name__)
        dockwidget.setWidget(view)
        
        # Set dock widget options.
        dockwidget.setFeatures(
            QtGui.QDockWidget.DockWidgetClosable | \
            QtGui.QDockWidget.DockWidgetFloatable | \
            QtGui.QDockWidget.DockWidgetMovable)
        dockwidget.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea |
            QtCore.Qt.RightDockWidgetArea |
            QtCore.Qt.TopDockWidgetArea |
            QtCore.Qt.BottomDockWidgetArea)
            
        # Add the dock widget to the main window.
        self.addDockWidget(position, dockwidget)
        
        return dockwidget
        
    def keyPressEvent(self, e):
        super(MainWindow, self).keyPressEvent(e)
        for view in self.views.values():
            view.keyPressEvent(e)
        
    def keyReleaseEvent(self, e):
        super(MainWindow, self).keyReleaseEvent(e)
        for view in self.views.values():
            view.keyReleaseEvent(e)
            
    def closeEvent(self, e):
        return super(MainWindow, self).closeEvent(e)
            
            
            
    def sizeHint(self):
        return QtCore.QSize(1200, 800)
        
        
        