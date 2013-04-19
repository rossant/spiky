"""Main window."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pprint
import os
import inspect
from collections import OrderedDict

import pandas as pd
import numpy as np
import numpy.random as rnd
from galry import QtGui, QtCore
from qtools import inprocess, inthread

import spiky.views as vw
from spiky.io.loader import KlustersLoader
import spiky.utils.logger as log
from spiky.utils.persistence import encode_bytearray, decode_bytearray
from spiky.utils.settings import SETTINGS
from spiky.utils.globalpaths import APPNAME
from spiky.gui.threads import ThreadedTasks


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
        
        self.loader = None
        
        # Create the views.
        self.create_views()
        self.create_actions()
        self.create_menu()
        self.create_threads()
        
        # Show the main window.
        self.set_styles()
        self.restore_geometry()
        self.show()
    
    def set_styles(self):
        # set stylesheet
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, "styles.css")
        with open(path, 'r') as f:
            stylesheet = f.read()
        stylesheet = stylesheet.replace('%ACCENT%', '#cdcdcd')
        stylesheet = stylesheet.replace('%ACCENT2%', '#a0a0a0')
        stylesheet = stylesheet.replace('%ACCENT3%', '#909090')
        stylesheet = stylesheet.replace('%ACCENT4%', '#cdcdcd')
        self.setStyleSheet(stylesheet)
    
    
    # Actions.
    # --------
    def create_actions(self):
        
        self.open_action = QtGui.QAction("&Open", self)
        self.open_action.triggered.connect(self.open_callback)
        self.open_action.setShortcut('Ctrl+O')
        
        self.quit_action = QtGui.QAction("&Quit", self)
        self.quit_action.triggered.connect(self.quit_callback)
        self.quit_action.setShortcut('Ctrl+Q')
        
    
    def create_menu(self):
        # File menu
        # ---------
        file_menu = self.menuBar().addMenu("&File")
        
        # file_menu.addAction(self.open_probe_action)
        # file_menu.addSeparator()
        
        file_menu.addAction(self.open_action)
        # file_menu.addAction(self.save_action)
        # file_menu.addAction(self.saveas_action)
        file_menu.addSeparator()
        
        # open last probe
        # self.open_last_probefile()
        
        # open last file
        # filename = SETTINGS.get('mainWindow/last_data_file', None)
        # if filename:
            # self.open_last_action = QtGui.QAction(filename, self)
            # self.open_last_action.setShortcut("CTRL+ALT+O")
            # self.open_last_action.triggered.connect(self.open_last_file, QtCore.Qt.UniqueConnection)
            # file_menu.addAction(self.open_last_action)
            # file_menu.addSeparator()
        
        file_menu.addAction(self.quit_action)
        
        
        # Views menu
        # ----------
        # views_menu = self.menuBar().addMenu("&Views")
        # views_menu.addAction(self.cluster_action)
        # views_menu.addAction(self.waveform_action)
        # views_menu.addAction(self.correlograms_action)
        # views_menu.addAction(self.correlationmatrix_action)
        # views_menu.addSeparator()
        # views_menu.addAction(self.override_color_action)
        
        
        # Actions menu
        # ------------
        # actions_menu = self.menuBar().addMenu("&Actions")
        # actions_menu.addAction(self.undo_action)
        # actions_menu.addAction(self.redo_action)
        # actions_menu.addSeparator()
        # actions_menu.addAction(self.merge_action)
        # actions_menu.addAction(self.split_action)
        # actions_menu.addSeparator()
        # actions_menu.addAction(self.move_to_mua_action)
        # actions_menu.addAction(self.move_to_noise_action)
    
    
    # Threads.
    # --------
    def create_threads(self):
        self.tasks = ThreadedTasks()
        self.tasks.open_task.dataOpened.connect(self.open_done)
    
    def join_threads(self):
         self.tasks.join()
    
    
    # View methods.
    # -------------
    def create_views(self):
        """Create all views at initialization."""
        
        # Create the default layout.
        self.views = {}
        self.views['ClusterView'] = self.add_view(vw.ClusterView,
            position=QtCore.Qt.LeftDockWidgetArea, closable=False)
        self.views['CorrelationMatrixView'] = self.add_view(vw.CorrelationMatrixView,
            position=QtCore.Qt.LeftDockWidgetArea,)
            
        self.splitDockWidget(
            self.views['ClusterView'].parentWidget(), 
            self.views['CorrelationMatrixView'].parentWidget(), 
            QtCore.Qt.Vertical
            )
            
        self.views['WaveformView'] = self.add_view(vw.WaveformView,
            position=QtCore.Qt.RightDockWidgetArea,)
        
        self.views['FeatureView'] = self.add_view(vw.FeatureView,
            position=QtCore.Qt.RightDockWidgetArea,)
            
        self.splitDockWidget(
            self.views['WaveformView'].parentWidget(), 
            self.views['FeatureView'].parentWidget(), 
            QtCore.Qt.Horizontal
            )
            
        self.views['CorrelogramsView'] = self.add_view(vw.CorrelogramsView,
            position=QtCore.Qt.RightDockWidgetArea,)
            
        self.splitDockWidget(
            self.views['FeatureView'].parentWidget(), 
            self.views['CorrelogramsView'].parentWidget(), 
            QtCore.Qt.Vertical
            )
    
    def add_view(self, view_class, position=None, 
        closable=True, **kwargs):
        """Add a widget to the main window."""
        view = view_class(self, getfocus=False)
        view.set_data(**kwargs)
        if not position:
            position = QtCore.Qt.LeftDockWidgetArea
            
        # Create the dock widget.
        dockwidget = QtGui.QDockWidget(view_class.__name__)
        dockwidget.setObjectName(view_class.__name__)
        dockwidget.setWidget(view)
        # dockwidget.view = view
        
        # Set dock widget options.
        if closable:
            options = (QtGui.QDockWidget.DockWidgetClosable | 
                QtGui.QDockWidget.DockWidgetFloatable | 
                QtGui.QDockWidget.DockWidgetMovable)
        else:
            options = (QtGui.QDockWidget.DockWidgetFloatable | 
                QtGui.QDockWidget.DockWidgetMovable)
            
        dockwidget.setFeatures(options)
        dockwidget.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea |
            QtCore.Qt.RightDockWidgetArea |
            QtCore.Qt.TopDockWidgetArea |
            QtCore.Qt.BottomDockWidgetArea)
            
        # Add the dock widget to the main window.
        self.addDockWidget(position, dockwidget)
        
        # return dockwidget
        return view
    
    
    # Update methods.
    # ---------------
    def update_cluster_view(self):
        """Update the cluster view using the data stored in the loader
        object."""
        data = dict(
            cluster_colors=self.loader.get_cluster_colors('all'),
            cluster_groups=self.loader.get_cluster_groups('all'),
            group_colors=self.loader.get_group_colors('all'),
            group_names=self.loader.get_group_names('all'),
            cluster_sizes=self.loader.get_cluster_sizes('all'),
        )
        self.views['ClusterView'].set_data(**data)
    
    
    # Callback functions.
    # -------------------
    def open_callback(self):
        folder = SETTINGS['main_window.last_data_dir']
        path = QtGui.QFileDialog.getOpenFileName(self, 
            "Open a file (.clu or other)", folder)[0]
        # If a file has been selected, open it.
        if path:
            # Launch the loading task in the background asynchronously.
            self.tasks.open_task.open(path)
            # Save the folder.
            folder = os.path.dirname(path)
            SETTINGS['main_window.last_data_dir'] = folder
            
            
        
    def quit_callback(self):
        self.close()
    
    
    # Task callbacks.
    # ---------------
    def open_done(self, loader):
        # Save the loader object.
        self.loader = loader
        # Update the views.
        self.update_cluster_view()
    
    
    # Geometry.
    # ---------
    def save_geometry(self):
        """Save the arrangement of the whole window."""
        SETTINGS['main_window.geometry'] = encode_bytearray(
            self.saveGeometry())
        SETTINGS['main_window.state'] = encode_bytearray(self.saveState())
        
    def restore_geometry(self):
        """Restore the arrangement of the whole window."""
        g = SETTINGS['main_window.geometry']
        s = SETTINGS['main_window.state']
        if g:
            self.restoreGeometry(decode_bytearray(g))
        if s:
            self.restoreState(decode_bytearray(s))
    
    
    # Event handlers.
    # ---------------
    def keyPressEvent(self, e):
        super(MainWindow, self).keyPressEvent(e)
        for view in self.views.values():
            view.keyPressEvent(e)
        
    def keyReleaseEvent(self, e):
        super(MainWindow, self).keyReleaseEvent(e)
        for view in self.views.values():
            view.keyReleaseEvent(e)
            
    def closeEvent(self, e):
        # Save the window geometry when closing the software.
        self.save_geometry()
        
        self.join_threads()
        
        for view in self.views.values():
            if hasattr(view, 'closeEvent'):
                view.closeEvent(e)
        return super(MainWindow, self).closeEvent(e)
            
            
            
    def sizeHint(self):
        return QtCore.QSize(1200, 800)
        
        
        