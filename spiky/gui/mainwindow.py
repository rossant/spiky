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
# Dock widget class
# -----------------------------------------------------------------------------
class ViewDockWidget(QtGui.QDockWidget):
    closed = QtCore.pyqtSignal(object)
    
    def closeEvent(self, e):
        self.closed.emit(self)
        super(ViewDockWidget, self).closeEvent(e)


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
        # self.installEventFilter(EventFilter(self))
        
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
    def add_action(self, name, text, callback=None, shortcut=None):
        action = QtGui.QAction(text, self)
        if callback is None:
            callback = getattr(self, name + '_callback')
        # if callback:
        action.triggered.connect(callback)
        if shortcut:
            action.setShortcut(shortcut)
        setattr(self, name + '_action', action)
        
    def create_actions(self):
        self.add_action('open', '&Open', shortcut='Ctrl+O')
        
        # Open last file action
        path = SETTINGS['main_window.last_data_file']
        if path:
            lastfile = os.path.basename(path)
            if len(lastfile) > 30:
                lastfile = '...' + lastfile[-30:]
            self.add_action('open_last', 'Open &last ({0:s})'.format(
                lastfile), shortcut='Ctrl+Alt+O')
        else:
            self.add_action('open_last', 'Open &last', shortcut='Ctrl+Alt+O')
            self.open_last_action.setEnabled(False)
        
        self.add_action('quit', '&Quit', shortcut='Ctrl+Q')
        
        self.add_action('add_feature_view', 'Add FeatureView')
        self.add_action('add_waveform_view', 'Add WaveformView')
        self.add_action('add_correlation_matrix_view',
            'Add CorrelationMatrixView')
        self.add_action('add_correlograms_view', 'Add CorrelogramsView')
    
    def create_menu(self):
        # File menu.
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.open_last_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)
        
        # Views menu.
        views_menu = self.menuBar().addMenu("&Views")
        views_menu.addAction(self.add_feature_view_action)
        views_menu.addAction(self.add_waveform_view_action)
        views_menu.addAction(self.add_correlograms_view_action)
        views_menu.addAction(self.add_correlation_matrix_view_action)
        
    
    # Callback functions.
    # -------------------
    def open_callback(self, checked):
        # HACK: Force release of Ctrl key.
        self.force_key_release()
        
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
            SETTINGS['main_window.last_data_file'] = path
            
    def open_last_callback(self, checked):
        path = SETTINGS['main_window.last_data_file']
        if path:
            self.tasks.open_task.open(path)
            
    def quit_callback(self, checked):
        self.close()
    
    # Add views callbacks.
    def add_feature_view_callback(self, checked):
        self.add_feature_view()
        
    def add_waveform_view_callback(self, checked):
        self.add_waveform_view()
        
    def add_correlation_matrix_view_callback(self, checked):
        self.add_correlation_matrix_view()
        
    def add_correlograms_view_callback(self, checked):
        self.add_correlograms_view()
    
    # Clusters callbacks.
    def clusters_selected_callback(self, clusters):
        self.tasks.select_task.select(self.loader, clusters)
    
    
    # Task callbacks.
    # ---------------
    def open_done(self, loader):
        # Save the loader object.
        self.loader = loader
        # Update the views.
        self.update_cluster_view()
        
    def selection_done(self, clusters):
        # print len(self.loader.get_features())
        # print "done", clusters
        self.update_waveform_view()
        self.update_feature_view()
    
    
    # Threads.
    # --------
    def create_threads(self):
        self.tasks = ThreadedTasks()
        self.tasks.open_task.dataOpened.connect(self.open_done)
        self.tasks.select_task.clustersSelected.connect(self.selection_done)
    
    def join_threads(self):
         self.tasks.join()
    
    
    # View methods.
    # -------------
    def create_view(self, view_class, position=None, 
        closable=True, **kwargs):
        """Add a widget to the main window."""
        view = view_class(self, getfocus=False)
        view.set_data(**kwargs)
        if not position:
            position = QtCore.Qt.LeftDockWidgetArea
            
        # Create the dock widget.
        dockwidget = ViewDockWidget(view_class.__name__)
        dockwidget.setObjectName(view_class.__name__)
        dockwidget.setWidget(view)
        dockwidget.closed.connect(self.dock_widget_closed)
        
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
        
        # Return the view widget.
        return view
    
    def add_cluster_view(self):
        view = self.create_view(vw.ClusterView,
            position=QtCore.Qt.LeftDockWidgetArea, closable=False)
        # Connect callback function when selecting clusters.
        view.clustersSelected.connect(self.clusters_selected_callback)
        self.views['ClusterView'].append(view)
        
    def add_correlation_matrix_view(self):
        self.views['CorrelationMatrixView'].append(self.create_view(vw.CorrelationMatrixView,
            position=QtCore.Qt.LeftDockWidgetArea,))
    
    def add_waveform_view(self):
        self.views['WaveformView'].append(self.create_view(vw.WaveformView,
            position=QtCore.Qt.RightDockWidgetArea,))
        
    def add_feature_view(self):
        self.views['FeatureView'].append(self.create_view(vw.FeatureView,
            position=QtCore.Qt.RightDockWidgetArea,))
            
    def add_correlograms_view(self):
        self.views['CorrelogramsView'].append(self.create_view(vw.CorrelogramsView,
            position=QtCore.Qt.RightDockWidgetArea,))
            
    def get_view(self, name, index=0):
        views = self.views[name] 
        if not views:
            return None
        else:
            return views[index]
            
    def get_views(self, name):
        return self.views[name]
            
    def create_views(self):
        """Create all views at initialization."""
        
        # Create the default layout.
        self.views = dict(
            ClusterView=[],
            CorrelationMatrixView=[],
            WaveformView=[],
            FeatureView=[],
            CorrelogramsView=[],
            )
        
        self.add_cluster_view()
        self.add_correlation_matrix_view()
            
        self.splitDockWidget(
            self.get_view('ClusterView').parentWidget(), 
            self.get_view('CorrelationMatrixView').parentWidget(), 
            QtCore.Qt.Vertical
            )
            
        self.add_waveform_view()
        self.add_feature_view()
            
        self.splitDockWidget(
            self.get_view('WaveformView').parentWidget(), 
            self.get_view('FeatureView').parentWidget(), 
            QtCore.Qt.Horizontal
            )
            
        self.add_correlograms_view()
            
        self.splitDockWidget(
            self.get_view('FeatureView').parentWidget(), 
            self.get_view('CorrelogramsView').parentWidget(), 
            QtCore.Qt.Vertical
            )
    
    def dock_widget_closed(self, dock):
        for key in self.views.keys():
            views = self.views[key]
            for i in xrange(len(views)):
                if views[i].parent() == dock:
                    # self.views[view][i] = None
                    del self.views[key][i]
    
    
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
        self.get_view('ClusterView').set_data(**data)
    
    def update_waveform_view(self):
        data = dict(
            waveforms=self.loader.get_waveforms(),
            clusters=self.loader.get_clusters(),
            cluster_colors=self.loader.get_cluster_colors(),
            clusters_selected=self.loader.get_clusters_selected(),
            masks=self.loader.get_masks(),
            geometrical_positions=self.loader.get_probe(),
        )
        [view.set_data(**data) for view in self.get_views('WaveformView')]
    
    def update_feature_view(self):
        data = dict(
            features=self.loader.get_features(),
            masks=self.loader.get_masks(),
            clusters=self.loader.get_clusters(),
            clusters_selected=self.loader.get_clusters_selected(),
            cluster_colors=self.loader.get_cluster_colors(),
            nchannels=self.loader.nchannels,
            fetdim=self.loader.fetdim,
            nextrafet=self.loader.nextrafet,
        )
        [view.set_data(**data) for view in self.get_views('FeatureView')]
        
    
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
    def force_key_release(self):
        """HACK: force release of Ctrl when opening a dialog with a keyboard
        shortcut."""
        self.keyReleaseEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyRelease,
            QtCore.Qt.Key_Control, QtCore.Qt.NoModifier))
    
    def contextMenuEvent(self, e):
        """Disable the context menu in the main window."""
        return
        
    def keyPressEvent(self, e):
        super(MainWindow, self).keyPressEvent(e)
        for views in self.views.values():
            [view.keyPressEvent(e) for view in views]
        
    def keyReleaseEvent(self, e):
        super(MainWindow, self).keyReleaseEvent(e)
        for views in self.views.values():
            [view.keyReleaseEvent(e) for view in views]
            
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
        
        
        