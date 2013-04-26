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
from spiky.control.controller import Controller
from spiky.io.tools import get_array
from spiky.io.loader import KlustersLoader
from spiky.stats.cache import StatsCache
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
        
        # Initialize some variables.
        self.loader = None
        self.controller = None
        self.spikes_highlighted = []
        self.spikes_selected = []
        
        # Create the main window.
        self.create_views()
        self.create_file_actions()
        self.create_view_actions()
        self.create_control_actions()
        self.create_robot_actions()
        self.create_menu()
        self.create_threads()
        
        # Update action enabled/disabled property.
        self.update_action_enabled()
        
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
        
    def create_file_actions(self):
        # Open actions.
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
        
        # Quit action.
        self.add_action('quit', '&Quit', shortcut='Ctrl+Q')
        
    def create_view_actions(self):
        self.add_action('add_feature_view', 'Add FeatureView')
        self.add_action('add_waveform_view', 'Add WaveformView')
        self.add_action('add_correlation_matrix_view',
            'Add CorrelationMatrixView')
        self.add_action('add_correlograms_view', 'Add CorrelogramsView')
    
    def create_control_actions(self):
        self.add_action('undo', '&Undo', shortcut='Ctrl+Z')
        self.add_action('redo', '&Redo', shortcut='Ctrl+Y')
        
        self.add_action('merge', '&Merge', shortcut='Ctrl+G')
        self.add_action('split', '&Split', shortcut='Ctrl+K')

    def create_robot_actions(self):
        self.add_action('previous_clusters', '&Previous clusters', 
            shortcut='CTRL+Space')
        self.add_action('next_clusters', '&Next clusters', 
            shortcut='Space')
        
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
        
        # Actions menu.
        actions_menu = self.menuBar().addMenu("&Actions")
        actions_menu.addAction(self.undo_action)
        actions_menu.addAction(self.redo_action)
        actions_menu.addSeparator()
        actions_menu.addAction(self.merge_action)
        actions_menu.addAction(self.split_action)
        
        # Robot menu.
        robot_menu = self.menuBar().addMenu("&Robot")
        robot_menu.addAction(self.previous_clusters_action)
        robot_menu.addAction(self.next_clusters_action)
        
    def update_action_enabled(self):
        self.undo_action.setEnabled(self.can_undo())
        self.redo_action.setEnabled(self.can_redo())
        self.merge_action.setEnabled(self.can_merge())
        self.split_action.setEnabled(self.can_split())
    
    def can_undo(self):
        if self.controller is None:
            return False
        return self.controller.can_undo()
    
    def can_redo(self):
        if self.controller is None:
            return False
        return self.controller.can_redo()
    
    def can_merge(self):
        cluster_view = self.get_view('ClusterView')
        clusters = cluster_view.selected_clusters()
        return len(clusters) >= 2
        
    def can_split(self):
        cluster_view = self.get_view('ClusterView')
        clusters = cluster_view.selected_clusters()
        spikes_selected = self.spikes_selected
        return len(spikes_selected) >= 1
    
    
    # File menu callbacks.
    # --------------------
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
    
    
    # Views menu callbacks.
    # ---------------------
    def add_feature_view_callback(self, checked):
        self.add_feature_view()
        
    def add_waveform_view_callback(self, checked):
        self.add_waveform_view()
        
    def add_correlation_matrix_view_callback(self, checked):
        self.add_correlation_matrix_view()
        
    def add_correlograms_view_callback(self, checked):
        self.add_correlograms_view()
    
    
    # Actions callbacks.
    # ------------------
    def update_cluster_selection(self, clusters_selected):
        self.update_action_enabled()
        self.update_cluster_view()
        self.get_view('ClusterView').select(clusters_selected)
    
    def action_processed(self, action, to_select=[], to_invalidate=[],
        to_compute=None, group_to_select=None):
        """Called after an action has been processed. Used to update the 
        different views and launch tasks."""
        if isinstance(to_select, (int, long)):
            to_select = [to_select]
        if isinstance(to_invalidate, (int, long)):
            to_invalidate = [to_invalidate]
        if isinstance(to_compute, (int, long)):
            to_compute = [to_compute]
        # Select clusters to be selected.
        if len(to_select) > 0:
            self.update_cluster_selection(to_select)
        # Invalidate clusters.
        if len(to_invalidate) > 0:
            self.statscache.invalidate(to_invalidate)
        # Compute the correlation matrix for the requested clusters.
        if to_compute is not None:
            self.start_compute_correlation_matrix(to_compute)
        
    def merge_callback(self, checked):
        cluster_view = self.get_view('ClusterView')
        clusters = cluster_view.selected_clusters()
        if len(clusters) >= 2:
            action, output = self.controller.merge_clusters(clusters)
            self.action_processed(action, **output)
            
    def split_callback(self, checked):
        cluster_view = self.get_view('ClusterView')
        clusters = cluster_view.selected_clusters()
        spikes_selected = self.spikes_selected
        if len(spikes_selected) >= 1:
            action, output = self.controller.split_clusters(
                clusters, spikes_selected)
            self.action_processed(action, **output)
            
    def undo_callback(self, checked):
        action, output = self.controller.undo()
        self.action_processed(action, **output)
        
    def redo_callback(self, checked):
        action, output = self.controller.redo()
        self.action_processed(action, **output)
        
    def cluster_color_changed_callback(self, cluster, color):
        action, output = self.controller.change_cluster_color(cluster, color)
        self.action_processed(action, **output)
        
    def group_color_changed_callback(self, group, color):
        action, output = self.controller.change_group_color(group, color)
        self.action_processed(action, **output)
        
    def group_renamed_callback(self, group, name):
        action, output = self.controller.rename_group(group, name)
        self.action_processed(action, **output)
        
    def clusters_moved_callback(self, clusters, group):
        action, output = self.controller.move_clusters(clusters, group)
        self.action_processed(action, **output)
        
    def group_removed_callback(self, group):
        action, output = self.controller.remove_group(group)
        self.action_processed(action, **output)
        
    def group_added_callback(self, group, name, color):
        action, output = self.controller.add_group(group, name, color)
        self.action_processed(action, **output)
    
    
    # Selection callbacks.
    # --------------------
    def clusters_selected_callback(self, clusters):
        # Launch cluster selection on the Loader in an external thread.
        self.tasks.select_task.select(self.loader, clusters)
        
    def cluster_pair_selected_callback(self, clusters):
        """Callback when the user clicks on a pair in the
        CorrelationMatrixView."""
        self.get_view('ClusterView').select(clusters)
    
    
    # Views callbacks.
    # ----------------
    def waveform_spikes_highlighted_callback(self, spikes):
        self.spikes_highlighted = spikes
        self.get_view('FeatureView').highlight_spikes(get_array(spikes))
        
    def features_spikes_highlighted_callback(self, spikes):
        self.spikes_highlighted = spikes
        self.get_view('WaveformView').highlight_spikes(get_array(spikes))
        
    def features_spikes_selected_callback(self, spikes):
        self.spikes_selected = spikes
        self.update_action_enabled()
        self.get_view('WaveformView').highlight_spikes(get_array(spikes))
        
    def waveform_box_clicked_callback(self, coord, cluster, channel):
        self.get_view('FeatureView').set_projection(coord, channel, coord)
        
    
    # Task methods.
    # -------------
    def open_done(self, loader):
        # Save the loader object.
        self.loader = loader
        # Create the Controller.
        self.controller = Controller(self.loader)
        # Create the cache for the cluster statistics that need to be
        # computed in the background.
        self.statscache = StatsCache(loader.ncorrbins)
        # Start computing the correlation matrix.
        self.start_compute_correlation_matrix()
        # Update the robot.
        self.initialize_robot()
        # Update the views.
        self.update_cluster_view()
        
    def start_compute_correlograms(self, clusters_selected):
        # Get the correlograms parameters.
        spiketimes = get_array(self.loader.get_spiketimes('all'))
        clusters = get_array(self.loader.get_clusters('all'))
        clusters_all = self.loader.get_clusters_unique()
        bin = self.loader.corrbin
        halfwidth = self.loader.ncorrbins * bin / 2
        
        # Add new cluster indices if needed.
        # self.statscache.correlograms.add_indices(clusters_selected)
        
        # Get cluster indices that need to be updated.
        clusters_to_update = (self.statscache.correlograms.
            not_in_key_indices(clusters_selected))
        
        # If there are pairs that need to be updated, launch the task.
        if len(clusters_to_update) > 0:
            # Launch the task.
            self.tasks.correlograms_task.compute(spiketimes, clusters,
                clusters_to_update, halfwidth=halfwidth, bin=bin)    
        # Otherwise, update directly the correlograms view without launching
        # the task in the external process.
        else:
            self.update_correlograms_view()
        
    def start_compute_correlation_matrix(self, clusters_to_update=None):
        # Get the correlation matrix parameters.
        features = get_array(self.loader.get_features('all'))
        masks = get_array(self.loader.get_masks('all', full=True))
        clusters = get_array(self.loader.get_clusters('all'))
        clusters_all = self.loader.get_clusters_unique()
        # Get cluster indices that need to be updated.
        if clusters_to_update is None:
            clusters_to_update = (self.statscache.correlation_matrix.
                not_in_key_indices(clusters_all))
        # If there are pairs that need to be updated, launch the task.
        if len(clusters_to_update) > 0:
            # Launch the task.
            self.tasks.correlation_matrix_task.compute(features,
                clusters, masks, clusters_to_update)
        # Otherwise, update directly the correlograms view without launching
        # the task in the external process.
        else:
            self.update_correlation_matrix_view()
        
    def selection_done(self, clusters_selected):
        """Called on the main thread once the clusters have been loaded 
        in the main thread."""
        # Update action enabled/disabled property.
        self.update_action_enabled()
        # Launch the computation of the correlograms.
        self.start_compute_correlograms(clusters_selected)
        # Update the different views.
        self.update_waveform_view()
        self.update_feature_view()
    
    def correlograms_computed(self, clusters, correlograms):
        # Put the computed correlograms in the cache.
        self.statscache.correlograms.update(clusters, correlograms)
        # Update the robot.
        # self.tasks.robot_task.update(
            # correlograms=self.statscache.correlograms)
        # Update the view.
        self.update_correlograms_view()
    
    def correlation_matrix_computed(self, clusters, matrix):
        self.statscache.correlation_matrix.update(clusters, matrix)
        # Update the robot.
        self.update_robot()
        # Update the view.
        self.update_correlation_matrix_view()
    
    
    # Robot.
    # ------
    def initialize_robot(self):
        self.tasks.robot_task.set_data(
            # Data.
            features=self.loader.get_features('all'),
            spiketimes=self.loader.get_spiketimes('all'),
            masks=self.loader.get_masks('all'),
            clusters=self.loader.get_clusters('all'),
            cluster_groups=self.loader.get_cluster_groups('all'),
            # Statistics.
            correlograms=self.statscache.correlograms,
            correlation_matrix=self.statscache.correlation_matrix,
            )
    
    def update_robot(self):
        self.tasks.robot_task.set_data(
            clusters=self.loader.get_clusters('all'),
            correlograms=self.statscache.correlograms,
            correlation_matrix=self.statscache.correlation_matrix,
            )
            
    def previous_clusters_callback(self, checked):
        clusters =  self.tasks.robot_task.previous(
            _sync=True)[2]['_result']
        # log.info("The robot proposes clusters {0:s}.".format(str(clusters)))
        self.get_view('ClusterView').select(clusters)
            
    def next_clusters_callback(self, checked):
        clusters =  self.tasks.robot_task.next(
            _sync=True)[2]['_result']
        log.info("The robot proposes clusters {0:s}.".format(str(clusters)))
        self.get_view('ClusterView').select(clusters)
        
    
    # Threads.
    # --------
    def create_threads(self):
        # Create the external threads.
        self.tasks = ThreadedTasks()
        self.tasks.open_task.dataOpened.connect(self.open_done)
        self.tasks.select_task.clustersSelected.connect(self.selection_done)
        self.tasks.correlograms_task.correlogramsComputed.connect(
            self.correlograms_computed)
        self.tasks.correlation_matrix_task.correlationMatrixComputed.connect(
            self.correlation_matrix_computed)
    
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
            
        # Connect callback functions.
        view.clustersSelected.connect(self.clusters_selected_callback)
        view.clusterColorChanged.connect(self.cluster_color_changed_callback)
        view.groupColorChanged.connect(self.group_color_changed_callback)
        view.groupRenamed.connect(self.group_renamed_callback)
        view.clustersMoved.connect(self.clusters_moved_callback)
        view.groupAdded.connect(self.group_added_callback)
        view.groupRemoved.connect(self.group_removed_callback)
        
        self.views['ClusterView'].append(view)
        
    def add_correlation_matrix_view(self):
        view = self.create_view(vw.CorrelationMatrixView,
            position=QtCore.Qt.LeftDockWidgetArea,)
        view.pairSelected.connect(self.cluster_pair_selected_callback)
        self.views['CorrelationMatrixView'].append(view)
    
    def add_waveform_view(self):
        view = self.create_view(vw.WaveformView,
            position=QtCore.Qt.RightDockWidgetArea,)
        view.spikesHighlighted.connect(
            self.waveform_spikes_highlighted_callback)
        view.boxClicked.connect(self.waveform_box_clicked_callback)
        self.views['WaveformView'].append(view)
        
    def add_feature_view(self):
        view = self.create_view(vw.FeatureView,
            position=QtCore.Qt.RightDockWidgetArea,)
        view.spikesHighlighted.connect(
            self.features_spikes_highlighted_callback)
        view.spikesSelected.connect(
            self.features_spikes_selected_callback)
        self.views['FeatureView'].append(view)
            
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
        
    def update_correlograms_view(self):
        clusters_selected = self.loader.get_clusters_selected()
        try:
            correlograms = self.statscache.correlograms.submatrix(
                clusters_selected)
            data = dict(
                correlograms=correlograms,
                clusters_selected=clusters_selected,
                cluster_colors=self.loader.get_cluster_colors(),
            )
            [view.set_data(**data) for view in self.get_views('CorrelogramsView')]
        except IndexError:
            log.debug("Skip update correlograms view.")
    
    def update_correlation_matrix_view(self):
        matrix = self.statscache.correlation_matrix
        # Clusters in groups 0 or 1 to hide.
        cluster_groups = self.loader.get_cluster_groups('all')
        clusters_hidden = np.nonzero(np.in1d(cluster_groups, [0, 1]))[0]
        data = dict(
            correlation_matrix=matrix.to_array(),
            cluster_colors_full=self.loader.get_cluster_colors('all'),
            clusters_hidden=clusters_hidden,
        )
        [view.set_data(**data) 
            for view in self.get_views('CorrelationMatrixView')]
    
    
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
        
        
        