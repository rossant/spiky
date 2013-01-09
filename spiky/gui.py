import os
import re
from galry import *
# from views import *
# from views.signals import SIGNALS
import tools
from collections import OrderedDict
import numpy as np
# from dataio import *
# from tools import Info
# from spiky import get_icon, SIGNALS, Info
# from icons import get_icon
import inspect
import spiky.signals as ssignals
import spiky
import spiky.views as sviews
import spiky.dataio as sdataio


SETTINGS = tools.get_settings()

STYLESHEET = """
QStatusBar::item
{
    border: none;
}
"""

__all__ = ['SpikyMainWindow', 'show_window']



class DataUpdater(QtGui.QWidget):
    """"Handle data updating in the data holder, responding to signals
    emitted by widgets.
    
    When a widget wants to update the data, it raises a signal with a 
    "ToChange" postfix. This signal means that some part of the data needs
    to change. The only object handling these signals is the DataUpdater,
    which responds to them and updates the DataHolder accordingly.
    Then, the DataUpdater raises new "Changed"-postfixed signals, that can
    be handled by any widget.
    
    """
    def __init__(self, dh):
        super(DataUpdater, self).__init__()
        self.dh = dh
        self.initialize_connections()
        
    def initialize_connections(self):
        ssignals.SIGNALS.ProjectionToChange.connect(self.slotProjectionToChange, QtCore.Qt.UniqueConnection)
        ssignals.SIGNALS.ClusterSelectionToChange.connect(self.slotClusterSelectionToChange, QtCore.Qt.UniqueConnection)
        
    def slotClusterSelectionToChange(self, sender, clusters):
        self.dh.select_clusters(clusters)
        ssignals.emit(sender, 'ClusterSelectionChanged', clusters)
        
    def slotProjectionToChange(self, sender, coord, channel, feature):
        ssignals.emit(sender, 'ProjectionChanged', coord, channel, feature)
        

class SpikyMainWindow(QtGui.QMainWindow):
    window_title = "Spiky"
    
    def __init__(self):
        super(SpikyMainWindow, self).__init__()
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        
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
        
        # initialize the data holder and select holder
        self.initialize_data()
        # list all dock and central widgets
        self.allwidgets = []
        # initialize actions
        self.initialize_actions()
        # make the UI initialization
        self.initialize()
        # initialize menu
        self.initialize_menu()
        # initialize toolbar
        self.initialize_toolbar()
        # initialize all signals/slots connections between widgets
        self.initialize_connections()
        
        # set stylesheet
        self.setStyleSheet(STYLESHEET)
        # set empty status bar
        self.statusBar().addPermanentWidget(QtGui.QWidget(self))
        self.restore_geometry()
        # show the window
        self.show()

        
    # Widget creation methods
    # -----------------------
    def add_dock(self, widget_class, position, name=None, minsize=None):
        """Add a dockable widget"""
        if name is None:
            name = widget_class.__name__.replace("Widget", "View")
        widget = widget_class(self, self.sdh)#, getfocus=False)
        if minsize is not None:
            widget.setMinimumSize(*minsize)
        dockwidget = QtGui.QDockWidget(name)
        dockwidget.setObjectName(name)
        dockwidget.setWidget(widget)
        dockwidget.setFeatures(
            QtGui.QDockWidget.DockWidgetClosable | \
            QtGui.QDockWidget.DockWidgetFloatable | \
            QtGui.QDockWidget.DockWidgetMovable)
        self.addDockWidget(position, dockwidget)
        # if isinstance(widget, VisualizationWidget):
        self.allwidgets.append(widget)
        return widget, dockwidget
        
    def add_central(self, widget_class, name=None, minsize=None):
        """Add a central widget in the main window."""
        if name is None:
            name = widget_class.__name__
        widget = widget_class(self, self.sdh)#, getfocus=False)
        widget.setObjectName(name)
        if minsize is not None:
            widget.setMinimumSize(*minsize)
        self.setCentralWidget(widget)
        # if isinstance(widget, VisualizationWidget):
        self.allwidgets.append(widget)
        return widget
        
    
    # Initialization
    # --------------
    def initialize(self):
        """Make the UI initialization."""
        
        # create the DataUpdater, which handles the ToChange signals and
        # change data in the DataHolder.
        self.du = DataUpdater(self.sdh)
        self.am = spiky.ActionManager(self.dh)
        
        # central window, the dockable widgets are arranged around it
        self.cluster_widget, self.cluster_dock_widget = self.add_dock(sviews.ClusterWidget, QtCore.Qt.RightDockWidgetArea)
        # dock widgets
        self.feature_widget = self.add_central(sviews.FeatureWidget)
        self.waveform_widget, self.waveform_dock_widget = self.add_dock(sviews.WaveformWidget, QtCore.Qt.RightDockWidgetArea)        
        self.correlograms_widget, self.correlograms_dock_widget = self.add_dock(sviews.CorrelogramsWidget, QtCore.Qt.RightDockWidgetArea)
        
        # self.correlationmatrix_widget = self.add_dock(CorrelationMatrixWidget, QtCore.Qt.RightDockWidgetArea)
        
        # widget actions
        self.cluster_action = self.cluster_dock_widget.toggleViewAction()
        self.waveform_action = self.waveform_dock_widget.toggleViewAction()
        self.correlograms_action = self.correlograms_dock_widget.toggleViewAction()
        
    def initialize_data(self):
        # load mock data
        self.provider = sdataio.MockDataProvider()
        self.dh = self.provider.load(nspikes=0, nclusters=0)
        self.sdh = sdataio.SelectDataHolder(self.dh)
        
    def initialize_actions(self):
        """Initialize all global actions."""
        # automatic projection action
        self.autoproj_action = QtGui.QAction("Automatic projection", self)
        self.autoproj_action.setIcon(spiky.get_icon("magic"))
        self.autoproj_action.setShortcut("P")
        self.autoproj_action.setStatusTip("Automatically choose the best " +
            "projection in the FeatureView.")
        self.autoproj_action.triggered.connect(lambda e: ssignals.emit(self, "AutomaticProjection"), QtCore.Qt.UniqueConnection)
        
        # open action
        self.open_action = QtGui.QAction("&Open", self)
        self.open_action.setShortcut("CTRL+O")
        self.open_action.setIcon(spiky.get_icon("open"))
        self.open_action.triggered.connect(self.open_file, QtCore.Qt.UniqueConnection)
        
        # save action
        self.save_action = QtGui.QAction("&Save", self)
        self.save_action.setShortcut("CTRL+S")
        self.save_action.setIcon(spiky.get_icon("save"))
        self.save_action.triggered.connect(self.save_file, QtCore.Qt.UniqueConnection)
        
        # exit action
        self.quit_action = QtGui.QAction("E&xit", self)
        self.quit_action.setShortcut("CTRL+Q")
        self.quit_action.triggered.connect(self.close, QtCore.Qt.UniqueConnection)
        
        # merge action
        self.merge_action = QtGui.QAction("&Merge", self)
        self.merge_action.setIcon(spiky.get_icon("merge"))
        self.merge_action.setShortcut("M")
        self.merge_action.setEnabled(False)
        self.merge_action.triggered.connect(self.merge, QtCore.Qt.UniqueConnection)
        
        # merge action
        self.split_action = QtGui.QAction("&Split", self)
        self.split_action.setIcon(spiky.get_icon("split"))
        self.split_action.setShortcut("S")
        self.split_action.setEnabled(False)
        self.split_action.triggered.connect(self.split, QtCore.Qt.UniqueConnection)
        
        # undo action
        self.undo_action = QtGui.QAction("&Undo", self)
        self.undo_action.setShortcut("CTRL+Z")
        self.undo_action.setIcon(spiky.get_icon("undo"))
        self.undo_action.setEnabled(False)
        self.undo_action.triggered.connect(self.undo, QtCore.Qt.UniqueConnection)
        
        # redo action
        self.redo_action = QtGui.QAction("&Redo", self)
        self.redo_action.setShortcut("CTRL+Y")
        self.redo_action.setIcon(spiky.get_icon("redo"))
        self.redo_action.setEnabled(False)
        self.redo_action.triggered.connect(self.redo, QtCore.Qt.UniqueConnection)
        
    def initialize_menu(self):
        """Initialize the menu."""
        # File menu
        # ---------
        file_menu = self.menuBar().addMenu("&File")
        
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        
        # open last file
        filename = SETTINGS.get('mainWindow/last_data_file', None)
        if filename:
            self.open_last_action = QtGui.QAction(filename, self)
            self.open_last_action.setShortcut("CTRL+ALT+O")
            self.open_last_action.triggered.connect(self.open_last_file, QtCore.Qt.UniqueConnection)
            file_menu.addAction(self.open_last_action)
            file_menu.addSeparator()
        
        file_menu.addAction(self.quit_action)
        
        
        # Views menu
        # ----------
        views_menu = self.menuBar().addMenu("&Views")
        views_menu.addAction(self.cluster_action)
        views_menu.addAction(self.waveform_action)
        views_menu.addAction(self.correlograms_action)
        
        
        # Actions menu
        # ------------
        actions_menu = self.menuBar().addMenu("&Actions")
        actions_menu.addAction(self.undo_action)
        actions_menu.addAction(self.redo_action)
        actions_menu.addSeparator()
        actions_menu.addAction(self.merge_action)
        actions_menu.addAction(self.split_action)
        
    def initialize_toolbar(self):
        # self.toolbar = QtGui.QToolBar(self)
        self.toolbar = self.addToolBar("SpikyToolbar")
        self.toolbar.setObjectName("SpikyToolbar")
        self.toolbar.addAction(self.open_action)
        self.toolbar.addAction(self.save_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.merge_action)
        self.toolbar.addAction(self.split_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.undo_action)
        self.toolbar.addAction(self.redo_action)
    
        
    # Action methods
    # --------------
    def open_file(self, *args):
        self.reset_action_generator()
        folder = SETTINGS.get('mainWindow/last_data_dir')
        filename = QtGui.QFileDialog.getOpenFileName(self, "Open a file (.clu or other)", folder)[0]
        if filename:
            self.load_file(filename)
        
    def open_last_file(self, *args):
        filename = SETTINGS.get('mainWindow/last_data_file', None)
        self.load_file(filename)
        
    def save_file(self, *args):
        self.provider.save()
        
    def load_file(self, filename):
        r = re.search(r"([^\n]+)\.[^\.]+\.([0-9]+)$", filename)
        if r:
            # save last opened file
            SETTINGS.set('mainWindow/last_data_file', filename)
            filename = r.group(1)
            fileindex = int(r.group(2))
        else:
            log_warn(("The file could not be loaded because it is not like",
                " *.i.*"))
            return
        
        # save folder
        folder = os.path.dirname(filename)
        SETTINGS.set('mainWindow/last_data_dir', folder)
        
        self.provider = sdataio.KlustersDataProvider()
        self.dh = self.provider.load(filename, fileindex)
        self.sdh = sdataio.SelectDataHolder(self.dh)
        self.du = DataUpdater(self.sdh)
        self.am = spiky.ActionManager(self.dh)
        
        self.cluster_widget.update_view(self.sdh)
        self.feature_widget.update_view(self.sdh)
        self.waveform_widget.update_view(self.sdh)
        self.correlograms_widget.update_view(self.sdh)
        
    def merge(self):
        """Merge selected clusters."""
        action = self.am.do(spiky.MergeAction, self.sdh.get_clusters())
        self.cluster_widget.update_view(self.sdh)
        self.cluster_widget.view.select(action.new_cluster)
        self.undo_action.setEnabled(self.am.undo_enabled())
        self.redo_action.setEnabled(self.am.redo_enabled())
        
    def split(self):
        """Split selected spikes."""
        action = self.am.do(spiky.SplitAction, self.selected_spikes)
        self.cluster_widget.update_view(self.sdh)
        # select old and new clusters after split
        cl = np.hstack((action.clusters_to_split, action.new_clusters))
        self.cluster_widget.view.select_multiple(cl)
        self.undo_action.setEnabled(self.am.undo_enabled())
        self.redo_action.setEnabled(self.am.redo_enabled())
        self.feature_widget.view.process_interaction('CancelSelectionPoint')
        
    def undo(self):
        action = self.am.undo()
        if action is not None:
            self.cluster_widget.update_view(self.sdh)
            if isinstance(action, spiky.MergeAction):
                self.cluster_widget.view.select_multiple(action.clusters_to_merge)
            if isinstance(action, spiky.SplitAction):
                self.cluster_widget.view.select_multiple(action.clusters_to_split)
        self.undo_action.setEnabled(self.am.undo_enabled())
        self.redo_action.setEnabled(self.am.redo_enabled())
        
    def redo(self):
        action = self.am.redo()
        if action is not None:
            self.cluster_widget.update_view(self.sdh)
            if isinstance(action, spiky.MergeAction):
                self.cluster_widget.view.select(action.new_cluster)
            if isinstance(action, spiky.SplitAction):
                cl = np.hstack((action.clusters_to_split, action.new_clusters))
                self.cluster_widget.view.select_multiple(cl)
        self.undo_action.setEnabled(self.am.undo_enabled())
        self.redo_action.setEnabled(self.am.redo_enabled())
        
        
    # Event methods
    # -------------
    def redirect_event(self, event_name, e):
        for widget in self.allwidgets:
            getattr(widget.view, event_name)(e)
        
    def keyPressEvent(self, e):
        # self.last_key_press_event = e
        self.redirect_event('keyPressEvent', e)

    def keyReleaseEvent(self, e):
        self.redirect_event('keyReleaseEvent', e)

    def mousePressEvent(self, e):
        self.redirect_event('mousePressEvent', e)

    def mouseReleaseEvent(self, e):
        self.redirect_event('mouseReleaseEvent', e)

    def contextMenuEvent(self, e):
        return
           
    def reset_action_generator(self):
        for widget in self.allwidgets:
            if hasattr(widget.view, 'reset_action_generator'):
                widget.view.reset_action_generator()
           
    def focusOutEvent(self, e):
        self.reset_action_generator()
            
            
    # Signals
    # -------
    def initialize_connections(self):
        """Initialize the signals/slots connections between widgets."""
        ssignals.SIGNALS.HighlightSpikes.connect(self.slotHighlightSpikes, QtCore.Qt.UniqueConnection)
        ssignals.SIGNALS.ClusterSelectionChanged.connect(self.slotClusterSelectionChanged)
        ssignals.SIGNALS.SelectSpikes.connect(self.slotSelectSpikes)
        ssignals.SIGNALS.NewGroup.connect(self.slotNewGroup)
        ssignals.SIGNALS.DeleteGroup.connect(self.slotDeleteGroup)
        ssignals.SIGNALS.ClusterChangedGroup.connect(self.slotClusterChangedGroup)
        ssignals.SIGNALS.ClusterChangedColor.connect(self.slotClusterChangedColor)
        ssignals.SIGNALS.GroupChangedColor.connect(self.slotGroupChangedColor)
        ssignals.SIGNALS.RenameGroup.connect(self.slotRenameGroup)
        
        
    # Highlight slots
    #----------------
    def slotHighlightSpikes(self, sender, spikes):
        """Called whenever spikes are selected in a view.
        
        Arguments:
          * sender: the view which is at the origin of the signal emission.
          * spikes: a Numpy array of integers with the indices of highlighted
            spikes.
        
        """
        # TODO: better design with the slot being in the target widgets directly
        
        # highlighting occurred in the feature widget
        if hasattr(self, 'feature_widget'):
            if sender == self.feature_widget.view:
                if hasattr(self, 'waveform_widget'):
                    self.waveform_widget.view.highlight_spikes(spikes)
            
        # highlighting occurred in the waveform widget
        if hasattr(self, 'waveform_widget'):
            if sender == self.waveform_widget.view:
                if hasattr(self, 'feature_widget'):
                    self.feature_widget.view.highlight_spikes(spikes)
    
    def slotClusterSelectionChanged(self, sender, clusters):
        # enable or disable merge action as a function of the number of 
        # selected clusters
        if len(clusters) >= 2:
            self.merge_action.setEnabled(True)
        else:
            self.merge_action.setEnabled(False)
        # disable split when changing selection of clusters
        self.split_action.setEnabled(False)
        self.selected_spikes = None
    
    
    # Selection slots
    #----------------
    def slotSelectSpikes(self, sender, spikes):
        self.selected_spikes = spikes
        # print spikes
        # print self.dh.clusters[spikes]
        if len(spikes) >= 1:
            self.split_action.setEnabled(True)
        else:
            self.split_action.setEnabled(False)
    
    
    # Group slots
    #------------
    def slotNewGroup(self, sender, groupidx):
        name = "Group %d" % groupidx
        self.dh.clusters_info.groups_info.append(dict(name=name,
            groupidx=groupidx, coloridx=0))
    
    def slotRenameGroup(self, sender, groupidx, name):
        for grp in self.dh.clusters_info.groups_info:
            print grp['groupidx'], grp['name'], groupidx
            if grp['groupidx'] == groupidx:
                grp['name'] = name
                break
        self.cluster_widget.update_view(self.sdh)
        print
        
    def slotDeleteGroup(self, sender, groupidx):
        for grp in self.dh.clusters_info.groups_info:
            if grp['groupidx'] == groupidx:
                self.dh.clusters_info.groups_info.remove(grp)
                break
    
    def slotGroupChangedColor(self, sender, groupidx, coloridx):
        groups = self.cluster_widget.view.selected_groups()
        
        for grp in self.dh.clusters_info.groups_info:
            if grp['groupidx'] == groupidx:
                grp['coloridx'] = coloridx
                break
        
        self.cluster_widget.update_view(self.sdh)
        
        
    # Cluster slots
    #--------------
    def slotClusterChangedGroup(self, sender, clusteridx, groupidx):
        # self.dh.clusters_info.groups
        cluster_rel = self.dh.clusters_info.cluster_indices[clusteridx]
        # print cluster_rel
        self.dh.clusters_info.groups[cluster_rel] = groupidx
        
    def slotClusterChangedColor(self, sender, clusteridx, coloridx):
        clusters = self.cluster_widget.view.selected_clusters()
        
        cluster_rel = self.dh.clusters_info.cluster_indices[clusteridx]
        self.dh.clusters_info.colors[cluster_rel] = coloridx
        
        self.cluster_widget.update_view(self.sdh)
        # self.feature_widget.update_view(self.sdh)
        # self.waveform_widget.update_view(self.sdh)
        # self.correlograms_widget.update_view(self.sdh)
        
        # update the views by selecting the clusters again
        self.cluster_widget.view.select_multiple(clusters)
        
        
    # User preferences related methods
    # --------------------------------
    def save_geometry(self):
        """Save the arrangement of all widgets into a INI file."""
        # save main window geometry
        self.save_mainwindow_geometry()
        # save geometry of all widgets
        for widget in self.allwidgets:
            if hasattr(widget, 'save_geometry'):
                widget.save_geometry()
        
    def restore_geometry(self):
        """Restore the arrangement of all widgets from a INI file."""
        # restore main window geometry
        self.restore_mainwindow_geometry()
        # save geometry of all widgets
        for widget in self.allwidgets:
            if hasattr(widget, 'restore_geometry'):
                widget.restore_geometry()
        
    def save_mainwindow_geometry(self):
        """Save the arrangement of the whole window into a INI file."""
        SETTINGS.set("mainWindow/geometry", self.saveGeometry())
        SETTINGS.set("mainWindow/windowState", self.saveState())
        # save size and pos
        SETTINGS.set("mainWindow/size", self.size())
        SETTINGS.set("mainWindow/pos", self.pos())
        
    def restore_mainwindow_geometry(self):
        """Restore the arrangement of the whole window from a INI file."""
        g = SETTINGS.get("mainWindow/geometry")
        w = SETTINGS.get("mainWindow/windowState")
        if g:
            self.restoreGeometry(g)
        if w:
            self.restoreState(w)
            
        
    # Cleaning up
    # -----------
    def closeEvent(self, e):
        """Automatically save the arrangement of the window when closing
        the window."""
        # reset all signals so that they are not bound several times to 
        # the same slots in an interactive session
        ssignals.SIGNALS.reset()
        self.save_geometry()
        # save the settings
        SETTINGS.save()
        # TODO: clean up all widgets
        super(SpikyMainWindow, self).closeEvent(e)


if __name__ == '__main__':
    # NOTE: not putting "window=" results in weird QT thread warnings upon
    # closing (??)
    window = show_window(SpikyMainWindow)


