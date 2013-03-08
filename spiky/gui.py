import os
import re
from galry import *
import tools
from collections import OrderedDict
import numpy as np
from colors import COLORMAP
from collections import Counter
from copy import deepcopy as dcopy
import numpy.random as rdn
import inspect
import spiky.signals as ssignals
import spiky
from qtools import inthread
import spiky.views as sviews
import spiky.dataio as sdataio
import rcicons

import spiky.tasks as tasks

SETTINGS = tools.get_settings()


__all__ = ['SpikyMainWindow', 'show_window']


        
class DataUpdater(QtGui.QWidget):
    """Handle data updating in the data holder, responding to signals
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
        self.probefile = None
        # self.queue = ClusterSelectionQueue(self, dh)
        tasks.TASKS.cluster_selection_queue = tasks.ClusterSelectionQueue(self, dh)
        self.initialize_connections()
        
    def initialize_connections(self):
        ssignals.SIGNALS.ProjectionToChange.connect(self.slotProjectionToChange, QtCore.Qt.UniqueConnection)
        ssignals.SIGNALS.ClusterSelectionToChange.connect(self.slotClusterSelectionToChange, QtCore.Qt.UniqueConnection)
        
    def slotClusterSelectionToChange(self, sender, clusters):
        tasks.TASKS.cluster_selection_queue.select(clusters)
        
    def slotProjectionToChange(self, sender, coord, channel, feature):
        ssignals.emit(sender, 'ProjectionChanged', coord, channel, feature)
        
    def stop(self):
        """Stop the cluster selection job queue."""
        try:
            ssignals.SIGNALS.ProjectionToChange.disconnect()
        except TypeError:
            pass
        try:
            ssignals.SIGNALS.ClusterSelectionToChange.disconnect()
        except TypeError:
            pass
        tasks.TASKS.cluster_selection_queue.join()
        # print "STOP"
        
        
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
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, "styles.css")
        with open(path, 'r') as f:
            STYLESHEET = f.read()
        STYLESHEET = STYLESHEET.replace('%ACCENT%', '#cdcdcd')
        STYLESHEET = STYLESHEET.replace('%ACCENT2%', '#a0a0a0')
        STYLESHEET = STYLESHEET.replace('%ACCENT3%', '#909090')
        STYLESHEET = STYLESHEET.replace('%ACCENT4%', '#cdcdcd')
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
        self.am = spiky.ActionManager(self.dh, self.sdh)
        self.robot = spiky.Robot(self.dh)
        
        # central window, the dockable widgets are arranged around it
        self.cluster_widget, self.cluster_dock_widget = self.add_dock(sviews.ClusterWidget, QtCore.Qt.RightDockWidgetArea)
        # dock widgets
        self.feature_widget = self.add_central(sviews.FeatureWidget)
        self.waveform_widget, self.waveform_dock_widget = self.add_dock(sviews.WaveformWidget, QtCore.Qt.RightDockWidgetArea)        
        self.correlograms_widget, self.correlograms_dock_widget = self.add_dock(sviews.CorrelogramsWidget, QtCore.Qt.RightDockWidgetArea)
        
        self.correlationmatrix_widget, self.correlationmatrix_dock_widget = self.add_dock(sviews.CorrelationMatrixWidget, QtCore.Qt.RightDockWidgetArea)
        
        # widget actions
        self.cluster_action = self.cluster_dock_widget.toggleViewAction()
        self.waveform_action = self.waveform_dock_widget.toggleViewAction()
        self.correlograms_action = self.correlograms_dock_widget.toggleViewAction()
        self.correlationmatrix_action = self.correlationmatrix_dock_widget.toggleViewAction()
        
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
        
        # open probe file action
        self.open_probe_action = QtGui.QAction("&Open probe file", self)
        self.open_probe_action.setShortcut("CTRL+SHIFT+O")
        self.open_probe_action.setIcon(spiky.get_icon("probe"))
        self.open_probe_action.triggered.connect(self.open_probefile, QtCore.Qt.UniqueConnection)
        
        # save action
        self.save_action = QtGui.QAction("&Save", self)
        self.save_action.setShortcut("CTRL+S")
        self.save_action.setIcon(spiky.get_icon("save"))
        self.save_action.triggered.connect(self.save_file, QtCore.Qt.UniqueConnection)
        
        # save action
        self.saveas_action = QtGui.QAction("Save &as", self)
        self.saveas_action.setShortcut("CTRL+SHIFT+S")
        self.saveas_action.setIcon(spiky.get_icon("saveas"))
        self.saveas_action.triggered.connect(self.saveas_file, QtCore.Qt.UniqueConnection)
        
        # exit action
        self.quit_action = QtGui.QAction("E&xit", self)
        self.quit_action.setShortcut("CTRL+Q")
        self.quit_action.triggered.connect(self.close, QtCore.Qt.UniqueConnection)
        
        # merge action
        self.merge_action = QtGui.QAction("Mer&ge", self)
        self.merge_action.setIcon(spiky.get_icon("merge"))
        self.merge_action.setShortcut("CTRL+G")
        self.merge_action.setEnabled(False)
        self.merge_action.triggered.connect(self.merge, QtCore.Qt.UniqueConnection)
        
        # split action
        self.split_action = QtGui.QAction("&Split", self)
        self.split_action.setIcon(spiky.get_icon("split"))
        self.split_action.setShortcut("CTRL+K")
        self.split_action.setEnabled(False)
        self.split_action.triggered.connect(self.split, QtCore.Qt.UniqueConnection)
        
        # DEL
        self.move_to_mua_action = QtGui.QAction("Move to &Multi-Unit", self)
        self.move_to_mua_action.setShortcut("Del")
        self.move_to_mua_action.setIcon(spiky.get_icon("multiunit"))
        self.move_to_mua_action.triggered.connect(self.move_to_mua, QtCore.Qt.UniqueConnection)
        self.move_to_mua_action.setEnabled(False)
        
        # SHIFT+DEL
        self.move_to_noise_action = QtGui.QAction("Move to &Noise", self)
        self.move_to_noise_action.setShortcut("Shift+Del")
        self.move_to_noise_action.setIcon(spiky.get_icon("noise"))
        self.move_to_noise_action.triggered.connect(self.move_to_noise, QtCore.Qt.UniqueConnection)
        self.move_to_noise_action.setEnabled(False)
        
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
        
        # override color action
        self.override_color_action = QtGui.QAction("Override &color", self)
        self.override_color_action.setShortcut("C")
        self.override_color_action.setIcon(spiky.get_icon("override_color"))
        self.override_color_action.triggered.connect(self.override_color, QtCore.Qt.UniqueConnection)
        
    def initialize_menu(self):
        """Initialize the menu."""
        # File menu
        # ---------
        file_menu = self.menuBar().addMenu("&File")
        
        file_menu.addAction(self.open_probe_action)
        file_menu.addSeparator()
        
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.saveas_action)
        file_menu.addSeparator()
        
        # open last probe
        self.open_last_probefile()
        
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
        views_menu.addAction(self.correlationmatrix_action)
        views_menu.addSeparator()
        views_menu.addAction(self.override_color_action)
        
        
        # Actions menu
        # ------------
        actions_menu = self.menuBar().addMenu("&Actions")
        actions_menu.addAction(self.undo_action)
        actions_menu.addAction(self.redo_action)
        actions_menu.addSeparator()
        actions_menu.addAction(self.merge_action)
        actions_menu.addAction(self.split_action)
        actions_menu.addSeparator()
        actions_menu.addAction(self.move_to_mua_action)
        actions_menu.addAction(self.move_to_noise_action)
        
    def initialize_toolbar(self):
        # self.toolbar = QtGui.QToolBar(self)
        self.toolbar = self.addToolBar("SpikyToolbar")
        self.toolbar.setObjectName("SpikyToolbar")
        self.toolbar.addAction(self.open_action)
        self.toolbar.addAction(self.save_action)
        self.toolbar.addAction(self.saveas_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.merge_action)
        self.toolbar.addAction(self.split_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.move_to_mua_action)
        self.toolbar.addAction(self.move_to_noise_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.undo_action)
        self.toolbar.addAction(self.redo_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.override_color_action)
    
    def initialize_connections(self):
        """Initialize the signals/slots connections between widgets."""
        # ssignals.SIGNALS.HighlightSpikes.connect(self.slotHighlightSpikes, QtCore.Qt.UniqueConnection)
        
        ssignals.SIGNALS.ClusterSelectionToChange.connect(self.slotClusterSelectionToChange, QtCore.Qt.UniqueConnection)
        
        ssignals.SIGNALS.ClusterSelectionChanged.connect(self.slotClusterSelectionChanged)
        ssignals.SIGNALS.SelectSpikes.connect(self.slotSelectSpikes)
        ssignals.SIGNALS.ClusterInfoToUpdate.connect(self.slotClusterInfoToUpdate)
        # signals emitted by child windows and request the main window to 
        # process an action
        ssignals.SIGNALS.RenameGroupRequested.connect(self.slotRenameGroupRequested)
        ssignals.SIGNALS.MoveClustersRequested.connect(self.slotMoveClustersRequested)
        ssignals.SIGNALS.AddGroupRequested.connect(self.slotAddGroupRequested)
        ssignals.SIGNALS.RemoveGroupsRequested.connect(self.slotRemoveGroupsRequested)
        ssignals.SIGNALS.ChangeGroupColorRequested.connect(self.slotChangeGroupColorRequested)
        ssignals.SIGNALS.ChangeClusterColorRequested.connect(self.slotChangeClusterColorRequested)
        ssignals.SIGNALS.CorrelogramsUpdated.connect(self.slotCorrelogramsUpdated)
        ssignals.SIGNALS.CorrelationMatrixUpdated.connect(self.slotCorrelationMatrixUpdated)
        ssignals.SIGNALS.FileLoaded.connect(self.slotFileLoaded)
        ssignals.SIGNALS.FileLoading.connect(self.slotFileLoading)
        
        
    # File methods
    # ------------
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
        self.reset_action_generator()
        # if a filename has not been set, save as
        if not hasattr(self, 'save_filename'):
            self.saveas_file()
        else:
            # save with the last used filename
            self.provider.save(self.save_filename)
            
    def saveas_file(self, *args):
        self.reset_action_generator()
        folder = SETTINGS.get('mainWindow/last_data_dir')
        
        # default filename
        if not hasattr(self, 'filename'):
            return
        default_filename = self.filename
        default_filename += "_spiky.clu.%d" % self.fileindex
        
        # ask a new file name
        filename = QtGui.QFileDialog.getSaveFileName(self, "Save a CLU file",
            os.path.join(folder, default_filename))[0]
        if filename:
            # save the new file name
            self.save_filename = filename
            # save
            self.provider.save(filename)
        
    def load_file(self, filename):
        # template: FILENAME.xxx.0  => FILENAME (can contain points), 0 (index)
        r = re.search(r"([^\n]+)\.[^\.]+\.([0-9]+)$", filename)
        if r:
            # save last opened file
            SETTINGS.set('mainWindow/last_data_file', filename)
            filename = r.group(1)
            fileindex = int(r.group(2))
            self.filename = filename
            self.fileindex = fileindex
        else:
            log_warn(("The file could not be loaded because it is not like",
                " *.i.*"))
            return
        
        # save folder
        folder = os.path.dirname(filename)
        SETTINGS.set('mainWindow/last_data_dir', folder)
        
        # # stop the cluster selection job queue when changing files
        if hasattr(self, 'du'):
            self.du.stop()
        # if hasattr(self, 'loadqueue'):
            # self.loadqueue.join()
            # # print "joined"
            
        self.progressbar = QtGui.QProgressDialog("Loading...", "Cancel", 0, 5, self)
        self.progressbar.setWindowModality(QtCore.Qt.WindowModal)
        self.progressbar.setValue(0)
        self.progressbar.setCancelButton(None)
        self.progressbar.setMinimumDuration(0)
        
        # self.loadqueue = KlustersLoadQueue()#self.progressbar)
        # self.loadqueue.load(filename, fileindex, self.probefile)
        self.loadqueue = sdataio.KlustersLoadQueue()#self.progressbar)
        self.loadqueue.load(filename, fileindex, self.probefile)
        
        
    def slotFileLoaded(self):
        # self.provider = sdataio.KlustersDataProvider()
        # self.dh = self.provider.load(self.filename, fileindex=self.fileindex,
            # probefile=self.probefile)
        # self.sdh = sdataio.SelectDataHolder(self.dh)
        # self.du = DataUpdater(self.sdh)
        # self.am = spiky.ActionManager(self.dh, self.sdh)
        
        # retrieve the provider and holder objects from the load queue
        self.provider = self.loadqueue.provider
        self.dh = self.loadqueue.dh
        # self.sdh = self.loadqueue.sdh
        # self.du = self.loadqueue.du
        # self.am = self.loadqueue.am
        self.sdh = sdataio.SelectDataHolder(self.dh)
        self.du = DataUpdater(self.sdh)
        self.am = spiky.ActionManager(self.dh, self.sdh)
        
        # update the views
        self.cluster_widget.update_view(self.sdh)
        self.feature_widget.update_view(self.sdh)
        self.waveform_widget.update_view(self.sdh)
        self.correlograms_widget.update_view(self.sdh)
        self.correlationmatrix_widget.update_view(self.sdh)

        # update undo and redo
        self.undo_action.setEnabled(self.am.undo_enabled())
        self.redo_action.setEnabled(self.am.redo_enabled())
        
        self.progressbar.setValue(5)
        
        self.loadqueue.join()
        
    def slotFileLoading(self, sender, value):
        self.progressbar.setValue(int(value * 5))
    
    
    def open_probefile(self):
        self.reset_action_generator()
        folder = SETTINGS.get('mainWindow/last_probe_dir')
        filename = QtGui.QFileDialog.getOpenFileName(self, "Open a probe file", folder)[0]
        if filename:
            self.probefile = filename
            # save probefile
            SETTINGS.set('mainWindow/last_probe_file', filename)
            # save probe folder
            folder = os.path.dirname(filename)
            SETTINGS.set('mainWindow/last_probe_dir', folder)
        
    def open_last_probefile(self, *args):
        self.probefile = SETTINGS.get('mainWindow/last_probe_file', None)
    
    
    # Generic Do/Redo methods
    # -----------------------
    def do(self, action_class, *args):
        action = self.am.do(action_class, *args)
        self.cluster_widget.update_view(self.sdh)
        clusters = action.selected_clusters_after_redo()
        if len(clusters) > 0:
            self.cluster_widget.view.select_multiple(clusters)
        self.undo_action.setEnabled(self.am.undo_enabled())
        self.redo_action.setEnabled(self.am.redo_enabled())
        
    def undo(self):
        action = self.am.undo()
        if action is not None:
            self.cluster_widget.update_view(self.sdh)
            clusters = action.selected_clusters_after_undo()
            if len(clusters) > 0:
                self.cluster_widget.view.select_multiple(clusters)
        self.undo_action.setEnabled(self.am.undo_enabled())
        self.redo_action.setEnabled(self.am.redo_enabled())
        
    def redo(self):
        action = self.am.redo()
        if action is not None:
            self.cluster_widget.update_view(self.sdh)
            clusters = action.selected_clusters_after_redo()
            if len(clusters) > 0:
                self.cluster_widget.view.select_multiple(clusters)
        self.undo_action.setEnabled(self.am.undo_enabled())
        self.redo_action.setEnabled(self.am.redo_enabled())
    
    
    # Do/Redo methods
    # ---------------
    def merge(self):
        """Merge selected clusters."""
        self.do(spiky.MergeAction, self.sdh.get_clusters())
        
    def split(self):
        """Split selected spikes."""
        self.do(spiky.SplitAction, self.selected_spikes)
        self.feature_widget.view.process_interaction('CancelSelectionPoint')
        
    def move_to_group(self, groupidx):
        clusters = self.sdh.get_clusters()
        self.do(spiky.MoveToGroupAction, clusters, groupidx)
        
    def move_to_mua(self):
        self.move_to_group(1)
        
    def move_to_noise(self):
        self.move_to_group(0)
        
    def override_color(self):
        self.sdh.override_color = not(self.sdh.override_color)
        self.feature_widget.update_view(self.sdh)
        self.waveform_widget.update_view(self.sdh)
        self.correlograms_widget.update_view(self.sdh)
        
    def slotRenameGroupRequested(self, sender, groupidx, name):
        self.do(spiky.RenameGroupAction, groupidx, name)
        
    def slotMoveClustersRequested(self, sender, clusters, groupidx):
        self.do(spiky.MoveToGroupAction, clusters, groupidx)
    
    def slotAddGroupRequested(self, sender):
        self.do(spiky.AddGroupAction)
        
    def slotRemoveGroupsRequested(self, sender, groups):
        self.do(spiky.RemoveGroupsAction, groups)
        
    def slotChangeGroupColorRequested(self, sender, groups, color):
        self.do(spiky.ChangeGroupColorAction, groups, color)
        
    def slotChangeClusterColorRequested(self, sender, clusters, color):
        self.do(spiky.ChangeClusterColorAction, clusters, color)
        
    def slotCorrelogramsUpdated(self, sender):
        # print self.sdh.
        self.correlograms_widget.update_view(self.sdh)
        
    def slotCorrelationMatrixUpdated(self, sender):
        # print self.sdh.
        self.correlationmatrix_widget.update_view(self.sdh)
        
        
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
            
            
    def slotClusterSelectionToChange(self, sender, clusters):
        pass
            
    def slotClusterSelectionChanged(self, sender, clusters):
        # print clusters
        # enable/disable del/shift+del when no clusters are selected
        if len(clusters) >= 1:
            self.move_to_mua_action.setEnabled(True)
            self.move_to_noise_action.setEnabled(True)
        else:
            self.move_to_mua_action.setEnabled(False)
            self.move_to_noise_action.setEnabled(False)
        
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
    def slotClusterInfoToUpdate(self, sender):
        # get selected clusters
        clusters = self.cluster_widget.view.selected_clusters()
        # update the data holder
        self.dh.clusters_info = self.cluster_widget.model.to_dict()
        self.cluster_widget.update_view(self.sdh)
        # re-select the selected clusters
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
        
        tasks.TASKS.join()
        
        if hasattr(self, 'du'):
            self.du.stop()
            
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


