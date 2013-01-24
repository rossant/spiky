import os
import re
from galry import *
import tools
from collections import OrderedDict
import numpy as np
import inspect
import spiky.signals as ssignals
import spiky
from spiky.qtqueue import qtjobqueue
import spiky.views as sviews
import spiky.dataio as sdataio
import rcicons


SETTINGS = tools.get_settings()

STYLESHEET = """
/*QMainWindow, QMenuBar, QToolBar, QPushButton {
    background-color: #000000;
    color: #ffffff;
}

QDockWidget::title, QDockWidget::float-button, QDockWidget::close-button {
    background-color: #999999;
    color: #ffffff;
}*/








/****************************************************/
/* http://tech-artists.org/forum/showthread.php?2359-Release-Qt-dark-orange-stylesheet */
QToolTip
{
     border: 1px solid black;
     background-color: #ffa02f;
     /*padding: 2px;*/
     /*border-radius: 3px;*/
     opacity: 100;
}

QWidget
{
    color: #b1b1b1;
    background-color: #323232;
}

QWidget:item:hover
{
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #ca0619);
    color: #000000;
}

QWidget:item:selected
{
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);
}


/* MENUS */
/***********************************************/
QMenuBar::item
{
    background: transparent;
}

QMenuBar::item:selected
{
    background: transparent;
    border: 1px solid #ffaa00;
}

QMenuBar::item:pressed
{
    background: #444;
    border: 1px solid #000;
    background-color: QLinearGradient(
        x1:0, y1:0,
        x2:0, y2:1,
        stop:1 #212121,
        stop:0.4 #343434/*,
        stop:0.2 #343434,
        stop:0.1 #ffaa00*/
    );
    margin-bottom:-1px;
    padding-bottom:1px;
}

QMenu
{
    border: 1px solid #000;
}

QMenu::item
{
    padding: 2px 20px 2px 20px;
}

QMenu::item:selected
{
    color: #000000;
}

QWidget:disabled
{
    /*color: #404040;*/
    background-color: #606060;
    /*background-color: #404040;*/
}

QAbstractItemView
{
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #4d4d4d, stop: 0.1 #646464, stop: 1 #5d5d5d);
}

QWidget:focus
{
    /*border: 2px solid QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);*/
}

QLineEdit
{
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #4d4d4d, stop: 0 #646464, stop: 1 #5d5d5d);
    padding: 1px;
    border-style: solid;
    border: 1px solid #1e1e1e;
    border-radius: 5;
}


/* CONTROLS */
/***********************************************/
QPushButton
{
    color: #b1b1b1;
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #565656, stop: 0.1 #525252, stop: 0.5 #4e4e4e, stop: 0.9 #4a4a4a, stop: 1 #464646);
    border-width: 1px;
    border-color: #1e1e1e;
    border-style: solid;
    border-radius: 6;
    padding: 3px;
    font-size: 12px;
    padding-left: 5px;
    padding-right: 5px;
}

QPushButton:pressed
{
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2d2d2d, stop: 0.1 #2b2b2b, stop: 0.5 #292929, stop: 0.9 #282828, stop: 1 #252525);
}

QComboBox
{
    selection-background-color: #ffaa00;
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #565656, stop: 0.1 #525252, stop: 0.5 #4e4e4e, stop: 0.9 #4a4a4a, stop: 1 #464646);
    border-style: solid;
    border: 1px solid #1e1e1e;
    border-radius: 5;
}

QComboBox:hover,QPushButton:hover
{
    border: 2px solid QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);
}


QComboBox:on
{
    padding-top: 3px;
    padding-left: 4px;
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2d2d2d, stop: 0.1 #2b2b2b, stop: 0.5 #292929, stop: 0.9 #282828, stop: 1 #252525);
    selection-background-color: #ffaa00;
}

QComboBox QAbstractItemView
{
    border: 2px solid darkgray;
    selection-background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);
}

QComboBox::drop-down
{
     subcontrol-origin: padding;
     subcontrol-position: top right;
     width: 15px;

     border-left-width: 0px;
     border-left-color: darkgray;
     border-left-style: solid; /* just a single line */
     border-top-right-radius: 3px; /* same radius as the QComboBox */
     border-bottom-right-radius: 3px;
 }

QComboBox::down-arrow
{
     image: url(:/icons/down_arrow.png);
}

QGroupBox:focus
{
border: 2px solid QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);
}

QTextEdit:focus
{
    border: 2px solid QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);
}



/* SCROLL BAR */
/***********************************************/

QScrollBar:horizontal {
     border: 1px solid #222222;
     background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0.0 #121212, stop: 0.2 #282828, stop: 1 #484848);
     height: 20px;
     margin: 0px 16px 0 16px;
}

QScrollBar::handle:horizontal
{
      background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #ffa02f, stop: 0.5 #d7801a, stop: 1 #ffa02f);
      min-height: 20px;
      border-radius: 2px;
}

QScrollBar::add-line:horizontal {
      border: 1px solid #1b1b19;
      border-radius: 2px;
      background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #ffa02f, stop: 1 #d7801a);
      width: 14px;
      subcontrol-position: right;
      subcontrol-origin: margin;
}

QScrollBar::sub-line:horizontal {
      border: 1px solid #1b1b19;
      border-radius: 2px;
      background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #ffa02f, stop: 1 #d7801a);
      width: 14px;
     subcontrol-position: left;
     subcontrol-origin: margin;
}


/*
QScrollBar::right-arrow:horizontal, QScrollBar::left-arrow:horizontal
{
      border: 1px solid black;
      width: 10px;
      height: 10px;
      background: white;
}*/

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal
{
      background: none;
}

QScrollBar:vertical
{
      background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0, stop: 0.0 #121212, stop: 0.2 #282828, stop: 1 #484848);
      width: 20px;
      margin: 16px 0 16px 0;
      border: 1px solid #222222;
}

QScrollBar::handle:vertical
{
      background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 0.5 #d7801a, stop: 1 #ffa02f);
      min-height: 20px;
      border-radius: 2px;
}

QScrollBar::add-line:vertical
{
      border: 1px solid #1b1b19;
      border-radius: 2px;
      background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);
      height: 14px;
      subcontrol-position: bottom;
      subcontrol-origin: margin;
}

QScrollBar::sub-line:vertical
{
      border: 1px solid #1b1b19;
      border-radius: 2px;
      background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #d7801a, stop: 1 #ffa02f);
      height: 14px;
      subcontrol-position: top;
      subcontrol-origin: margin;
}


QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical
{
      image: url(':/icons/up.png');
}
QScrollBar::down-arrow:vertical, QScrollBar::down-arrow:vertical
{
      image: url(':/icons/down.png');
}


QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical
{
      background: none;
}





/* TEXT */
/***********************************************/
QTextEdit
{
    background-color: #242424;
}

QPlainTextEdit
{
    background-color: #242424;
}

QHeaderView::section
{
    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #616161, stop: 0.5 #505050, stop: 0.6 #434343, stop:1 #656565);
    color: white;
    padding-left: 4px;
    border: 1px solid #6c6c6c;
}

QCheckBox:disabled
{
color: #414141;
}






/* DOCK WIDGET */
/***********************************************/
QDockWidget::title
{
    text-align: center;
    spacing: 3px; /* spacing between items in the tool bar */
    border-color: #343434;
   /*background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #323232, stop: 0.5 #242424, stop:1 #323232);*/
}

QDockWidget::close-button, QDockWidget::float-button
{
    text-align: center;
    spacing: 1px; /* spacing between items in the tool bar */
    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #323232, stop: 0.5 #242424, stop:1 #323232);
}

QDockWidget::close-button:hover, QDockWidget::float-button:hover
{
    background: #242424;
}

QDockWidget::close-button:pressed, QDockWidget::float-button:pressed
{
    padding: 1px -1px -1px 1px;
}

QMainWindow::separator
{
    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #161616, stop: 0.5 #151515, stop: 0.6 #212121, stop:1 #343434);
    color: white;
    padding-left: 4px;
    border: 1px solid #4c4c4c;
    spacing: 3px; /* spacing between items in the tool bar */
}

QMainWindow::separator:hover
{

    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #d7801a, stop:0.5 #b56c17 stop:1 #ffa02f);
    color: white;
    padding-left: 4px;
    border: 1px solid #6c6c6c;
    spacing: 3px; /* spacing between items in the tool bar */
}

QToolBar
{
    border-color: #323232;
}

QToolBar::handle
{
     spacing: 3px; /* spacing between items in the tool bar */
     background: url(':/icons/handle.png');
}

QToolButton:hover
{
    background-color: #ffaa00;
}

QMenu::separator
{
    height: 2px;
    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #161616, stop: 0.5 #151515, stop: 0.6 #212121, stop:1 #343434);
    color: white;
    padding-left: 4px;
    margin-left: 10px;
    margin-right: 5px;
}

QProgressBar
{
    border: 2px solid grey;
    border-radius: 5px;
    text-align: center;
}

QProgressBar::chunk
{
    background-color: #d7801a;
    width: 2.15px;
    margin: 0.5px;
}

QTabBar::tab {
    color: #b1b1b1;
    border: 1px solid #444;
    border-bottom-style: none;
    background-color: #323232;
    padding-left: 10px;
    padding-right: 10px;
    padding-top: 3px;
    padding-bottom: 2px;
    margin-right: -1px;
}

QTabWidget::pane {
    border: 1px solid #444;
    top: 1px;
}

QTabBar::tab:last
{
    margin-right: 0; /* the last selected tab has nothing to overlap with on the right */
    border-top-right-radius: 3px;
}

QTabBar::tab:first:!selected
{
 margin-left: 0px; /* the last selected tab has nothing to overlap with on the right */


    border-top-left-radius: 3px;
}

QTabBar::tab:!selected
{
    color: #b1b1b1;
    border-bottom-style: solid;
    margin-top: 3px;
    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:1 #212121, stop:.4 #343434);
}

QTabBar::tab:selected
{
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
    margin-bottom: 0px;
}

QTabBar::tab:!selected:hover
{
    /*border-top: 2px solid #ffaa00;
    padding-bottom: 3px;*/
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:1 #212121, stop:0.4 #343434, stop:0.2 #343434, stop:0.1 #ffaa00);
}

QRadioButton::indicator:checked, QRadioButton::indicator:unchecked{
    color: #b1b1b1;
    background-color: #323232;
    border: 1px solid #b1b1b1;
    border-radius: 6px;
}

QRadioButton::indicator:checked
{
    background-color: qradialgradient(
        cx: 0.5, cy: 0.5,
        fx: 0.5, fy: 0.5,
        radius: 1.0,
        stop: 0.25 #ffaa00,
        stop: 0.3 #323232
    );
}

QCheckBox::indicator{
    color: #b1b1b1;
    background-color: #323232;
    border: 1px solid #b1b1b1;
    width: 9px;
    height: 9px;
}

QRadioButton::indicator
{
    border-radius: 6px;
}

QRadioButton::indicator:hover, QCheckBox::indicator:hover
{
    border: 1px solid #ffaa00;
}

QCheckBox::indicator:checked
{
    image:url(:/icons/checkbox.png);
}

QCheckBox::indicator:disabled, QRadioButton::indicator:disabled
{
    border: 1px solid #444;
}
/****************************************************/
















QStatusBar::item
{
    border: none;
}
"""

__all__ = ['SpikyMainWindow', 'show_window']


@qtjobqueue
class ClusterSelectionQueue(object):
    def __init__(self, du, dh):
        self.du = du
        self.dh = dh
        
    def select(self, clusters):
        self.dh.select_clusters(clusters)
        ssignals.emit(self.du, 'ClusterSelectionChanged', clusters)

    
@qtjobqueue    
class KlustersLoadQueue(object):
    def __init__(self, progressbar=None):
        self.progressbar = progressbar
    
    def load(self, filename, fileindex, probefile):
        
        # if hasattr(self, 'du'):
            # self.du.stop()
        
        self.provider = sdataio.KlustersDataProvider()
        self.dh = self.provider.load(filename, fileindex=fileindex,
            probefile=probefile, progressbar=self.progressbar)
        # self.sdh = sdataio.SelectDataHolder(self.dh)
        # self.du = DataUpdater(self.sdh)
        # self.am = spiky.ActionManager(self.dh, self.sdh)
        ssignals.emit(self, 'FileLoaded')
        
        
        

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
        self.queue = ClusterSelectionQueue(self, dh)
        self.initialize_connections()
        
    def initialize_connections(self):
        ssignals.SIGNALS.ProjectionToChange.connect(self.slotProjectionToChange, QtCore.Qt.UniqueConnection)
        ssignals.SIGNALS.ClusterSelectionToChange.connect(self.slotClusterSelectionToChange, QtCore.Qt.UniqueConnection)
        
    def slotClusterSelectionToChange(self, sender, clusters):
        self.queue.select(clusters)
        
    def slotProjectionToChange(self, sender, coord, channel, feature):
        ssignals.emit(sender, 'ProjectionChanged', coord, channel, feature)
        
    def stop(self):
        """Stop the cluster selection job queue."""
        self.queue.join()
        
        
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
        self.am = spiky.ActionManager(self.dh, self.sdh)
        
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
        
        # exit action
        self.quit_action = QtGui.QAction("E&xit", self)
        self.quit_action.setShortcut("CTRL+Q")
        self.quit_action.triggered.connect(self.close, QtCore.Qt.UniqueConnection)
        
        # merge action
        self.merge_action = QtGui.QAction("Mer&ge", self)
        self.merge_action.setIcon(spiky.get_icon("merge"))
        self.merge_action.setShortcut("G")
        self.merge_action.setEnabled(False)
        self.merge_action.triggered.connect(self.merge, QtCore.Qt.UniqueConnection)
        
        # split action
        self.split_action = QtGui.QAction("&Split", self)
        self.split_action.setIcon(spiky.get_icon("split"))
        self.split_action.setShortcut("S")
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
        ssignals.SIGNALS.FileLoaded.connect(self.slotFileLoaded)
        
        
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
        folder = SETTINGS.get('mainWindow/last_data_dir')
        
        default_filename = self.filename
        default_filename += "_spiky.clu.%d" % self.fileindex
        
        filename = None
        if not hasattr(self, 'save_filename'):
            filename = QtGui.QFileDialog.getSaveFileName(self, "Save a CLU file",
                os.path.join(folder, default_filename))[0]
            if filename:
                self.save_filename = filename
        else:
            filename = self.save_filename
        if filename:
            self.provider.save(filename)
        
    def load_file(self, filename):
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
        
        self.loadqueue = KlustersLoadQueue(self.progressbar)
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

        # update undo and redo
        self.undo_action.setEnabled(self.am.undo_enabled())
        self.redo_action.setEnabled(self.am.redo_enabled())
        
        self.progressbar.setValue(5)
        
        self.loadqueue.join()
        
    
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


