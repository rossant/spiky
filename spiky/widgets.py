from galry import *
from views import *
import tools
import numpy as np
import numpy.random as rnd
from dataio import MockDataProvider
from tools import Info
from collections import OrderedDict
import re

SETTINGS = tools.init_settings()

__all__ = ['WaveformWidget',
           'FeatureWidget',
           'CorrelogramsWidget',
           'CorrelationMatrixWidget',
           'ClusterWidget',
           ]




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
        
        # horizontal layout for the controller
        hbox = QtGui.QHBoxLayout()
        
        # we add the "isolated" checkbox
        # self.isolated_control = QtGui.QCheckBox("isolated")
        # hbox.addWidget(self.isolated_control, stretch=1, alignment=QtCore.Qt.AlignLeft)
        
        # # add the reset view button
        # self.reset_view_control = QtGui.QPushButton("reset view")
        # hbox.addWidget(self.reset_view_control, stretch=1, alignment=QtCore.Qt.AlignLeft)
        
        # # hbox.addWidget(QtGui.QCheckBox("test"), stretch=1, alignment=QtCore.Qt.AlignLeft)
        # # add lots of space to the right to make sure everything is aligned to 
        # # the left
        # hbox.addStretch(100)
        
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
        self.dh = dh
        view = FeatureView()
        view.set_data(dh.features, clusters=dh.clusters,
                      fetdim=3,
                      cluster_colors=dh.clusters_info.colors,
                      masks=dh.masks)
        return view

    def create_navigation_toolbar(self):
        toolbar = QtGui.QToolBar(self)
        toolbar.setObjectName("toolbar")
        toolbar.setIconSize(QtCore.QSize(32, 32))
        toolbar.addAction(get_icon('hand'), "Move (press M to switch)",
            self.set_navigation)
        toolbar.addAction(get_icon('selection'), "Selection (press M to switch)",
            self.set_selection)
        return toolbar
        
    def select_feature(self, coord, fet=0):
        """Select channel coord, feature fet."""
        print fet
        
    def _select_feature_getter(self, coord, fet):
        """Return the callback function for the feature selection."""
        return lambda e: self.select_feature(coord, fet)
        
    def channel_combo_changed(self, text):
        """Called when the combobox text has changed."""
        text = text.lower()
        # select time dimension
        if text == "time":
            print "time"
        # find if there is a number in the text, if so, it is the channel
        # dimension
        g = re.match("[^0-9]*([0-9]+)[^0-9]*", text)
        if g:
            channel = np.clip(int(g.groups()[0]), 0, self.dh.nchannels - 1)
            self.channel_selection_combobox.setCurrentIndex(channel)
            print "channel", channel
            # TODO: use qt signals instead
            # self.view.interaction_manager.select_projection((channel, channel, 0, 1))
            self.view.process_interaction(FeatureEventEnum.SelectProjectionEvent, (channel, channel, 0, 1))
        
    def create_feature_widget(self, coord=0):
        gridLayout = QtGui.QGridLayout()
        gridLayout.setSpacing(0)
        gridLayout.setMargin(0)
        
        # channel selection
        comboBox = QtGui.QComboBox(self)
        comboBox.setEditable(True)
        comboBox.setInsertPolicy(QtGui.QComboBox.NoInsert)
        comboBox.addItems(["Channel %d" % i for i in xrange(self.dh.nchannels)])
        comboBox.editTextChanged.connect(self.channel_combo_changed)
        # editTextChanged
        # currentIndexChanged
        # activated
        gridLayout.addWidget(comboBox, 0, 0, 1, 3)
        self.channel_selection_combobox = comboBox
        
        # create 3 buttons for selecting the feature
        widths = [60, 30, 30]
        labels = ['A', 'B', 'C']
        for i in xrange(3):
            # selecting feature i
            pushButton = QtGui.QPushButton(labels[i], self)
            pushButton.setMaximumSize(QtCore.QSize(widths[i], 20))
            pushButton.clicked.connect(self._select_feature_getter(coord, i))
            gridLayout.addWidget(pushButton, 1, i)
        
        return gridLayout
        
    def create_controller(self):
        box = super(FeatureWidget, self).create_controller()
        
        # add navigation toolbar
        self.toolbar = self.create_navigation_toolbar()
        box.addWidget(self.toolbar)
        
        # add feature widget
        self.feature_widget = self.create_feature_widget(0)
        box.addLayout(self.feature_widget)
        
        return box
    
    def set_navigation(self):
        self.view.set_interaction_mode(FeatureNavigationBindings)
    
    def set_selection(self):
        self.view.set_interaction_mode(FeatureSelectionBindings)
    
    
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


class ClusterWidget(QtGui.QWidget):
    
    class ClusterDelegate(QtGui.QStyledItemDelegate):
        def paint(self, painter, option, index):
            """Disable the color column so that the color remains the same even
            when it is selected."""
            # deactivate all columns except the first one, so that selection
            # is only possible in the first column
            if index.column() >= 1:
                if option.state and QtGui.QStyle.State_Selected:
                    option.state = option.state and QtGui.QStyle.State_Off
            super(ClusterWidget.ClusterDelegate, self).paint(painter, option, index)
    
    def __init__(self, dh):
        super(ClusterWidget, self).__init__()
        # put the controller and the view vertically
        vbox = QtGui.QVBoxLayout()
        
        # add context menu
        self.add_menu()
        
        # add controller
        self.controller = QtGui.QPushButton()
        vbox.addWidget(self.controller, stretch=1)
        
        # add the tree view
        self.view = self.create_tree_view(dh)
        vbox.addWidget(self.view, stretch=100)
        
        # set the VBox as layout of the widget
        self.setLayout(vbox)
        
    def add_menu(self):
        self.context_menu = QtGui.QMenu(self)
        self.context_menu.addAction("Add group", self.add_group_action)
        self.context_menu.addAction("Remove group", self.remove_group_action)
        
    def create_tree_view(self, dh):
        """Create the Tree View widget, and populates it using the data 
        handler `dh`."""
        # pass the cluster data to the ClusterView
        self.model = ClusterGroupManager(clusters=dh.clusters,
                                    clusters_info=dh.clusters_info)
        
        # set the QTreeView options
        view = QtGui.QTreeView()
        view.setModel(self.model)
        view.header().resizeSection(2, 20)
        view.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        view.expandAll()
        view.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        view.setAllColumnsShowFocus(True)
        view.setFirstColumnSpanned(0, QtCore.QModelIndex(), True)
        # view.setRootIsDecorated(False)
        view.setItemDelegate(self.ClusterDelegate())
        # self.setStyleSheet(STYLESHEET)
        
        return view

    def contextMenuEvent(self, event):
        action = self.context_menu.exec_(self.mapToGlobal(event.pos()))
        # TODO

    def selected_clusters(self):
        """Return the list of selected clusters."""
        return [(v.internalPointer().clusteridx()) \
                    for v in self.view.selectedIndexes() \
                        if v.column() == 0 and \
                           type(v.internalPointer()) == ClusterItem]
              
    def selected_groups(self):
        """Return the list of selected groups."""
        return [(v.internalPointer().groupidx()) \
                    for v in self.view.selectedIndexes() \
                        if v.column() == 0 and \
                           type(v.internalPointer()) == GroupItem]
                                            
    def add_group_action(self):
        groupindices = [g.groupidx() for g in self.model.get_groups()]
        groupidx = max(groupindices) + 1
        self.model.add_group(groupidx, "Group %d" % groupidx)
        self.view.expandAll()
    
    def remove_group_action(self):
        errors = []
        for groupidx in self.selected_groups():
            try:
                self.model.remove_group(groupidx)
            except:
                errors.append(groupidx)
        if errors:
            msg = "Non-empty groups were not deleted."
            self.mainwindow.statusBar().showMessage(msg, 5000)
    