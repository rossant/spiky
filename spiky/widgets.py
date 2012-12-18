from galry import *
from views import *
import tools
import numpy as np
import numpy.random as rnd
from dataio import MockDataProvider
from tools import Info
from collections import OrderedDict
import re


SETTINGS = tools.get_settings()


__all__ = ['VisualizationWidget',
           'WaveformWidget',
           'FeatureWidget',
           'CorrelogramsWidget',
           'CorrelationMatrixWidget',
           'ClusterWidget',
           ]


class VisualizationWidget(QtGui.QWidget):
    def __init__(self, main_window, dataholder):
        super(VisualizationWidget, self).__init__()
        self.dataholder = dataholder
        self.main_window = main_window
        self.view = self.create_view(dataholder)
        self.controller = self.create_controller()
        self.initialize()
        self.initialize_connections()

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

    def initialize_connections(self):
        """Initialize signals/slots connections."""
        pass


class WaveformWidget(VisualizationWidget):
    def create_view(self, dh):
        self.dh = dh
        self.view = WaveformView(getfocus=False)
        
        # load user preferences
        geometry_preferences = self.restore_geometry()
        if geometry_preferences is None:
            geometry_preferences = {}
        # print geometry_preferences
        self.view.set_data(self.dh.waveforms,
                      clusters=self.dh.clusters,
                      cluster_colors=self.dh.cluster_colors,
                      geometrical_positions=self.dh.probe.positions,
                      masks=self.dh.masks,
                      **geometry_preferences
                      # user preferences
                      # TODO: store this in the data file, or in the user
                      # preferences
                      # spatial_arrangement=geometry_preferences.get('spatial_arrangement', None),
                      # superposition=geometry_preferencesget('superposition', None),
                      # box_size=geometry_preferencesget('box_size', None),
                      # probe_scale=geometry_preferencesget('probe_scale', None)
                      )
        return self.view
        
    def update_view(self):
        self.view.set_data(self.dh.waveforms,
                      clusters=self.dh.clusters,
                      clusters_unique=self.dh.clusters_unique,
                      cluster_colors=self.dh.cluster_colors,
                      geometrical_positions=self.dh.probe.positions,
                      masks=self.dh.masks
                      )
    
    def initialize_connections(self):
        SIGNALS.ProjectionChanged.connect(self.slotProjectionChanged, QtCore.Qt.UniqueConnection)
        SIGNALS.ClusterSelectionChanged.connect(self.slotClusterSelectionChanged, QtCore.Qt.UniqueConnection)
        
    def slotClusterSelectionChanged(self, sender, clusters):
        self.update_view()
        
    def slotProjectionChanged(self, sender, coord, channel, feature):
        pass
        
        
    # Save and restore geometry
    # -------------------------
    def save_geometry(self):
        geometry_preferences = {
            'spatial_arrangement': self.view.position_manager.spatial_arrangement,
            'superposition': self.view.position_manager.superposition,
            'box_size': self.view.position_manager.load_box_size(),
            'probe_scale': self.view.position_manager.probe_scale,
        }
        SETTINGS.set("waveformWidget/geometry", geometry_preferences)
        
    def restore_geometry(self):
        """Return a dictionary with the user preferences regarding geometry
        in the WaveformView."""
        return SETTINGS.get("waveformWidget/geometry")
        
        
class FeatureWidget(VisualizationWidget):
    def create_view(self, dh):
        self.dh = dh
        self.view = FeatureView(getfocus=False)
        self.view.set_data(fetdim=self.dh.fetdim,
                      features=self.dh.features,
                      clusters=self.dh.clusters,
                      # colormap=self.dh.colormap,
                      cluster_colors=self.dh.cluster_colors,
                      masks=self.dh.masks)
        return self.view
        
    def update_view(self):
        self.view.set_data(fetdim=self.dh.fetdim,
                      features=self.dh.features,
                      clusters=self.dh.clusters,
                      cluster_colors=self.dh.cluster_colors,
                      clusters_unique=self.dh.clusters_unique,
                      masks=self.dh.masks)
        self.update_nspikes_viewer(self.dh.nspikes, 0)

    def create_toolbar(self):
        toolbar = QtGui.QToolBar(self)
        toolbar.setObjectName("toolbar")
        toolbar.setIconSize(QtCore.QSize(32, 32))
        
        # navigation toolbar
        toolbar.addAction(get_icon('hand'), "Move (press I to switch)",
            self.set_navigation)
        toolbar.addAction(get_icon('selection'), "Selection (press I to switch)",
            self.set_selection)
            
        toolbar.addSeparator()
            
        # autoprojection
        toolbar.addAction(self.main_window.autoproj_action)
        
        toolbar.addSeparator()
        
        return toolbar
        
    def initialize_connections(self):
        SIGNALS.ProjectionChanged.connect(self.slotProjectionChanged, QtCore.Qt.UniqueConnection)
        SIGNALS.ClusterSelectionChanged.connect(self.slotClusterSelectionChanged, QtCore.Qt.UniqueConnection)
        SIGNALS.HighlightSpikes.connect(self.slotHighlightSpikes, QtCore.Qt.UniqueConnection)
        
    def slotHighlightSpikes(self, parent, highlighted):
        self.update_nspikes_viewer(self.dh.nspikes, len(highlighted))
        
    def slotClusterSelectionChanged(self, sender, clusters):
        self.update_view()
        
    def slotProjectionChanged(self, sender, coord, channel, feature):
        """Process the ProjectionChanged signal."""
        
        # feature == -1 means that it should be automatically selected as
        # a function of the current projection
        if feature < 0:
            # current channel and feature in the other coordinate
            other_channel, other_feature = self.view.data_manager.projection[1 - coord]
            fetdim = self.dh.fetdim
            # first dimension: we force to 0
            if coord == 0:
                feature = 0
            # other dimension: 0 if different channel, or next feature if the same
            # channel
            else:
                # same channel case
                if channel == other_channel:
                    feature = np.mod(other_feature + 1, fetdim)
                # different channel case
                else:
                    feature = 0
        
        # print sender
        log_info("Projection changed in coord %s, channel=%d, feature=%s" \
            % (('X', 'Y')[coord], channel, ('A', 'B', 'C')[feature]))
        # record the new projection
        self.projection[coord] = (channel, feature)
        
        # prevent the channelbox to raise signals when we change its state
        # programmatically
        self.channel_box[coord].blockSignals(True)
        # update the channel box
        self.set_channel_box(coord, channel)
        # update the feature button
        self.set_feature_button(coord, feature)
        # reactive signals for the channel box
        self.channel_box[coord].blockSignals(False)
        
        # update the view
        self.view.process_interaction('SelectProjection', 
                                      (coord, channel, feature))
        
    def set_channel_box(self, coord, channel):
        """Select the adequate line in the channel selection combo box."""
        self.channel_box[coord].setCurrentIndex(channel)
        
    def set_feature_button(self, coord, feature):
        """Push the corresponding button."""
        self.feature_buttons[coord][feature].setChecked(True)
        
    def select_feature(self, coord, fet=0):
        """Select channel coord, feature fet."""
        # raise the ProjectionToChange signal, and keep the previously
        # selected channel
        emit(self, "ProjectionToChange", coord, self.projection[coord][0], fet)
        
    def select_channel(self, channel, coord=0):
        """Raise the ProjectionToChange signal when the channel is changed."""
        
        try:
            channel = int(channel)
            emit(self, "ProjectionToChange", coord, channel,
                     self.projection[coord][1])
        except:
            log_warn("'%s' is not a valid channel." % channel)
        
    def _select_feature_getter(self, coord, fet):
        """Return the callback function for the feature selection."""
        return lambda *args: self.select_feature(coord, fet)
        
    def _select_channel_getter(self, coord):
        """Return the callback function for the channel selection."""
        return lambda channel: self.select_channel(channel, coord)
        
    def create_feature_widget(self, coord=0):
        # coord => (channel, feature)
        self.projection = [(0, 0), (0, 1)]
        
        gridLayout = QtGui.QGridLayout()
        gridLayout.setSpacing(0)
        # HACK: pyside does not have this function
        if hasattr(gridLayout, 'setMargin'):
            gridLayout.setMargin(0)
        
        # channel selection
        comboBox = QtGui.QComboBox(self)
        comboBox.setEditable(True)
        comboBox.setInsertPolicy(QtGui.QComboBox.NoInsert)
        comboBox.addItems(["%d" % i for i in xrange(self.dh.nchannels)])
        comboBox.editTextChanged.connect(self._select_channel_getter(coord), QtCore.Qt.UniqueConnection)
        comboBox.currentIndexChanged.connect(self._select_channel_getter(coord), QtCore.Qt.UniqueConnection)
        comboBox.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.channel_box[coord] = comboBox
        gridLayout.addWidget(comboBox, 0, 0, 1, 3)
        
        # TODO: use dh.fetdim instead of hard coded "3 features"
        # create 3 buttons for selecting the feature
        widths = [30] * self.dh.fetdim
        labels = ['PC%d' % i for i in xrange(1, self.dh.fetdim + 1)]
        
        # ensure exclusivity of the group of buttons
        pushButtonGroup = QtGui.QButtonGroup(self)
        for i in xrange(len(labels)):
            # selecting feature i
            pushButton = QtGui.QPushButton(labels[i], self)
            pushButton.setCheckable(True)
            if coord == i:
                pushButton.setChecked(True)
            pushButton.setMaximumSize(QtCore.QSize(widths[i], 20))
            pushButton.clicked.connect(self._select_feature_getter(coord, i), QtCore.Qt.UniqueConnection)
            pushButtonGroup.addButton(pushButton, i)
            self.feature_buttons[coord][i] = pushButton
            gridLayout.addWidget(pushButton, 1, i)
        
        return gridLayout
        
    def create_nspikes_viewer(self):
        self.nspikes_viewer = QtGui.QLabel("", self)
        return self.nspikes_viewer
        
    def get_nspikes_text(self, nspikes, nspikes_highlighted):
        return "Spikes: %d. Highlighted: %d." % (nspikes, nspikes_highlighted)
        
    def update_nspikes_viewer(self, nspikes, nspikes_highlighted):
        text = self.get_nspikes_text(nspikes, nspikes_highlighted)
        self.nspikes_viewer.setText(text)
        
    def create_controller(self):
        box = super(FeatureWidget, self).create_controller()
        
        # coord => channel combo box
        self.channel_box = [None, None]
        # coord => (butA, butB, butC)
        self.feature_buttons = [[None] * 3, [None] * 3]
        
        # add navigation toolbar
        self.toolbar = self.create_toolbar()
        box.addWidget(self.toolbar)

        # add number of spikes
        self.nspikes_viewer = self.create_nspikes_viewer()
        box.addWidget(self.nspikes_viewer)
        
        # add feature widget
        self.feature_widget1 = self.create_feature_widget(0)
        box.addLayout(self.feature_widget1)
        
        # add feature widget
        self.feature_widget2 = self.create_feature_widget(1)
        box.addLayout(self.feature_widget2)
        
        return box
    
    def set_navigation(self):
        self.view.set_interaction_mode(FeatureNavigationBindings)
    
    def set_selection(self):
        self.view.set_interaction_mode(FeatureSelectionBindings)
    
    
class CorrelogramsWidget(VisualizationWidget):
    def create_view(self, dh):
        self.dh = dh
        view = CorrelogramsView(getfocus=False)
        view.set_data(histograms=dh.correlograms,
                      baselines=dh.baselines,
                      cluster_colors=dh.cluster_colors)
        return view
        
    def update_view(self):
        self.view.set_data(histograms=self.dh.correlograms,
                      baselines=self.dh.baselines,
                      cluster_colors=self.dh.cluster_colors)

    def initialize_connections(self):
        SIGNALS.ClusterSelectionChanged.connect(self.slotClusterSelectionChanged, QtCore.Qt.UniqueConnection)
    
    def slotClusterSelectionChanged(self, sender, clusters):
        self.update_view()
        
        
class CorrelationMatrixWidget(VisualizationWidget):
    def create_view(self, dh):
        view = CorrelationMatrixView()
        view.set_data(dh.correlationmatrix)
        return view


class ClusterWidget(QtGui.QWidget):
    def __init__(self, main_window, dh, getfocus=True):
        super(ClusterWidget, self).__init__()
        
        # Capture keyboard events.
        if getfocus:
            self.setFocusPolicy(QtCore.Qt.WheelFocus)
        
        self.main_window = main_window
        # put the controller and the view vertically
        vbox = QtGui.QVBoxLayout()
        
        # add context menu
        self.add_menu()
        
        # add controller
        # self.controller = QtGui.QPushButton()
        # vbox.addWidget(self.controller, stretch=1)
        
        # add the tree view
        self.view = self.create_tree_view(dh)
        vbox.addWidget(self.view, stretch=100)
        
        # self.restore_geometry()
        
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
        self.model = ClusterGroupManager(clusters_info=dh.clusters_info)
        
        # set the QTreeView options
        view = ClusterTreeView()
        view.set_model(self.model)
        return view

    def contextMenuEvent(self, event):
        action = self.context_menu.exec_(self.mapToGlobal(event.pos()))
            
    def add_group_action(self):
        groupindices = [g.groupidx() for g in self.model.get_groups()]
        groupidx = max(groupindices) + 1
        self.model.add_group(groupidx, "Group %d" % groupidx)
        self.view.expandAll()
    
    def remove_group_action(self):
        errors = []
        for groupidx in self.view.selected_groups():
            try:
                self.model.remove_group(groupidx)
            except:
                errors.append(groupidx)
        if errors:
            msg = "Non-empty groups were not deleted."
            self.main_window.statusBar().showMessage(msg, 5000)
    
    
    # Save and restore geometry
    # -------------------------
    def save_geometry(self):
        SETTINGS.set("clusterWidget/geometry", self.view.saveGeometry())
        SETTINGS.set("clusterWidget/headerState", self.view.header().saveState())
        
    def restore_geometry(self):
        g = SETTINGS.get("clusterWidget/geometry")
        h = SETTINGS.get("clusterWidget/headerState")
        if g:
            self.view.restoreGeometry(g)
        if h:
            self.view.header().restoreState(h)
    
    
    # def focusOutEvent(self, e):
        # self.view.focusOutEvent(e)
    
    