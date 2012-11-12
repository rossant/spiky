from galry import *
from views import *
import tools
import numpy as np
import numpy.random as rnd
from dataio import MockDataProvider
from tools import Info
from collections import OrderedDict

SETTINGS = tools.init_settings()

__all__ = ['WaveformWidget',
           'FeatureWidget',
           'CorrelogramsWidget',
           'CorrelationMatrixWidget',
           'ClusterWidget',
           ]



def get_default_widget_controller():
    vbox = QtGui.QVBoxLayout()
    
    return vbox


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
        self.isolated_control = QtGui.QCheckBox("isolated")
        hbox.addWidget(self.isolated_control, stretch=1, alignment=QtCore.Qt.AlignLeft)
        
        # add the reset view button
        self.reset_view_control = QtGui.QPushButton("reset view")
        hbox.addWidget(self.reset_view_control, stretch=1, alignment=QtCore.Qt.AlignLeft)
        
        # hbox.addWidget(QtGui.QCheckBox("test"), stretch=1, alignment=QtCore.Qt.AlignLeft)
        # add lots of space to the right to make sure everything is aligned to 
        # the left
        hbox.addStretch(100)
        
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
        view = FeatureView()
        view.set_data(dh.features, clusters=dh.clusters,
                      fetdim=3,
                      cluster_colors=dh.clusters_info.colors,
                      masks=dh.masks)
        return view

    def create_controller(self):
        box = super(FeatureWidget, self).create_controller()
        # box.addWidget(QtGui.QCheckBox("hi"))
        return box
    
    
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
        
        # add controller
        self.controller = QtGui.QPushButton()
        vbox.addWidget(self.controller, stretch=1)
        
        # pass the cluster data to the ClusterView
        model = ClusterGroupManager(clusters=dh.clusters,
                                  clusters_info=dh.clusters_info)
        
        # set the QTreeView options
        self.view = QtGui.QTreeView()
        self.view.setModel(model)
        self.view.header().resizeSection(2, 20)
        self.view.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        self.view.expandAll()
        self.view.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.view.setAllColumnsShowFocus(True)
        self.view.setFirstColumnSpanned(0, QtCore.QModelIndex(), True)
        self.view.setRootIsDecorated(False)
        self.view.setItemDelegate(self.ClusterDelegate())
        # self.setStyleSheet(STYLESHEET)
        
        vbox.addWidget(self.view, stretch=100)
        
        # set the VBox as layout of the widget
        self.setLayout(vbox)



