from galry import *
from views import *
import tools
import numpy as np
from dataio import MockDataProvider
from tools import Info
from collections import OrderedDict

SETTINGS = tools.init_settings()

__all__ = ['SpikyMainWindow']



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





        
class TreeItem(object):
    def __init__(self, data, parent=None):
        """data is an OrderedDict"""
        self.children = []
        self.item_data = data
        self.parent_item = parent
    
    def appendChild(self, child):
        self.children.append(child)
    
    def removeChild(self, child):
        self.children.remove(child)

    def child(self, row):
        return self.children[row]
        
    def childCount(self):
        return len(self.children)
        
    def columnCount(self):
        return len(self.item_data)

    def data(self, column):
        return self.item_data.get(self.item_data.keys()[column], None)
        
    def row(self):
        return self.parent_item.children.index(self)
        
    def parent(self):
        return self.parent_item
        

    # DEBUG
    def __repr__(self):
        return self.item_data["name"]
        
        
def create_group_data(name="", idx=0):
    item = OrderedDict()
    item["name"] = name
    # TODO: add other fields
    item["groupidx"] = idx
    return item
def create_cluster_data(name="", idx=0):
    item = OrderedDict()
    item["name"] = name
    # TODO: add other fields
    item["clusteridx"] = idx
    return item

    
    
class ClusterGroupManager(object):
    def __init__(self, clusters, groups=None):
        """
        groups is a dict groupid => {name, description..}
        clusters is a dict idx => [name, ...}
        """
        self.clusters = clusters
        if groups is None:
            groups = {0: dict(name="Group0", clusters=clusters)}
        self.groups = groups
        self.initialize_treemodel()
        
    def initialize_treemodel(self):
        self.treemodel = TreeModel()
        for groupidx, group in self.groups.iteritems():
            group["treeitem"] = self.treemodel.add_node(create_group_data(group['name'], groupidx))
            for clusteridx, cluster in group['clusters'].iteritems():
                cluster["treeitem"] = self.treemodel.add_node(create_cluster_data(cluster['name'], clusteridx), group["treeitem"])
        self.treemodel.drag = self.drag
        
    def drag(self, target, sources):
        source_items = []
        nodes = self.treemodel.all_nodes()
        for node in nodes:
            if str(node) in sources:
                source_items.append(node)
        # get the groupidx if the target is a group
        groupidx = target.item_data.get('groupidx', None)
        # if it is a cluster, take the corresponding group
        print target.item_data
        if groupidx is None:
            groupidx = self.get_group(target.item_data.get('clusteridx', None))
        # assign groups to selected clusters
        for source in source_items:
            print source.item_data
            clusteridx = source.item_data.get('clusteridx', None)
            # print clusteridx, groupidx
            self.assign(clusteridx, groupidx)
        print
        
    def get_group(self, clusteridx):
        """Return the group index currently assigned to the specifyed cluster
        index."""
        for groupidx, group in self.groups.iteritems():
            if clusteridx in group['clusters'].keys():
                return groupidx
        return None
        
    def assign(self, clusteridx, groupidx):
        """Assign a group to a cluster."""
        # remove this cluster from its previous group
        oldgroup = self.groups[self.get_group(clusteridx)]
        newgroup = self.groups[groupidx]
        cluster = oldgroup['clusters'][clusteridx]
        row = oldgroup['clusters'].values().index(cluster)
        newgroup['clusters'][clusteridx] = cluster
        del oldgroup['clusters'][clusteridx]
        
        self.treemodel.add_node(create_cluster_data(cluster['name'], clusteridx), newgroup['treeitem'])
        self.treemodel.remove_node(row, oldgroup['treeitem'])
        # add it to the new group
        # self.groups[groupidx]['clusters'][clusteridx] = .append(clusteridx)
        # make the update in the treeview model too
        
        # row = self.groups[groupidx]['clusters'].values().index(cluster)
        # print self.groups[groupidx]['clusters'].values(), row, cluster
        # self.treemodel.remove_node(row, oldgroup["treeitem"])



class TreeModel(QtCore.QAbstractItemModel):
    def __init__(self):
        QtCore.QAbstractItemModel.__init__(self)
        self.root_item = TreeItem(dict(name="root"))
        # self.setHeaderData(1, QtCore.Qt.Horizontal, "Cluster")
        
    def add_node(self, data, parent=None):
        """Add a node in the tree.
        
        Arguments:
          * data: any Python object indexable with an integer, such that
            element 0 is the display name.
          * parent=None: the parent of the node. If Node, parent is the root
            item.
        
        """
        if parent is None:
            parent = self.root_item
        item = TreeItem(data, parent)
        parent.appendChild(item)
        return item
        
    def remove_node(self, row, parent=None):
        if parent is None:
            parent = self.root_item
        parent.removeChild(parent.child(row))

    def get_nodes(self, parents):
        if type(parents) != list:
            parents = [parents]
        nodes = []
        for parent in parents:
            nodes.append(parent)
            if parent.children:
                nodes.extend(self.get_nodes(parent.children))
        return nodes
        
    def all_nodes(self):
        return self.get_nodes(self.root_item)
        
    def index(self, row, column, parent):
        if not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()
        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()
        child_item = parent_item.child(row)
        if child_item:
            return self.createIndex(row, column, child_item)
        else:
            return QtCore.QModelIndex()

    def parent(self, parent):
        if not parent.isValid():
            return QtCore.QModelIndex()
        child_item = parent.internalPointer()
        parent_item = child_item.parent()
        if (parent_item == self.root_item):
            return QtCore.QModelIndex()
        return self.createIndex(parent_item.row(), 0, parent_item)

    def rowCount(self, parent):
        if parent.column() > 0:
            return 0
        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()
        return parent_item.childCount()
        
    def columnCount(self, parent):
        if not parent.isValid():
            return 1
        return parent.internalPointer().columnCount()

    def data(self, index, role):
        if role != QtCore.Qt.DisplayRole:
            return None
        item = index.internalPointer()
        return item.data(index.column())

    def supportedDropActions(self): 
        return QtCore.Qt.MoveAction         

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.ItemIsEnabled
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | \
               QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsDropEnabled        

    def mimeTypes(self):
        return ['text/xml']

    def mimeData(self, indexes):
        data = ",".join(set([str(index.internalPointer()) for index in indexes]))
        mimedata = QtCore.QMimeData()
        mimedata.setData('text/xml', data)
        return mimedata

    def dropMimeData(self, data, action, row, column, parent):
        parent_item = parent.internalPointer()
        target = parent_item
        sources = data.data('text/xml').split(',')
        self.drag(target, sources)
        # print target, sources
        # TODO: handle this
        return True

    def drag(self, target, sources):
        print target, sources
        

        

class ClusterWidget(QtGui.QWidget):
    def __init__(self, dataholder):
        super(ClusterWidget, self).__init__()
        # put the controller and the view vertically
        vbox = QtGui.QVBoxLayout()
        
        self.controller = QtGui.QPushButton()
        vbox.addWidget(self.controller, stretch=1)
        
        self.view = QtGui.QTreeView()
        self.view.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        
        clusters1 = OrderedDict()
        for i in xrange(5):
            clusters1[i] = dict(name='cluster%d' % i)
        clusters2 = OrderedDict()
        for i in xrange(5, 10):
            clusters2[i] = dict(name='cluster%d' % i)
        
        groups = {0: dict(name="group0", clusters=clusters1),
                  1: dict(name="group1", clusters=clusters2)}
                  
        clusters = clusters1.copy()
        clusters.update(clusters2)
        
        clm = ClusterGroupManager(clusters, groups)
        
        self.view.setModel(clm.treemodel)
        self.view.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        
        vbox.addWidget(self.view, stretch=100)
        
        
        # set the VBox as layout of the widget
        self.setLayout(vbox)




        
        
        
        



class SpikyMainWindow(QtGui.QMainWindow):
    window_title = "Spiky"
    
    def __init__(self):
        super(SpikyMainWindow, self).__init__()
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
        # make the UI initialization
        self.initialize()
        self.restore_geometry()
        self.show()
        
    def initialize(self):
        """Make the UI initialization."""
        
        # load mock data
        provider = MockDataProvider()
        self.dh = provider.load(nspikes=100)
        
        # central window, the dockable widgets are arranged around it
        # self.add_central(FeatureWidget)
        
        # dockable widgets
        # self.add_dock(WaveformWidget, QtCore.Qt.RightDockWidgetArea)        
        # self.add_dock(CorrelogramsWidget, QtCore.Qt.RightDockWidgetArea)
        # self.add_dock(CorrelationMatrixWidget, QtCore.Qt.RightDockWidgetArea)
        self.add_dock(ClusterWidget, QtCore.Qt.RightDockWidgetArea)

        

    def add_dock(self, widget_class, position, name=None, minsize=None):
        """Add a dockable widget"""
        if name is None:
            name = widget_class.__name__
        widget = widget_class(self.dh)
        if minsize is not None:
            widget.setMinimumSize(*minsize)
        dockwidget = QtGui.QDockWidget(name)
        dockwidget.setObjectName(name)
        dockwidget.setWidget(widget)
        dockwidget.setFeatures(QtGui.QDockWidget.DockWidgetFloatable | \
            QtGui.QDockWidget.DockWidgetMovable)
        self.addDockWidget(position, dockwidget)
        
    def add_central(self, widget_class, name=None, minsize=None):
        """Add a central widget in the main window."""
        if name is None:
            name = widget_class.__name__
        widget = widget_class(self.dh)
        widget.setObjectName(name)
        if minsize is not None:
            widget.setMinimumSize(*minsize)
        self.setCentralWidget(widget)
        
    def save_geometry(self):
        """Save the arrangement of the whole window into a INI file."""
        SETTINGS.set("mainWindow/geometry", self.saveGeometry())
        SETTINGS.set("mainWindow/windowState", self.saveState())
        
    def restore_geometry(self):
        """Restore the arrangement of the whole window from a INI file."""
        g = SETTINGS.get("mainWindow/geometry")
        w = SETTINGS.get("mainWindow/windowState")
        if g:
            self.restoreGeometry(g)
        if w:
            self.restoreState(w)
        
    def closeEvent(self, e):
        """Automatically save the arrangement of the window when closing
        the window."""
        self.save_geometry()
        super(SpikyMainWindow, self).closeEvent(e)


if __name__ == '__main__':
    window = show_window(SpikyMainWindow)


