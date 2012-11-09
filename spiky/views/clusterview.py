import numpy as np
# from .tools import Info
# from ..dataio import MockDataProvider
from galry import *
# from views import *
# import ..tools
from collections import OrderedDict

# SETTINGS = tools.init_settings()

__all__ = ['ClusterGroupManager', 'ClusterWidget']

class TreeItem(object):
    def __init__(self, parent=None, data=None):
        """data is an OrderedDict"""
        self.parent_item = parent
        self.index = QtCore.QModelIndex()
        self.children = []
        # by default: root
        if data is None:
            data = OrderedDict(name='root')
        self.item_data = data
    
    def appendChild(self, child):
        self.children.append(child)
    
    def removeChild(self, child):
        self.children.remove(child)

    def child(self, row):
        return self.children[row]
        
    def rowCount(self):
        return len(self.children)
        
    def columnCount(self):
        return len(self.item_data)

    def data(self, column):
        return self.item_data.get(self.item_data.keys()[column], None)
        
    def row(self):
        if self.parent_item is None:
            return 0
        return self.parent_item.children.index(self)
        
    def parent(self):
        # if not hasattr(self, "parent_item"):
            # return None
        return self.parent_item
       
        
class TreeModel(QtCore.QAbstractItemModel):
    def __init__(self, headers):
        QtCore.QAbstractItemModel.__init__(self)
        self.root_item = TreeItem()
        self.headers = headers
        
    def add_node(self, item_class=None, parent=None, **kwargs):
        """Add a node in the tree.
        
        Arguments:
          * data: an OrderedDict instance.
          * parent=None: the parent of the node. If Node, parent is the root
            item.
        
        """
        if parent is None:
            parent = self.root_item
        if item_class is None:
            item_class = TreeItem
        item = item_class(parent=parent, **kwargs)
        
        row = parent.rowCount()
        item.index = self.createIndex(row, 0, item)
        
        self.beginInsertRows(parent.index, row-1, row-1)
        parent.appendChild(item)
        self.endInsertRows()
        
        return item
        
    def remove_node(self, child, parent=None):
        if parent is None:
            parent = self.root_item
            
        row = child.row()
        self.beginRemoveRows(parent.index, row, row)
        parent.removeChild(child)
        self.endRemoveRows()
        
    def get_descendants(self, parents):
        if type(parents) != list:
            parents = [parents]
        nodes = []
        for parent in parents:
            nodes.append(parent)
            if parent.children:
                nodes.extend(self.get_descendants(parent.children))
        return nodes
        
    def all_nodes(self):
        return self.get_descendants(self.root_item)
        
    def index(self, row, column, parent):
        if not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()
        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()
        child_item = parent_item.child(row)
        if child_item:
            index = self.createIndex(row, column, child_item)
            child_item.index = index
            return index
        else:
            return QtCore.QModelIndex()

    def parent(self, item):
        if not item.isValid():
            return QtCore.QModelIndex()
        item = item.internalPointer()
        parent_item = item.parent()
        if (parent_item == self.root_item):
            return QtCore.QModelIndex()
        index = self.createIndex(parent_item.row(), 0, parent_item)
        parent_item.index = index
        return index

    def rowCount(self, parent):
        if parent.column() > 0:
            return 0
        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()
        return parent_item.rowCount()
        
    def columnCount(self, parent):
        if not parent.isValid():
            return len(self.headers)
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
        return True

    def drag(self, target, sources):
        """
        
        To be overriden.
        
        """
        print "drag", target, sources

"""

To enable editing in your model, you must also implement setData(), and 
reimplement flags() to ensure that ItemIsEditable is returned. You can also 
reimplement headerData() and setHeaderData() to control the way the headers 
for your model are presented.
The dataChanged() and headerDataChanged() signals must be emitted explicitly 
when reimplementing the setData() and setHeaderData() functions, respectively.

"""

class ClusterItem(TreeItem):
    def __init__(self, parent=None, name=None, clusteridx=None):
        data = OrderedDict()
        # different columns fields
        data['name'] = name
        data['clusteridx'] = clusteridx
        super(ClusterItem, self).__init__(parent=parent, data=data)

    def name(self):
        return self.data(0)

    def clusteridx(self):
        return self.data(1)


class GroupItem(TreeItem):
    def __init__(self, parent=None, name=None, groupidx=None):
        data = OrderedDict()
        # different columns fields
        data['name'] = name
        data['groupidx'] = groupidx
        super(GroupItem, self).__init__(parent=parent, data=data)

    def name(self):
        return self.data(0)

    def groupidx(self):
        return self.data(1)
        

class ClusterGroupManager(TreeModel):
    def __init__(self, clusters, groups=None):
        """
        groups is a dict groupid => {name, description..}
        clusters is a dict idx => [name, ...}
        """
        super(ClusterGroupManager, self).__init__(['name', 'info'])
        for groupidx, group in groups.iteritems():
            # add the group node
            groupitem = self.add_group(groupidx, group['name'])
            for clusteridx, cluster in group['clusters'].iteritems():
                # add the cluster node as a child of the current group node
                clusteritem = self.add_cluster(clusteridx,
                    name=cluster['name'], parent=groupitem)
    
    def headerData(self, section, orientation, role):
        if (orientation == QtCore.Qt.Horizontal) and (role == QtCore.Qt.DisplayRole):
            return self.headers[section]
        
    def add_group(self, groupidx, name):
        groupitem = self.add_node(item_class=GroupItem, name=name,
            groupidx=groupidx)
        return groupitem
        
    def add_cluster(self, clusteridx, name, parent=None):
        if parent is None:
            parent = self.item_root
        cluster = self.add_node(item_class=ClusterItem, parent=parent, 
                            name=name, clusteridx=clusteridx)
        return cluster
        
    def drag(self, target, sources):
        # get source ClusterItem nodes
        source_items = []
        nodes = self.all_nodes()
        for node in nodes:
            if str(node) in sources and type(node) == ClusterItem \
                and node not in source_items:
                source_items.append(node)
                
        # get the groupidx if the target is a group
        if type(target) == GroupItem:
            groupidx = target.groupidx()
        # else, if it is a cluster, take the corresponding group
        elif type(target) == ClusterItem:
            groupidx = self.get_groupidx(target.clusteridx())
            
        # assign groups to selected clusters
        for source in source_items:
            clusteridx = source.clusteridx()
            oldgroupidx = self.get_groupidx(clusteridx)
            if oldgroupidx != groupidx:
                self.assign(clusteridx, groupidx)
        
    def get_groups(self):
        return [group for group in self.get_descendants(self.root_item) \
            if (type(group) == GroupItem)]
        
    def get_group(self, groupidx):
        return [group for group in self.get_descendants(self.root_item) \
            if (type(group) == GroupItem) and \
                (group.groupidx() == groupidx)][0]
        
    def get_clusters(self):
        return [cluster for cluster in self.get_descendants(self.root_item) \
          if (type(cluster) == ClusterItem)]
            
    def get_cluster(self, clusteridx):
        return [cluster for cluster in self.get_descendants(self.root_item) \
          if (type(cluster) == ClusterItem) and \
                (cluster.clusteridx() == clusteridx)][0]
                
    def get_clusters_in_group(self, groupidx):
        group = self.get_group(groupidx)
        return [cluster for cluster in self.get_descendants(group) \
            if (type(cluster) == ClusterItem)]
        
    def get_groupidx(self, clusteridx):
        """Return the group index currently assigned to the specifyed cluster
        index."""
        for group in self.get_groups():
            clusterindices = [cluster.clusteridx() \
                            for cluster in self.get_clusters_in_group(group.groupidx())]
            if clusteridx in clusterindices:
                return group.groupidx()
        return None
            
    def assign(self, clusteridx, groupidx):
        """Assign a group to a cluster."""
        # remove this cluster from its previous group
        oldgroup = self.get_group(self.get_groupidx(clusteridx))
        oldgroupidx = oldgroup.groupidx()
        newgroup = self.get_group(groupidx)
        cluster = self.get_cluster(clusteridx)
        # oldrow = (self.get_clusters_in_group(oldgroupidx)).index(cluster)
        # add cluster in the new group
        self.add_cluster(clusteridx, cluster.name(), newgroup)
        # remove it from the old group
        self.remove_node(cluster, oldgroup)

        

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
        
        self.view.setModel(clm)
        self.view.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        
        vbox.addWidget(self.view, stretch=100)
        
        # set the VBox as layout of the widget
        self.setLayout(vbox)


# if __name__ == '__main__':



    # class SpikyMainWindow(QtGui.QMainWindow):
        # window_title = "Spiky"
        
        # def __init__(self):
            # super(SpikyMainWindow, self).__init__()
            # # parameters related to docking
            # self.setAnimated(False)
            # self.setTabPosition(
                # QtCore.Qt.LeftDockWidgetArea |
                # QtCore.Qt.RightDockWidgetArea |
                # QtCore.Qt.TopDockWidgetArea |
                # QtCore.Qt.BottomDockWidgetArea,
                # QtGui.QTabWidget.North)
            # self.setDockNestingEnabled(True)
            # self.setWindowTitle(self.window_title)
            # # make the UI initialization
            # self.initialize()
            # self.restore_geometry()
            # self.show()
            
        # def initialize(self):
            # """Make the UI initialization."""
            
            # # load mock data
            # provider = MockDataProvider()
            # self.dh = provider.load(nspikes=100)
            # self.add_dock(ClusterWidget, QtCore.Qt.RightDockWidgetArea)

            

        # def add_dock(self, widget_class, position, name=None, minsize=None):
            # """Add a dockable widget"""
            # if name is None:
                # name = widget_class.__name__
            # widget = widget_class(self.dh)
            # if minsize is not None:
                # widget.setMinimumSize(*minsize)
            # dockwidget = QtGui.QDockWidget(name)
            # dockwidget.setObjectName(name)
            # dockwidget.setWidget(widget)
            # dockwidget.setFeatures(QtGui.QDockWidget.DockWidgetFloatable | \
                # QtGui.QDockWidget.DockWidgetMovable)
            # self.addDockWidget(position, dockwidget)
            
        # def add_central(self, widget_class, name=None, minsize=None):
            # """Add a central widget in the main window."""
            # if name is None:
                # name = widget_class.__name__
            # widget = widget_class(self.dh)
            # widget.setObjectName(name)
            # if minsize is not None:
                # widget.setMinimumSize(*minsize)
            # self.setCentralWidget(widget)
            
        # def save_geometry(self):
            # """Save the arrangement of the whole window into a INI file."""
            # SETTINGS.set("mainWindow/geometry", self.saveGeometry())
            # SETTINGS.set("mainWindow/windowState", self.saveState())
            
        # def restore_geometry(self):
            # """Restore the arrangement of the whole window from a INI file."""
            # g = SETTINGS.get("mainWindow/geometry")
            # w = SETTINGS.get("mainWindow/windowState")
            # if g:
                # self.restoreGeometry(g)
            # if w:
                # self.restoreState(w)
            
        # def closeEvent(self, e):
            # """Automatically save the arrangement of the window when closing
            # the window."""
            # self.save_geometry()
            # super(SpikyMainWindow, self).closeEvent(e)


    # window = show_window(SpikyMainWindow)



# def create_group_data(name="", idx=0):
    # item = OrderedDict()
    # item["name"] = name
    # # TODO: add other fields
    # item["groupidx"] = idx
    # return item
# def create_cluster_data(name="", idx=0):
    # item = OrderedDict()
    # item["name"] = name
    # # TODO: add other fields
    # item["clusteridx"] = idx
    # return item