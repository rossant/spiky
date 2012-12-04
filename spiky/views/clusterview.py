import numpy as np
import numpy.random as rnd
from galry import *
from collections import OrderedDict
from signals import *
from colors import COLORMAP

__all__ = ['ClusterGroupManager', 'ClusterItem', 'GroupItem',
           'ClusterTreeView']


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
        if column >= self.columnCount():
            return None
        return self.item_data.get(self.item_data.keys()[column], None)
        
    def row(self):
        if self.parent_item is None:
            return 0
        return self.parent_item.children.index(self)
        
    def parent(self):
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

    def rowCount(self, parent=None):
        if parent is None:
            parent = QtCore.QModelIndex()
        if parent.column() > 0:
            return 0
        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()
        return parent_item.rowCount()
        
    def columnCount(self, parent=None):
        if parent is None:
            parent = QtCore.QModelIndex()
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
    def __init__(self, parent=None, name=None, clusteridx=None, color=None,
            rate=None):
        if color is None:
            color = 0#(1., 1., 1.)
        data = OrderedDict()
        # different columns fields
        data['name'] = name
        data['rate'] = rate
        data['color'] = color
        # the index is the last column
        data['clusteridx'] = clusteridx
        super(ClusterItem, self).__init__(parent=parent, data=data)

    def name(self):
        return self.item_data['name']

    def rate(self):
        return self.item_data['rate']

    def color(self):
        return self.item_data['color']
                
    def clusteridx(self):
        return self.item_data['clusteridx']


class GroupItem(TreeItem):
    def __init__(self, parent=None, name=None, groupidx=None):
        data = OrderedDict()
        # different columns fields
        data['name'] = name
        data['rate'] = None
        data['color'] = None
        # the index is the last column
        data['groupidx'] = groupidx
        super(GroupItem, self).__init__(parent=parent, data=data)

    def name(self):
        return self.item_data['name']

    def groupidx(self):
        return self.item_data['groupidx']
        

class ClusterGroupManager(TreeModel):
    headers = ['name', 'rate', 'color']
    
    def __init__(self, clusters_info=None):
        """Initialize the tree model.
        
        Arguments:
          * clusters: a Nspikes long array with the cluster index for each
            spike.
          * clusters_info: an Info object with fields names, colors, rates,
            groups_info.
        
        """
        super(ClusterGroupManager, self).__init__(self.headers)
        self.initialize(clusters_info=clusters_info)
        
    def initialize(self, clusters_info=None):
        for idx, groupinfo in enumerate(clusters_info.groups_info):
            groupitem = self.add_group(idx, groupinfo['name'])
            clusterindices = sorted(np.nonzero(clusters_info.groups == idx)[0])
            for clusteridx in clusterindices:
                clusteritem = self.add_cluster(
                    clusteridx,
                    name=clusters_info.names[clusteridx],
                    color=clusters_info.colors[clusteridx],
                    rate=clusters_info.rates[clusteridx],
                    parent=groupitem)
    
    def headerData(self, section, orientation, role):
        if (orientation == QtCore.Qt.Horizontal) and (role == QtCore.Qt.DisplayRole):
            return self.headers[section]
        
    def add_group(self, groupidx, name):
        """Add a group."""
        groupitem = self.add_node(item_class=GroupItem, name=name,
            groupidx=groupidx)
        return groupitem
        
    def remove_group(self, groupidx):
        """Remove an empty group. Raise an error if the group is not empty."""
        # check that the group is empty
        if self.get_clusters_in_group(groupidx):
            raise ValueError("group %d is not empty, unable to delete it" % \
                    groupidx)
        groups = [g for g in self.get_groups() if g.groupidx() == groupidx]
        if groups:
            group = groups[0]
            self.remove_node(group)
        else:
            log_warn("group %d does not exist" % groupidx)
        
    def add_cluster(self, clusteridx, name, color=None, rate=None,
                    parent=None):
        # if parent is None:
            # parent = self.item_root
        cluster = self.add_node(item_class=ClusterItem, parent=parent, 
                            name=name, color=color, rate=rate,
                            clusteridx=clusteridx)
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
        
    def data(self, index, role):
        """Return custom background color for the last column of cluster
        items."""
        item = index.internalPointer()
        col = index.column()
        # group item
        if type(item) == GroupItem:
            if role == QtCore.Qt.DisplayRole:
                return item.data(col)
        # cluster item
        if type(item) == ClusterItem:
            # rate
            if col == 1:
                if role == QtCore.Qt.TextAlignmentRole:
                    return QtCore.Qt.AlignRight
                if role == QtCore.Qt.DisplayRole:
                    return "%.1f Hz" % item.rate()
            # color
            elif col == self.columnCount() - 1:
                if role == QtCore.Qt.BackgroundRole:
                    color = np.array(COLORMAP[item.color()]) * 255
                    return QtGui.QColor(*color)
                    
        # default
        if role == QtCore.Qt.DisplayRole:
            return item.data(col)
        # all text in black
        if role == QtCore.Qt.ForegroundRole:
            return QtGui.QBrush(QtGui.QColor(0, 0, 0, 255))
        
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
        # add cluster in the new group
        self.add_cluster(clusteridx, name=cluster.name(), parent=newgroup,
            rate=cluster.rate(), color=cluster.color())
        # remove it from the old group
        self.remove_node(cluster, oldgroup)

        
class ClusterTreeView(QtGui.QTreeView):
    class ClusterDelegate(QtGui.QStyledItemDelegate):
        def paint(self, painter, option, index):
            """Disable the color column so that the color remains the same even
            when it is selected."""
            # deactivate all columns except the first one, so that selection
            # is only possible in the first column
            if index.column() >= 1:
                if option.state and QtGui.QStyle.State_Selected:
                    option.state = option.state and QtGui.QStyle.State_Off
            super(ClusterTreeView.ClusterDelegate, self).paint(painter, option, index)
    
    def set_model(self, model):
        # Capture keyboard events.
        # getfocus = False
        # if getfocus:
        self.setFocusPolicy(QtCore.Qt.NoFocus)
    
        self.setStyleSheet("""
        QTreeView::item:selected {
            background-color: #3399ff;
            color: #ffffff;
        }
        """);
        
        self.setModel(model)
        self.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        self.expandAll()
        self.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.setAllColumnsShowFocus(True)
        self.setFirstColumnSpanned(0, QtCore.QModelIndex(), True)
        # select full rows
        self.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        
        # # set rate column size
        # self.header().resizeSection(1, 80)
        # # set color column size
        self.header().resizeSection(2, 40)
        
        
        # self.setRootIsDecorated(False)
        self.setItemDelegate(self.ClusterDelegate())
    
    
    # Cluster methods
    # ---------------
    def get_clusters(self):
        return self.model().get_clusters()
    
    def get_cluster(self, clusteridx):
        return self.model().get_cluster(clusteridx)
    
    
    # Selection methods
    # -----------------
    def selectionChanged(self, selected, deselected):
        super(ClusterTreeView, self).selectionChanged(selected, deselected)
        # emit the ClusterSelectionToChange signal
        emit(self, "ClusterSelectionToChange",
            np.sort(np.array(self.selected_clusters(), dtype=np.int32)))
        
    def select(self, cluster):
        """Select a cluster.
        
        Arguments:
          * cluster: either a clusteridx integer, ClusterItem instance,
            or a QModelIndex instance.
          
        """
        # if cluster is an int, get the ClusterItem
        if (type(cluster) != QtCore.QModelIndex and 
                type(cluster) != ClusterItem):
            cluster = self.get_cluster(cluster)
        # now, cluster shoud be a ClusterItem, so we take the QModelIndex
        if isinstance(cluster, ClusterItem):
            cluster = cluster.index
        # finally, cluster should be a QModelIndex instance here
        sel_model = self.selectionModel()
        # sel_model.clearSelection()
        # sel_model.setCurrentIndex(cluster, sel_model.Current)
        sel_model.select(cluster, sel_model.Clear | sel_model.SelectCurrent | sel_model.Rows)
        self.scrollTo(cluster, QtGui.QAbstractItemView.EnsureVisible)
        
    def select_cluster(self, direction):
        # list of all cluster indices
        clusters = [cluster.clusteridx() for cluster in self.get_clusters()]
        if not clusters:
            return
        # list of selected cluster indices
        selected = self.selected_clusters()
        
        # find the cluster to select
        to_select = None
        # previous case
        if direction == 'previous':
            if len(selected) == 0:
                to_select = clusters[-1]
            else:
                i = max(0, clusters.index(selected[0]) - 1)
                to_select = clusters[i]
        # next case
        if direction == 'next':
            if len(selected) == 0:
                to_select = clusters[0]
            else:
                i = min(len(clusters) - 1, clusters.index(selected[-1]) + 1)
                to_select = clusters[i]
        if direction == 'top':
            to_select = clusters[0]
        if direction == 'bottom':
            to_select = clusters[-1]
                
        # select the cluster
        if to_select is not None:
            self.select(to_select)
        
    def selected_clusters(self):
        """Return the list of selected cluster indices."""
        return [(v.internalPointer().clusteridx()) \
                    for v in self.selectedIndexes() \
                        if v.column() == 0 and \
                           type(v.internalPointer()) == ClusterItem]
              
    def selected_groups(self):
        """Return the list of selected groups."""
        return [(v.internalPointer().groupidx()) \
                    for v in self.selectedIndexes() \
                        if v.column() == 0 and \
                           type(v.internalPointer()) == GroupItem]
                

    # Event methods
    # -------------
    modifiers = [QtCore.Qt.Key_Control, QtCore.Qt.Key_Shift, QtCore.Qt.Key_Alt]
    modifier = False
    def keyReleaseEvent(self, e):
        key = e.key()
        if key in self.modifiers:
            self.modifier = False
        # print "release", self.modifier
            
    def keyPressEvent(self, e):
        key = e.key()
        if key in self.modifiers:
            self.modifier = True
        # print "press", self.modifier
        if self.modifier:
            return
        
        # previous and next cluster
        if key == QtCore.Qt.Key_Up:
            self.select_cluster('previous')
        if key == QtCore.Qt.Key_Down:
            self.select_cluster('next')
        
        # first and last cluster
        if key == QtCore.Qt.Key_Home:
            self.select_cluster('top')
        if key == QtCore.Qt.Key_End:
            self.select_cluster('bottom')
            
    # def focusOutEvent(self, e):
        # self.modifier = False
        # print "out", self.modifier
    
    