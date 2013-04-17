import pprint
import numpy as np
import numpy.random as rnd
from galry import *
from collections import OrderedDict
# from signals import *
# from colors import COLORMAP
from spiky.utils.settings import SETTINGS
# import spiky.gui.signals as ssignals
from spiky.utils.colors import COLORMAP


# __all__ = ['ClusterGroupManager', 'ClusterItem', 'GroupItem',
           # 'ClusterTreeView', 'ClusterWidget']


# Generic classes
# ---------------
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

    def setData(self, index, data, role):
        return False
        
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


# Specific item classes
# ---------------------
class ClusterItem(TreeItem):
    def __init__(self, parent=None, clusteridx=None, color=None, #name=None,
            spkcount=None):
        if color is None:
            color = 0 #(1., 1., 1.)
        data = OrderedDict()
        # different columns fields
        # data['name'] = str(name)
        data['spkcount'] = spkcount
        data['color'] = color
        # the index is the last column
        data['clusteridx'] = clusteridx
        super(ClusterItem, self).__init__(parent=parent, data=data)

    # def name(self):
        # return self.item_data['name']

    def spkcount(self):
        return self.item_data['spkcount']

    def color(self):
        return self.item_data['color']
                
    def clusteridx(self):
        return self.item_data['clusteridx']


class GroupItem(TreeItem):
    def __init__(self, parent=None, name=None, groupidx=None, color=None, spkcount=None):
        data = OrderedDict()
        # different columns fields
        data['name'] = name
        data['spkcount'] = spkcount
        data['color'] = color
        # the index is the last column
        data['groupidx'] = groupidx
        super(GroupItem, self).__init__(parent=parent, data=data)

    def name(self):
        return self.item_data['name']

    def color(self):
        return self.item_data['color']

    def spkcount(self):
        return self.item_data['spkcount']

    def groupidx(self):
        return self.item_data['groupidx']
        
    def __repr__(self):
        return "<group {0:d} '{1:s}'>".format(self.groupidx(), self.name())
        

# Custom model
# ------------
class ClusterGroupManager(TreeModel):
    headers = ['Cluster', 'Spikes', 'Color']
    
    def __init__(self, info=None):
        """Initialize the tree model.
        
        Arguments:
          * clusters: a Nspikes long array with the cluster index for each
            spike.
          * clusters_info: an Info object with fields names, colors, spkcounts,
            groups_info.
        
        """
        super(ClusterGroupManager, self).__init__(self.headers)
        self.from_dict(info)
        
    
    # I/O methods
    # -----------
    def from_dict(self, info):
        # go through all groups
        for groupidx, groupinfo in info['groups_info'].iteritems():
            # add group
            spkcount = np.sum([cl['spkcount'] for cl in info['clusters_info'].values() if cl['groupidx'] == groupidx])
            groupitem = self.add_group(groupidx=groupidx, name=groupinfo['name'],
                color=groupinfo['color'], spkcount=spkcount)
        
        # go through all clusters
        for clusteridx, clusterinfo in info['clusters_info'].iteritems():
            # cluster = info.clusters_info[clusteridx]
            # add cluster
            clusteritem = self.add_cluster(
                clusteridx=clusteridx,
                # name=info.names[clusteridx],
                color=clusterinfo['color'],
                spkcount=clusterinfo['spkcount'],
                # assign the group as a parent of this cluster
                parent=self.get_group(clusterinfo['groupidx']))
    
    def to_dict(self):
        dic = {'groups_info': {}, 'clusters_info': {}}
        groups = self.get_groups()
        allclusters = self.get_clusters()
        # dic['groups'] = np.zeros(len(allclusters), dtype=np.int32)
        for group in groups:
            groupidx = group.groupidx()
            clusters = self.get_clusters_in_group(groupidx)
            # group spike count
            spkcount = np.array([cluster.spkcount() for cluster in clusters]).sum()
            # set the group info object
            dic['groups_info'][groupidx] = {
                    'groupidx': groupidx,
                    'name': group.name(),
                    'color': group.color(),
                    'spkcount': spkcount,
                }
            # assign the group to all the clusters in the group
            # dic['groups'][clusters] = groupidx
            for cluster in clusters:
                clusteridx = cluster.clusteridx()
                dic['clusters_info'][clusteridx] = {
                        'clusteridx': clusteridx,
                        'groupidx': groupidx,
                        'color': cluster.color(),
                        'spkcount': cluster.spkcount(),
                    }
        
        # DEBUG
        # pprint.pprint(dic)
        
        return dic
    
    
    # Data methods
    # ------------
    def headerData(self, section, orientation, role):
        if (orientation == QtCore.Qt.Horizontal) and (role == QtCore.Qt.DisplayRole):
            return self.headers[section]
        
    def data(self, index, role):
        """Return custom background color for the last column of cluster
        items."""
        item = index.internalPointer()
        
        # default
        # if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
            # return item.data(index.column())
        
        col = index.column()
        # group item
        if type(item) == GroupItem:
            if col == 0:
                if role == QtCore.Qt.DisplayRole:
                    return str(item.name())
            # spkcount
            elif col == 1:
                if role == QtCore.Qt.TextAlignmentRole:
                    return QtCore.Qt.AlignRight
                if role == QtCore.Qt.DisplayRole:
                    return "%d" % item.spkcount()
            # color
            elif col == self.columnCount() - 1:
                if role == QtCore.Qt.BackgroundRole:
                    if item.color() >= 0:
                        color = np.array(COLORMAP[item.color()]) * 255
                        return QtGui.QColor(*color)
                elif role == QtCore.Qt.DisplayRole:
                    return ""
                
        # cluster item
        if type(item) == ClusterItem:
            # clusteridx
            if col == 0:
                if role == QtCore.Qt.DisplayRole:
                    return str(item.clusteridx())
            # spkcount
            elif col == 1:
                if role == QtCore.Qt.TextAlignmentRole:
                    return QtCore.Qt.AlignRight
                if role == QtCore.Qt.DisplayRole:
                    return "%d" % item.spkcount()
            # color
            elif col == self.columnCount() - 1:
                if role == QtCore.Qt.BackgroundRole:
                    color = np.array(COLORMAP[item.color()]) * 255
                    return QtGui.QColor(*color)
                    
        # default
        if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
            return item.data(col)
        
        # all text in black
        # if role == QtCore.Qt.ForegroundRole:
            # return QtGui.QBrush(QtGui.QColor(0, 0, 0, 255))
          
    def setData(self, index, data, role=None):
        # print index, data, role, index.isValid()
        if role is None:
            role = QtCore.Qt.EditRole
        if index.isValid() and role == QtCore.Qt.EditRole:
            item = index.internalPointer()
            if index.column() == 0:
                item.item_data['name'] = data
                # ssignals.emit(self, 'RenameGroupRequested', item.groupidx(), data)
            elif index.column() == 1:
                item.item_data['spkcount'] = data
            elif index.column() == 2:
                item.item_data['color'] = data
            self.dataChanged.emit(index, index)
            # ssignals.emit(self, "ClusterInfoToUpdate")
            return True
    
    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.ItemIsEnabled
        # editable: only groups, and first column
        # if isinstance(index.internalPointer(), GroupItem) and index.column() == 0:
            # return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | \
                   # QtCore.Qt.ItemIsEditable | \
                   # QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsDropEnabled
        # else:
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | \
               QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsDropEnabled
        
    
    # Tree methods
    # ------------
    def add_group(self, **kwargs):
        """Add a group."""
        groupitem = self.add_node(item_class=GroupItem, **kwargs)
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
        
    def add_cluster(self, parent=None, **kwargs):
        cluster = self.add_node(item_class=ClusterItem, parent=parent, 
                            **kwargs)
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
        else:
            # empty target
            return
            
        # clusters to move
        clusters = np.array([source.clusteridx() for source in source_items])
        clusters = np.array([source.clusteridx() for source in source_items])
        # emit signals
        ssignals.emit(self, "MoveClustersRequested", clusters, groupidx)
        
    
    # Getter methods
    # --------------
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
        l = [cluster for cluster in self.get_descendants(self.root_item) \
                  if (type(cluster) == ClusterItem) and \
                        (cluster.clusteridx() == clusteridx)]
        if l:
            return l[0]
                
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
    
        # self.setStyleSheet("""
        # QTreeView {
            # background-color: #000000;
            # color: #b1b1b1;
        # }
        # QTreeView::item {
            # color: #b1b1b1;
        # }
        # QTreeView::item:selected {
            # /*background-color: #3399ff;*/
            # color: #000000;
        # }
        # """)
        
        self.setModel(model)
        self.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        self.expandAll()
        self.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.setAllColumnsShowFocus(True)
        # self.setFirstColumnSpanned(0, QtCore.QModelIndex(), True)
        # select full rows
        self.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        
        # # set spkcount column size
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
        can_signal_selection = getattr(self, 'can_signal_selection', True)
        can_signal_selection = can_signal_selection and getattr(self.model(), 'can_signal_selection', True)
        
        if can_signal_selection:
            # emit the ClusterSelectionToChange signal
            clusters = self.selected_clusters()
            groups = self.selected_groups()
            allclusters = []
            # groups first
            for group in groups:
                allclusters.extend([cl.clusteridx() for cl in self.model().get_clusters_in_group(group)])
                
            # add clusters that are not in the selected groups, and
            # remove the others
            clusters_to_add = []
            for cluster in clusters:
                if cluster not in allclusters:
                    clusters_to_add.append(cluster)
                else:
                    allclusters.remove(cluster)
            allclusters.extend(clusters_to_add)
            
            # remove duplicates while preserving the order
            clusters_unique = []
            for clu in allclusters:
                if clu not in clusters_unique:
                    clusters_unique.append(clu)
            clusters_unique = np.array(clusters_unique, dtype=np.int32)
            
            # ssignals.emit(self, "ClusterSelectionToChange", clusters_unique)
        
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
        
    def select_multiple(self, clusters):
        if len(clusters) == 0:
            return
        elif len(clusters) == 1:
            self.select(clusters[0])
        else:
            # HACK: loop to select multiple clusters without sending signals
            self.can_signal_selection = False
            sel_model = self.selectionModel()
            for cluster in clusters[:-1]:
                cl = self.get_cluster(cluster)
                if cl:
                    cl = cl.index
                    sel_model.select(cl, sel_model.Select | sel_model.Rows)
            self.can_signal_selection = True
            cl = self.get_cluster(clusters[-1])
            if cl:
                cl = cl.index
                sel_model.select(cl, sel_model.Select | sel_model.Rows)
            
    def select_all(self):
        self.select_multiple([cl.clusteridx() for cl in self.get_clusters()])
        
    def selected_items(self):
        """Return the list of selected cluster indices."""
        return [(v.internalPointer()) \
                    for v in self.selectedIndexes() \
                        if v.column() == 0]
                            
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
    keys_accepted = [QtCore.Qt.Key_Left, QtCore.Qt.Key_Right, QtCore.Qt.Key_Up,
        QtCore.Qt.Key_Down, QtCore.Qt.Key_Home, QtCore.Qt.Key_End, 
        QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown]
    def keyPressEvent(self, e):
        # Disable all keyboard actions with modifiers, to avoid conflicts with
        # CTRL+arrows in FeatureView
        key = e.key()
        modif = e.modifiers()
        ctrl = modif & QtCore.Qt.ControlModifier
        shift = modif & QtCore.Qt.ShiftModifier
        alt = modif & QtCore.Qt.AltModifier
        if ctrl and (key == QtCore.Qt.Key_A):
            # select all
            self.select_all()
            return
        elif ctrl or shift or alt:
            return
        if key in self.keys_accepted:
            return super(ClusterTreeView, self).keyPressEvent(e)
        
ClusterView = ClusterTreeView
# class ClusterWidget(QtGui.QWidget):
    # def __init__(self, main_window, dh, getfocus=True):
        # super(ClusterWidget, self).__init__()
        # self.dh = dh
        
        # # Capture keyboard events.
        # if getfocus:
            # self.setFocusPolicy(QtCore.Qt.WheelFocus)
        
        # self.main_window = main_window
        # # put the controller and the view vertically
        # vbox = QtGui.QVBoxLayout()
        
        # # add context menu
        # self.add_menu()
        
        # # add controller
        # # self.controller = QtGui.QPushButton()
        # # vbox.addWidget(self.controller, stretch=1)
        
        # # add the tree view
        # self.view = self.create_tree_view(dh)
        # vbox.addWidget(self.view, stretch=100)
        
        # # self.restore_geometry()
        
        # # set the VBox as layout of the widget
        # self.setLayout(vbox)
        
        # self.initialize_connections()
        
    # def initialize_connections(self):
        # # ssignals.SIGNALS.ClusterSelectionToChange.connect(self.slotClusterSelectionToChange, QtCore.Qt.UniqueConnection)
        # pass
        
    # def slotClusterSelectionToChange(self, sender, clusters):
        # # mark the clusters as selected without sending signals
        # if sender != self.view:
            # # HACK: loop to select multiple clusters without sending signals
            # self.view.can_signal_selection = False
            # sel_model = self.view.selectionModel()
            # for i, cluster in enumerate(clusters):
                # if i == 0:
                    # mod = sel_model.Clear | sel_model.Select | sel_model.Rows
                # else:
                    # mod = sel_model.Select | sel_model.Rows
                # cl = self.view.get_cluster(cluster)
                # if cl:
                    # cl = cl.index
                    # sel_model.select(cl, mod)
            # self.view.can_signal_selection = True
            # self.view.scrollTo(cl, QtGui.QAbstractItemView.EnsureVisible)
                
    # def add_menu(self):
        
        # self.create_color_dialog()
        
        # self.change_color_action = QtGui.QAction("Change color", self)
        # self.change_color_action.triggered.connect(self.change_color, QtCore.Qt.UniqueConnection)
        
        # self.add_group_action = QtGui.QAction("Add group", self)
        # self.add_group_action.triggered.connect(self.add_group, QtCore.Qt.UniqueConnection)
        
        # self.rename_group_action = QtGui.QAction("Rename group", self)
        # # self.rename_group_action.setShortcut("F2")
        # self.rename_group_action.triggered.connect(self.rename_group, QtCore.Qt.UniqueConnection)
        
        # self.remove_group_action = QtGui.QAction("Remove group", self)
        # self.remove_group_action.triggered.connect(self.remove_groups, QtCore.Qt.UniqueConnection)
        
        # self.context_menu = QtGui.QMenu(self)
        # self.context_menu.addAction(self.change_color_action)
        # self.context_menu.addAction(self.rename_group_action)
        # self.context_menu.addSeparator()
        # self.context_menu.addAction(self.add_group_action)
        # self.context_menu.addAction(self.remove_group_action)
        
    # def create_color_dialog(self):
        # self.color_dialog = QtGui.QColorDialog(self)
        # self.color_dialog.setOptions(QtGui.QColorDialog.DontUseNativeDialog)
        # for i in xrange(48):
            # if i < len(COLORMAP):
                # rgb = COLORMAP[i] * 255
                # # rgb = colors.COLORMAP[i]
            # else:
                # rgb = (255, 255, 255)
                # # rgb = (1., 1., 1.)
            # k = 6 * (np.mod(i, 8)) + i // 8
            # self.color_dialog.setStandardColor(k, QtGui.qRgb(*rgb))
        
    # def create_tree_view(self, dh):
        # """Create the Tree View widget, and populates it using the data 
        # handler `dh`."""
        # # pass the cluster data to the ClusterView
        # self.model = ClusterGroupManager(dh.clusters_info)
        
        # # set the QTreeView options
        # view = ClusterTreeView()
        # view.set_model(self.model)
        # return view

    # def update_view(self, dh=None):
        # if dh is not None:
            # self.dh = dh
        # self.model = ClusterGroupManager(self.dh.clusters_info)
        # self.view.set_model(self.model)
        
    # def contextMenuEvent(self, event):
        # clusters = self.view.selected_clusters()
        # groups = self.view.selected_groups()
        
        # if len(groups) > 0:
            # self.rename_group_action.setEnabled(True)
            # self.remove_group_action.setEnabled(True)
        # else:
            # self.rename_group_action.setEnabled(False)
            # self.remove_group_action.setEnabled(False)
            
        # if len(clusters) > 0 or len(groups) > 0:
            # self.change_color_action.setEnabled(True)
        # else:
            # self.change_color_action.setEnabled(False)
            
        # action = self.context_menu.exec_(self.mapToGlobal(event.pos()))
    
    
    # # Change methods
    # # --------------
    # def add_group(self):
        # # ssignals.emit(self, "AddGroupRequested")
        # pass
        
    # def remove_groups(self):
        # # ssignals.emit(self, "RemoveGroupsRequested", np.array(self.view.selected_groups()))
        # pass
        
    # def rename_group(self):
        # groups = self.view.selected_groups()
        # if groups:
            # groupidx = groups[0]
            # group = self.model.get_group(groupidx)
            # name = group.name()
            # text, ok = QtGui.QInputDialog.getText(self, "Group name", "Rename group:",
                    # QtGui.QLineEdit.Normal, name)
            # # if ok:
                # # ssignals.emit(self, "RenameGroupRequested", groupidx, text)
    
    # def change_color(self):
        # items = self.view.selected_items()
        # if not items:
            # return
        # initial_color = items[0].color()
        # if initial_color >= 0:
            # initial_color = 255 * COLORMAP[initial_color]
            # initial_color = QtGui.QColor(*initial_color)
            # color = QtGui.QColorDialog.getColor(initial_color)
        # else:
            # color = QtGui.QColorDialog.getColor()
        # # return if the user canceled
        # if not color.isValid():
            # return
        # # get the RGB values of the chosen color
        # rgb = np.array(color.getRgbF()[:3]).reshape((1, -1))
        # # test white: in this case, no color
        # if rgb.sum() >= 2.999:
            # nocolor = True
        # else:
            # nocolor = False
        # # take the closest color in the palette
        # i = np.argmin(np.abs(COLORMAP - rgb).sum(axis=1))
        
        # # emit signal
        # groups = np.array(self.view.selected_groups())
        # clusters = np.array(self.view.selected_clusters())
        # # if len(groups) > 0:
            # # ssignals.emit(self, "ChangeGroupColorRequested", groups, i)
        # # if len(clusters) > 0:
            # # ssignals.emit(self, "ChangeClusterColorRequested", clusters, i)
    
    
    # # Save and restore geometry
    # # -------------------------
    # def save_geometry(self):
        # SETTINGS.set("clusterWidget/geometry", self.view.saveGeometry())
        # SETTINGS.set("clusterWidget/headerState", self.view.header().saveState())
        
    # def restore_geometry(self):
        # g = SETTINGS.get("clusterWidget/geometry")
        # h = SETTINGS.get("clusterWidget/headerState")
        # if g:
            # self.view.restoreGeometry(g)
        # if h:
            # self.view.header().restoreState(h)
    
    