"""Tasks running in external threads or processes."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from qtools import inthread, inprocess
from qtools import QtGui, QtCore

from spiky.io import KlustersLoader

# -----------------------------------------------------------------------------
# Tasks
# -----------------------------------------------------------------------------
class OpenTask(QtCore.QObject):
    dataOpened = QtCore.pyqtSignal(object)
    
    def open(self, path):
        loader = KlustersLoader(path)
        self.dataOpened.emit(loader)

class SelectTask(QtCore.QObject):
    clustersSelected = QtCore.pyqtSignal(np.ndarray)
    
    def select(self, loader, clusters):
        loader.select(clusters=clusters)
        self.clustersSelected.emit(np.array(clusters))
        

# -----------------------------------------------------------------------------
# Container
# -----------------------------------------------------------------------------
class ThreadedTasks(QtCore.QObject):
    def __init__(self):
        self.open_task = inthread(OpenTask)()
        self.select_task = inthread(SelectTask)(impatient=True)

    def join(self):
        self.open_task.join()
        self.select_task.join()
        
    def terminate(self):
        pass

        
        