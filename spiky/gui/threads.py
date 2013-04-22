"""Tasks running in external threads or processes."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from qtools import inthread, inprocess
from qtools import QtGui, QtCore

from spiky.io import KlustersLoader
from spiky.io.selection import to_array
from spiky.stats import compute_correlograms

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
        
class CorrelogramsTask(QtCore.QObject):
    correlogramsComputed = QtCore.pyqtSignal(np.ndarray, object)
    
    def compute(self, spiketimes, clusters, clusters_selected,
            halfwidth=None, bin=None):
        if len(clusters_selected) == 0:
            return {}
        correlograms = compute_correlograms(spiketimes, clusters,
            clusters_to_update=clusters_selected,
            halfwidth=halfwidth, bin=bin)
        return correlograms
        
    def compute_done(self, spiketimes, clusters, clusters_selected,
            halfwidth=None, bin=None, _result=None):
        correlograms = _result
        self.correlogramsComputed.emit(np.array(clusters_selected),
            correlograms)

    
# -----------------------------------------------------------------------------
# Container
# -----------------------------------------------------------------------------
class ThreadedTasks(QtCore.QObject):
    def __init__(self):
        self.open_task = inthread(OpenTask)()
        self.select_task = inthread(SelectTask)(impatient=True)
        self.correlograms_task = inprocess(CorrelogramsTask)(impatient=True)

    def join(self):
        self.open_task.join()
        self.select_task.join()
        self.correlograms_task.join()
        
    def terminate(self):
        self.correlograms_task.terminate()

        
        