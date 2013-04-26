"""Tasks running in external threads or processes."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import time
from threading import Lock

import numpy as np
from qtools import inthread, inprocess
from qtools import QtGui, QtCore

from spiky.io import KlustersLoader
from spiky.robot.robot import Robot
import spiky.utils.logger as log
from spiky.stats import compute_correlograms, compute_correlations


# -----------------------------------------------------------------------------
# Synchronisation
# -----------------------------------------------------------------------------
LOCK = Lock()


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
        with LOCK:
            loader.select(clusters=clusters)
        self.clustersSelected.emit(np.array(clusters))
        
        
class CorrelogramsTask(QtCore.QObject):
    correlogramsComputed = QtCore.pyqtSignal(np.ndarray, object)
    
    def compute(self, spiketimes, clusters, clusters_to_update=None,
            clusters_selected=None, halfwidth=None, bin=None):
        log.debug("Computing correlograms for clusters {0:s}.".format(
            str(list(clusters_selected))))
        if len(clusters_selected) == 0:
            return {}
        correlograms = compute_correlograms(spiketimes, clusters,
            clusters_to_update=clusters_to_update,
            halfwidth=halfwidth, bin=bin)
        return correlograms
        
    def compute_done(self, spiketimes, clusters, clusters_to_update=None,
            clusters_selected=None, halfwidth=None, bin=None, _result=None):
        correlograms = _result
        self.correlogramsComputed.emit(np.array(clusters_selected),
            correlograms)

            
class CorrelationMatrixTask(QtCore.QObject):
    correlationMatrixComputed = QtCore.pyqtSignal(np.ndarray, object)
    
    def compute(self, features, clusters, masks, clusters_selected):
        log.debug("Computing correlation for clusters {0:s}.".format(
            str(list(clusters_selected))))
        if len(clusters_selected) == 0:
            return {}
        correlations = compute_correlations(features, clusters, masks, 
            clusters_selected)
        return correlations
        
    def compute_done(self, features, clusters, masks, clusters_selected,
        _result=None):
        correlations = _result
        self.correlationMatrixComputed.emit(np.array(clusters_selected),
            correlations)
            

class RobotTask(QtCore.QObject):
    def __init__(self,):
        self.robot = Robot()
        
    def set_data(self, **kwargs):
        self.robot.set_data(**kwargs)
        
    def previous(self):
        return self.robot.previous()
        
    def next(self):
        return self.robot.next()


# -----------------------------------------------------------------------------
# Container
# -----------------------------------------------------------------------------
class ThreadedTasks(QtCore.QObject):
    def __init__(self):
        # In external threads.
        self.open_task = inthread(OpenTask)()
        self.select_task = inthread(SelectTask)(impatient=True)
        # In external processes.
        self.correlograms_task = inprocess(CorrelogramsTask)(impatient=True)
        self.correlation_matrix_task = inprocess(CorrelationMatrixTask)(
            impatient=True)
        self.robot_task = inprocess(RobotTask)(impatient=False)

    def join(self):
        self.open_task.join()
        self.select_task.join()
        self.correlograms_task.join()
        self.correlation_matrix_task.join()
        self.robot_task.join()
        
    def terminate(self):
        self.correlograms_task.terminate()
        self.correlation_matrix_task.terminate()
        self.robot_task.terminate()
    
    
        