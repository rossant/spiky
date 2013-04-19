"""Tasks running in external threads or processes."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Container
# -----------------------------------------------------------------------------
class ThreadedTasks(QtCore.QObject):
    def __init__(self):
        self.open_task = inthread(OpenTask)()

    def join(self):
        self.open_task.join()
        
    def terminate(self):
        pass

        
        