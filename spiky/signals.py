from galry import *
import numpy as np


__all__ = ['SIGNALS', 'emit', 'reset_signals', 'SpikySignals']


class SpikySignals(QtCore.QObject):
    """This class contains all global signals to be used by Spiky widgets.
    
    For all signals, the first argument is a Python object which refers to
    the sender widget. This is used for correct routing of signals across
    widgets by the main window. By convention, when the signal is emitted
    by a spiky view, the sender is that SpikyView object.
    """
    
    # transient highlight of spikes, with the sorted array of spike integers
    HighlightSpikes = QtCore.pyqtSignal(object, np.ndarray)
    
    # Projections have changed, the parameters are: coord, channel, feature
    ProjectionToChange = QtCore.pyqtSignal(object, int, int, int)
    ProjectionChanged = QtCore.pyqtSignal(object, int, int, int)
    
    # Automatic projection in FeatureView
    AutomaticProjection = QtCore.pyqtSignal(object)
    
    # ClusterSelection
    ClusterSelectionToChange = QtCore.pyqtSignal(object, np.ndarray)
    ClusterSelectionChanged = QtCore.pyqtSignal(object, np.ndarray)
    
    # The cluster information has changed
    ClusterInfoToUpdate = QtCore.pyqtSignal(object)
    
    AddGroupRequested = QtCore.pyqtSignal(object)
    RemoveGroupsRequested = QtCore.pyqtSignal(object, np.ndarray)
    RenameGroupRequested = QtCore.pyqtSignal(object, int, str)
    MoveClustersRequested = QtCore.pyqtSignal(object, np.ndarray, int)
    ChangeGroupColorRequested = QtCore.pyqtSignal(object, np.ndarray, int)
    ChangeClusterColorRequested = QtCore.pyqtSignal(object, np.ndarray, int)
    
    # Select spikes for splitting
    SelectSpikes = QtCore.pyqtSignal(object, np.ndarray)
    
    CorrelogramsUpdated = QtCore.pyqtSignal(object)
    CorrelationMatrixUpdated = QtCore.pyqtSignal(object)
    
    FileLoaded = QtCore.pyqtSignal(object)
    
    FileLoading = QtCore.pyqtSignal(object, float)
    
    def reset(self):
        self.HighlightSpikes.disconnect()
        self.ProjectionToChange.disconnect()
        self.ProjectionChanged.disconnect()
        self.AutomaticProjection.disconnect()
        self.ClusterSelectionToChange.disconnect()
        self.ClusterSelectionChanged.disconnect()
        self.SelectSpikes.disconnect()
        self.ClusterInfoToUpdate.disconnect()
        self.CorrelogramsUpdated.disconnect()
        self.FileLoaded.disconnect()
        self.FileLoading.disconnect()
        
        
    
SIGNALS = None

def reset_signals():
    global SIGNALS
    SIGNALS = SpikySignals()

def emit(sender, signalname, *args):
    # we add the sender to the arguments
    args = (sender,) + args
    getattr(SIGNALS, signalname).emit(*args)

reset_signals()


