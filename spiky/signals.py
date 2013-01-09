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
    
    # A cluster has been assigned to a new group
    ClusterChangedGroup = QtCore.pyqtSignal(object, int, int)
    NewClusterGroup = QtCore.pyqtSignal(object, int)
    DeleteClusterGroup = QtCore.pyqtSignal(object, int)
    ClusterColorChanged = QtCore.pyqtSignal(object, int, int)#float, float, float)
    GroupColorChanged = QtCore.pyqtSignal(object, int, int)#float, float, float)
    
    # Select spikes for splitting
    SelectSpikes = QtCore.pyqtSignal(object, np.ndarray)
    
    def reset(self):
        self.HighlightSpikes.disconnect()
        self.ProjectionToChange.disconnect()
        self.ProjectionChanged.disconnect()
        self.AutomaticProjection.disconnect()
        self.ClusterSelectionToChange.disconnect()
        self.ClusterSelectionChanged.disconnect()
        self.SelectSpikes.disconnect()
        self.ClusterChangedGroup.disconnect()
        self.NewClusterGroup.disconnect()
        self.DeleteClusterGroup.disconnect()
        self.ClusterColorChanged.disconnect()
        self.GroupColorChanged.disconnect()
        
        
    
SIGNALS = None

def reset_signals():
    global SIGNALS
    SIGNALS = SpikySignals()

def emit(sender, signalname, *args):
    # we add the sender to the arguments
    args = (sender,) + args
    getattr(SIGNALS, signalname).emit(*args)

reset_signals()


