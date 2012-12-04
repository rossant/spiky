from galry import *
import numpy as np


__all__ = ['SIGNALS', 'emit']


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
    
    # Toggle waveform superposition
    ToggleWaveformSuperposition = QtCore.pyqtSignal(object)
    
    # Automatic projection in FeatureView
    AutomaticProjection = QtCore.pyqtSignal(object)
    
    # ClusterSelection
    ClusterSelectionToChange = QtCore.pyqtSignal(object, np.ndarray)
    ClusterSelectionChanged = QtCore.pyqtSignal(object, np.ndarray)
    
    # ChannelSelection
    # ChannelSelection = QtCore.pyqtSignal(object, int, int)
    

SIGNALS = SpikySignals()

def emit(sender, signalname, *args):
    # we add the sender to the arguments
    args = (sender,) + args
    getattr(SIGNALS, signalname).emit(*args)

