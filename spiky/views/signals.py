from galry import *
import numpy as np


__all__ = ['Signals', 'emit']


class SpikySignals(QtCore.QObject):
    """This class contains all global signals to be used by Spiky widgets.
    
    For all signals, the first argument is a Python object which refers to
    the sender widget. This is used for correct routing of signals across
    widgets by the main window. By convention, when the signal is emitted
    by a spiky view, the sender is that SpikyView object.
    """
    
    # transient highlight of spikes, with the sorted array of spike integers
    HighlightSpikes = QtCore.pyqtSignal(object, np.ndarray)

    

Signals = SpikySignals()

def emit(sender, signalname, *args):
    # we add the sender to the arguments
    args = (sender,) + args
    getattr(Signals, signalname).emit(*args)

