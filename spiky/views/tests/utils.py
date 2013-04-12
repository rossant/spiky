"""Utils for unit tests for views package."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from galry import QtGui, show_window


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def show_view(view_class, **kwargs):
    # Display a view.
    class TestWindow(QtGui.QMainWindow):
        def __init__(self):
            super(TestWindow, self).__init__()
            self.view = view_class(self)
            self.view.set_data(**kwargs)
            self.setCentralWidget(self.view)
            self.show()
    show_window(TestWindow)
    