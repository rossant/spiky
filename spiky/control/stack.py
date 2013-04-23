"""Stack data structure for undo/redo."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Action stack
# -----------------------------------------------------------------------------
class Stack(object):
    """Stack structure used for undo/redo, with optional maximum size."""
    def __init__(self, maxsize=None):
        self._stack = []
        self.maxsize = maxsize
        # Current position in the stack.
        self.position = 0
    
    
    # Action methods
    # --------------
    def add(self, obj):
        """Add a new element to the stack. Delete all elements after the
        current position in the stack."""
        # If the current position is before the end of the stack, remove all
        # elements after the current position before adding a new element.
        if self.position < len(self._stack) - 1:
            self._stack = self._stack[:self.position + 1]
        # Add the element at the end of the stack.
        self._stack.append(obj)
        # Remove the first element of the stack if it is larger than the
        # maximum size.
        if self.maxsize is not None and len(self._stack) > self.maxsize:
            self._stack.pop(0)
        # Set the current position to the end of the stack.
        self.position = len(self._stack) - 1
        
    def undo(self):
        """Go back by one step in the stack."""
        if self.can_undo():
            self.position -= 1
        
    def redo(self):
        """Go forward by one step in the stack."""
        if self.can_redo():
            self.position += 1
        
        
    # Get methods
    # -----------
    def get_current(self):
        """Return the current element."""
        if self.position < len(self._stack):
            return self._stack[self.position]
        else:
            return None
        
    def get_stack(self):
        """Return the full stack."""
        return self._stack
        
        
    # Boolean methods
    # ---------------
    def can_undo(self):
        """Return whether an undo action is possible."""
        return self.position > 0
        
    def can_redo(self):
        """Return whether an redo action is possible."""
        return self.position < len(self._stack) - 1
    
    