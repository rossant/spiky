"""Logging all actions."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys
import logging


# -----------------------------------------------------------------------------
# Logging classes
# -----------------------------------------------------------------------------
class StringStream(object):
    """Logger stream used to store all logs in a string."""
    def __init__(self):
        self.string = ""
        
    def write(self, line):
        self.string += line
        
    def flush(self):
        pass
        
    def get_log(self):
        return self.string
        
        
class Logger(object):
    """Save logging information to a string."""
    def __init__(self, fmt=None):
        if fmt is None:
            fmt = '%(asctime)s  %(message)s'
        self.stream = StringStream()
        formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
        handler = logging.StreamHandler(self.stream)
        handler.setFormatter(formatter)
        self._logger = logging.getLogger('spiky.control')
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)
        
    def write(self, line):
        self._logger.info(line)
        
    def get_log(self):
        return self.stream.get_log()

