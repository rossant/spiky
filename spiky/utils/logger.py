"""Logger utility classes and functions."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys
import logging

from spiky.utils.globalpaths import APPNAME


# -----------------------------------------------------------------------------
# Stream classes
# -----------------------------------------------------------------------------
class StringStream(object):
    """Logger stream used to store all logs in a string."""
    def __init__(self):
        self.string = ""
        
    def write(self, line):
        self.string += line
        
    def flush(self):
        pass
        
    def __repr__(self):
        return self.string
        
        
# -----------------------------------------------------------------------------
# Logging classes
# -----------------------------------------------------------------------------
class Logger(object):
    """Save logging information to a stream."""
    def __init__(self, fmt=None, stream=None, level=None):
        if stream is None:
            stream = sys.stdout
        if fmt is None:
            fmt = '%(asctime)s  %(message)s'
        if level is None:
            level = logging.INFO
        self.stream = stream
        formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
        handler = logging.StreamHandler(self.stream)
        handler.setFormatter(formatter)
        self._logger = logging.getLogger(APPNAME)
        self._logger.addHandler(handler)
        self._logger.setLevel(level)
        
    def set_level(self, level):
        self._logger.setLevel(level)
        
    def debug(self, msg):
        self._logger.debug(msg)
        
    def info(self, msg):
        self._logger.info(msg)
        
    def warn(self, msg):
        self._logger.warn(msg)
        
    def exception(self, msg):
        self._logger.exception(msg)


class StringLogger(Logger):
    def __init__(self, **kwargs):
        kwargs['stream'] = StringStream()
        super(StringLogger, self).__init__(**kwargs)
        
    def __repr__(self):
        return self.stream.__repr__()


class ConsoleLogger(Logger):
    def __init__(self, **kwargs):
        kwargs['stream'] = sys.stdout
        super(ConsoleLogger, self).__init__(**kwargs)


# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
LOGGER = ConsoleLogger()
debug = LOGGER.debug
info = LOGGER.info
warn = LOGGER.warn
exception = LOGGER.exception
