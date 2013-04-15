"""Logger utility classes and functions."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys
import logging


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
    def __init__(self, fmt=None, stream=None):
        if stream is None:
            stream = sys.stdout
        if fmt is None:
            fmt = '%(asctime)s  %(message)s'
        self.stream = stream
        formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
        handler = logging.StreamHandler(self.stream)
        handler.setFormatter(formatter)
        self._logger = logging.getLogger('spiky')
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)
        
    def debug(self, line):
        self._logger.debug(line)
        
    def info(self, line):
        self._logger.info(line)
        
    def warn(self, line):
        self._logger.warn(line)


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
