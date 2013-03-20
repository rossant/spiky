"""Provide logging options."""
import logging
import os.path
import traceback
import sys


# -----------------------------------------------------------------------------
# Logging level and name
# -----------------------------------------------------------------------------
LEVEL = logging.DEBUG
LOGGER_NAME = 'spiky'


# -----------------------------------------------------------------------------
# Top-level variables
# -----------------------------------------------------------------------------
if LEVEL == logging.DEBUG:
    fmt = '%(asctime)s,%(msecs)03d  %(levelname)-7s  %(message)s'
else:
    fmt = '%(levelname)-7s:  %(message)s'  
formatter = logging.Formatter(fmt, datefmt='%H:%M:%S')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
LOGGER = logging.getLogger(LOGGER_NAME)
LOGGER.addHandler(handler)
LOGGER.setLevel(LEVEL)


# -----------------------------------------------------------------------------
# Logging functions
# -----------------------------------------------------------------------------
def get_caller():
    """Return the line and module of the caller function."""
    tb = traceback.extract_stack()[-3]
    module = os.path.splitext(os.path.basename(tb[0]))[0].ljust(18)
    line = str(tb[1]).ljust(4)
    return "L:%s  %s" % (line, module)

def debug(obj):
    if LEVEL == logging.DEBUG:
        string = str(obj)
        string = get_caller() + string
    LOGGER.debug(string)

def info(obj):
    if LEVEL == logging.DEBUG:
        obj = get_caller() + str(obj)
    LOGGER.info(obj)

def warn(obj):
    if LEVEL == logging.DEBUG:
        obj = get_caller() + str(obj)
    LOGGER.warn(obj)
