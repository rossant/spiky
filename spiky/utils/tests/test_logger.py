"""Unit tests for logger module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys

from spiky.utils.logger import StringLogger, ConsoleLogger, StringStream


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_string_logger():
    l = StringLogger(fmt='')
    l.info("test 1")
    l.info("test 2")
    
    log = str(l)
    logs = log.split('\n')

    assert logs[0][32:] == "test 1"
    assert logs[1][32:] == "test 2"
    
def test_consoler_logger():
    l = ConsoleLogger(fmt='')
    l.info("test 1")
    l.info("test 2")
    