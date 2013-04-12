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
    assert log == "test 1\ntest 2\n"
    
def test_consoler_logger():
    l = ConsoleLogger(fmt='')
    l.info("test 1")
    l.info("test 2")
    