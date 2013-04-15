"""Unit tests for settings module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys

from nose import with_setup

# Use a special directory to avoid conflicts with the actual user
# preferences.
import spiky.utils.globalpaths
spiky.utils.globalpaths.APPNAME = 'spiky_test'

from spiky.utils.globalpaths import (APPNAME, delete_file, delete_folder,
    ensure_folder_exists)
from spiky.utils.userpref import USERPREF, FOLDER, FILEPATH, load, save


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
def setup():
    userpref = """field1 = 123"""
    ensure_folder_exists(FOLDER)
    save(FILEPATH, userpref, appname=APPNAME)
    
def teardown():
    delete_file(FILEPATH)
    delete_folder(FOLDER)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
@with_setup(setup, teardown)
def test_userpref():
    USERPREF._load_once()
    assert USERPREF['field1'] == 123    
    