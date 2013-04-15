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

from spiky.utils.globalpaths import (delete_file, delete_folder, 
    ensure_folder_exists)
from spiky.utils.settings import SETTINGS, FILEPATH, FOLDER, load, save


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
def setup():
    settings = {'field1': 'value1', 'field2': 123}
    ensure_folder_exists(FOLDER)
    save(FILEPATH, settings)
    
def teardown():
    delete_file(FILEPATH)
    delete_folder(FOLDER)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
@with_setup(setup, teardown)
def test_settings():
    assert SETTINGS['field1'] == 'value1'
    assert SETTINGS['field2'] == 123
    SETTINGS['field2'] = 456
    assert SETTINGS['field3'] == None
    SETTINGS['field3'] = {'key': 789}
    SETTINGS.save()
    
    assert SETTINGS['field1'] == 'value1'
    assert SETTINGS['field2'] == 456
    assert SETTINGS['field3'].get('key') == 789
    
    import time
    time.sleep(2)
    