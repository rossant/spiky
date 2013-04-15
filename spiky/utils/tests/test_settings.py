"""Unit tests for settings module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys

from nose import with_setup

from spiky.utils.settings import SETTINGS, FILEPATH, delete, load, save


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
def setup():
    settings = {'field1': 'value1', 'field2': 123}
    save(FILEPATH, settings)
    
def teardown():
    delete(FILEPATH)


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
    
    
    