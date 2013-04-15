"""Unit tests for settings module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys

import spiky
import spiky.utils.globalpaths as paths

APPNAME_ORIGINAL = paths.APPNAME


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
def setup():
    # HACK: monkey patch
    paths.APPNAME = APPNAME_ORIGINAL + '_test'
    reload(spiky.utils.settings)
    import spiky.utils.settings as stg
    
    settings = {'field1': 'value1', 'field2': 123}
    paths.ensure_folder_exists(stg.FOLDER)
    stg.save(stg.FILEPATH, settings)
    
def teardown():
    import spiky.utils.settings as stg
    
    paths.delete_file(stg.FILEPATH)
    paths.delete_folder(stg.FOLDER)
    
    # HACK: cancel monkey patch
    paths.APPNAME = APPNAME_ORIGINAL
    
    reload(spiky.utils.settings)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_settings():
    import spiky.utils.settings as stg
    SETTINGS = stg.SETTINGS
    
    assert SETTINGS['field1'] == 'value1'
    assert SETTINGS['field2'] == 123
    SETTINGS['field2'] = 456
    assert SETTINGS['field3'] == None
    SETTINGS['field3'] = {'key': 789}
    SETTINGS.save()
    
    assert SETTINGS['field1'] == 'value1'
    assert SETTINGS['field2'] == 456
    assert SETTINGS['field3'].get('key') == 789
    