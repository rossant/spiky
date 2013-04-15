"""Unit tests for settings module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys

from nose import with_setup

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
    import spiky.utils.userpref as pref
    
    userpref = """field1 = 123"""
    paths.ensure_folder_exists(pref.FOLDER)
    pref.save(pref.FILEPATH, userpref, appname=pref.APPNAME)
    
def teardown():
    import spiky.utils.userpref as pref
    
    # print pref.FILEPATH
    # print pref.FOLDER
    paths.delete_file(pref.FILEPATH)
    paths.delete_folder(pref.FOLDER)
    
    # HACK: cancel monkey patch
    paths.APPNAME = APPNAME_ORIGINAL
    
    reload(spiky.utils.userpref)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
# @with_setup(setup, teardown)
def test_userpref():
    import spiky.utils.userpref as pref
    
    pref.USERPREF._load_once()
    assert pref.USERPREF['field1'] == 123    
    