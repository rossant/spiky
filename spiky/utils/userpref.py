"""Manager read-only user preferences stored in a user-editable Python file."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import cPickle
import os

from spiky.utils.globalpaths import (get_global_path, get_app_folder, APPNAME)
import spiky.utils.logger as log
from spiky.utils.settings import ensure_folder_exists


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def load(filepath, appname=''):
    """Load the settings from the file, and creates it if it does not exist."""
    if not os.path.exists(filepath):
        save(filepath, appname=appname)
    with open(filepath, 'r') as f:
        preferences_string = f.read()
    # Parse the preferences string.
    preferences = {}
    try:
        exec(preferences_string, {}, preferences)
    except Exception as e:
        log.exception("An exception occurred in the user preferences file.")
    return preferences
    
def save(filepath, preferences="", appname=''):
    """Save the preferences in the file."""
    with open(filepath, 'w') as f:
        f.write("# User preferences for {0:s}\n{1:s}".format(appname,
            preferences))
    return preferences


# -----------------------------------------------------------------------------
# User preferences
# -----------------------------------------------------------------------------
class UserPreferences(object):
    """Manage user preferences.
    
    They are stored in a user-editable Python file in the user home folder.
    
    Preferences are only loaded once from disk as soon as an user preference field
    is explicitely requested.
    
    """
    def __init__(self, appname=None, folder=None, filepath=None):
        """The preferences file is not loaded here, but only once when a field is
        first accessed."""
        self.appname = appname
        self.folder = folder
        self.filepath = filepath
        self.preferences = None
    
    
    # I/O methods
    # -----------
    def _load_once(self):
        """Load or create the preferences file, unless it has already been
        loaded."""
        if self.preferences is None:
            # Create the folder if it does not exist.
            ensure_folder_exists(self.folder)
            # Load or create the preferences file.
            self.preferences = load(self.filepath, appname=self.appname)
    
    
    # Getter methods
    # --------------
    def get(self, key, default=None):
        self._load_once()
        return self.preferences.get(key, default)
        
    def __getitem__(self, key):
        return self.get(key)


# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
FILENAME = 'preferences.py'
FOLDER = get_app_folder()
FILEPATH = get_global_path(FILENAME)
USERPREF = UserPreferences(appname=APPNAME, folder=FOLDER, filepath=FILEPATH)


