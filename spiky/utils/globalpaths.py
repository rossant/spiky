"""Define global variables with different useful paths."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def ensure_folder_exists(folder):
    """Create the settings folder if it does not exist."""
    if not os.path.exists(folder):
        os.mkdir(folder)

def delete_file(filepath):
    """Delete a file."""
    if os.path.exists(filepath):
        os.remove(filepath)

def delete_folder(folderpath):
    """Delete a folder."""
    if os.path.exists(folderpath):
        if os.path.isdir(folderpath):
            os.rmdir(folderpath)


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
APPNAME = 'spiky'

def get_app_folder(appname=None):
    if appname is None:
        appname = APPNAME
    return os.path.expanduser(os.path.join('~', '.' + appname))

def get_global_path(filename, folder=None):
    if folder is None:
        folder = get_app_folder(APPNAME)
    return os.path.join(folder, filename)
