"""This module provides utility classes and functions to load spike sorting
data sets."""

import os.path
import re

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def find_filename(filename, extension_requested):
    """Search the most plausible existing filename corresponding to the
    requested approximate filename, which has the required file index and
    extension.
    
    Arguments:
    
      * filename: the full filename of an existing file in a given dataset
      * extension_requested: the extension of the file that is requested
    
    """
    
    # get the extension-free filename, extension, and file index
    # template: FILENAME.xxx.0  => FILENAME (can contain points), 0 (index)
    # try different patterns
    patterns = [r"([^\n]+)\.([^\.]+)\.([0-9]+)$",
                r"([^\n]+)\.([^\.]+)$"]
    for pattern in patterns:
        r = re.search(pattern, filename)
        if r:
            filename = r.group(1)
            extension = r.group(2)
            if len(r.groups()) >= 3:
                fileindex = int(r.group(3))
            else:
                fileindex = 1
            break
            
    # get the full path
    dir = os.path.dirname(filename)
    filename = os.path.basename(filename)
    files = os.listdir(dir)
    
    # try different suffixes
    suffixes = ['.{0:s}'.format(extension_requested),
                '.{0:s}.{1:d}'.format(extension_requested, fileindex)]
    
    # find the real filename with the longest path that fits the requested
    # filename
    for suffix in suffixes:
        filtered = []
        prefix = filename
        # print suffixes
        while prefix and not filtered:
            filtered = filter(lambda file: (file.startswith(prefix) and 
                file.endswith(suffix)), files)
            prefix = prefix[:-1]
        # order by increasing length and return the shortest
        filtered = sorted(filtered, cmp=lambda k, v: len(k) - len(v))
        if filtered:
            return os.path.join(dir, filtered[0])
            
    return None






# -----------------------------------------------------------------------------
# KlustersLoader class
# -----------------------------------------------------------------------------
class KlustersLoader(object):
    def __init__(self, filename):
        """Initialize a Loader object for loading Klusters-formatted files.
        
        Arguments:
          * filename: the full path of any file belonging to the same
            dataset."""
        self.filename = filename
    
    
    # Input-Output methods
    # --------------------
    def open(self):
        pass
        
    def close(self):
        pass
    
    


if __name__ == '__main__':
    
    filename = "D:\Git\spiky\_test\data\subset41test.clu.1"
    
    
    
    