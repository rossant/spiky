import cPickle
import os
from galry import *


__all__ = ['Settings']#, 'Info']


# class Info(object):
    # def __init__(self, **kwargs):
        # self.kwargs = kwargs
        # self.__dict__.update(kwargs)

    # def __repr__(self):
        # return str(self.kwargs)
    
        
class Settings(object):
    appname = "spiky"
    
    def __init__(self):
        """Configure the settings at initialization. A QT Application should
        already have been created here."""
        # create the settings file if it does not exist, or load it
        self.settings = {}
        self.configure_settings()
    
    def configure_settings(self):
        """Configure the file path and creates it if necessary."""
        self.appdata = os.path.expanduser(os.path.join("~", "." + self.appname))
        self.filename = os.path.join(self.appdata, 'settings.dat')
        if not os.path.exists(self.appdata):
            os.mkdir(self.appdata)
        if not os.path.exists(self.filename):
            self.save()
        else:
            self.load()
        
    def load(self):
        f = open(self.filename, 'rb')
        self.settings = cPickle.load(f)
        f.close()
        
    def save(self):
        f = open(self.filename, 'wb')
        cPickle.dump(self.settings, f)
        f.close()
    
    def set(self, key, value):
        self.settings[key] = value
    
    def get(self, key, default=None):
        return self.settings.get(key, default)
        
        
SETTINGS = None


def get_settings():
    global SETTINGS
    if not SETTINGS:
        SETTINGS = Settings()
    return SETTINGS

