from galry import *

__all__ = ['Settings', 'Info']

# HACK: settings cause a segmentation fault on Linux, so before we find
# a workaround, we temporarily disable them.
DISABLE_SETTINGS = True

class Info(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Settings(object):
    organization_name = "SpikeSorters"
    application_name = "Spiky"
    
    def __init__(self):
        """Configure the settings at initialization. A QT Application should
        already have been created here."""
        self.configure_settings()
        self.settings = self.get_settings()
    
    def configure_settings(self):
        # app = QtCore.QCoreApplication.instance()
        # app, app_created = get_application()
        # if app is not None:
            # app.setOrganizationName(self.organization_name)
            # app.setApplicationName(self.application_name)
        pass
            
    def get_settings(self):
        # return QtCore.QSettings(
            # QtCore.QSettings.IniFormat,
            # QtCore.QSettings.UserScope,
            # self.organization_name,
            # self.application_name)
        return {}
    
    def set(self, key, value):
        # self.settings.setValue(key, value)
        pass
    
    def get(self, key, default=None):
        # return self.settings.value(key, default)
        return default
        
        
SETTINGS = None
        
def init_settings():
    global SETTINGS
    SETTINGS = Settings()
    return SETTINGS

