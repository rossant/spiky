from galry import *

__all__ = ['Settings']

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
        app, app_created = get_application()
        if app is not None:
            app.setOrganizationName(self.organization_name)
            app.setApplicationName(self.application_name)

    def get_settings(self):
        return QtCore.QSettings(
            QtCore.QSettings.IniFormat,
            QtCore.QSettings.UserScope,
            self.organization_name,
            self.application_name)
    
    def set(self, key, value):
        self.settings.setValue(key, value)
    
    # def set(self, **kwargs):
        # for key, value in kwargs.iteritems():
            # self.settings.setValue(key, value)
    
    def get(self, key):
        return self.settings.value(key)
        
SETTINGS = None
        
def init_settings():
    global SETTINGS
    SETTINGS = Settings()
    return SETTINGS

