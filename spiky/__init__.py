import spiky.utils.logger as log
import spiky.utils.userpref as pref

# Set the logging level specified in the user preferences.
loglevel = pref.USERPREF['loglevel']
if loglevel:
    log.set_level(loglevel)

