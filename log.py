import os
import sys
import logging
import logging.config

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

COLORS = {'WARNING'  : YELLOW,
          'INFO'     : GREEN,
          'DEBUG'    : BLUE,
          'CRITICAL' : YELLOW,
          'ERROR'    : RED,
          'RED'      : RED,
          'GREEN'    : GREEN,
          'YELLOW'   : YELLOW,
          'BLUE'     : BLUE,
          'MAGENTA'  : MAGENTA,
          'CYAN'     : CYAN,
          'WHITE'    : WHITE}

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ  = "\033[1m"




class ColorFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        # can't do super(...) here because Formatter is an old school class
        logging.Formatter.__init__(self, *args, **kwargs)

    def format(self, record):
        levelname = record.levelname
        color     = COLOR_SEQ % (30 + COLORS[levelname])
        message   = logging.Formatter.format(self, record)
        message   = message.replace("$RESET", RESET_SEQ)\
                           .replace("$BOLD",  BOLD_SEQ)\
                           .replace("$COLOR", color)
        for k,v in COLORS.items():
            message = message.replace("$" + k,    COLOR_SEQ % (v+30))\
                             .replace("$BG" + k,  COLOR_SEQ % (v+40))\
                             .replace("$BG-" + k, COLOR_SEQ % (v+40))
        return message + RESET_SEQ


class LocalLogger(object):
    """
    fullrmc logger with different levels of logging. 
    A single instance of the logger is initialized when the module is imported for the first time.
    
    :levels:
        #. information: Must be used to log some useful information
        #. warning: Must be used to log some useful warnings
        #. debugging: Must be used to log for debugging purposes only
        #. critical: Must be used to log some critical behaviour
        #. error: Must be used to log errors
    
    :Usage:
        #. information: LocalLogger().logger.info("some message")
        #. warning: LocalLogger().logger.warn("some message")
        #. debugging: LocalLogger().logger.debug("some message")
        #. critical: LocalLogger().logger.critical("some message")
        #. error: LocalLogger().logger.error("some message")
    """
    def __new__(cls, *args, **kwds):
        #Singleton interface for logger
        thisSingleton = cls.__dict__.get("__thisSingleton__")
        if thisSingleton is not None:
            return thisSingleton
        cls.__thisSingleton__ = thisSingleton = object.__new__(cls)
        thisSingleton.__init__(*args, **kwds)
        return thisSingleton
        
    def __init__(self, name = None, config_file = None) :
        # Configure logging
        if config_file is None:
            fullrmcPath = os.path.join(os.path.abspath(__file__).split("fullrmc")[0], "fullrmc")
            self.logConfFileName = os.path.join( fullrmcPath ,"log.ini"  )
        else:
            self.logConfFileName = config_file
        # set colour formatter
        logging.ColorFormatter = ColorFormatter
        # set log configuration file
        logging.config.fileConfig(self.logConfFileName)
        # create logger
        if name == None :
            self.logger = logging.getLogger()
        else :
            self.logger = logging.getLogger(name)


if __name__ == "__main__":
    lr = LocalLogger()
    lr.logger.debug("debug message")
    lr.logger.info("info message")
    lr.logger.warn("warn message")
    
    fullrmcLogger = LocalLogger("fullrmc")
    fullrmcLogger = LocalLogger("fullrmc")
    fullrmcLogger = LocalLogger("fullrmc")
    fullrmcLogger.logger.info("info message")
    fullrmcLogger.logger.error("error message")
    fullrmcLogger.logger.critical("critical message")
    
else:
    # initialize fullrmc logger
    __fullrmcLogger__ = LocalLogger("fullrmc")
    __fullrmcLogger__.logger.debug("fullrmc imported. \n#########################################################")
    




