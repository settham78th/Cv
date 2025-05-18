import os
import logging
from logging.handlers import RotatingFileHandler

def configure_logging(app=None, log_level=None):
    """
    Configure logging for the application.
    
    Args:
        app (Flask): Flask application instance
        log_level (str): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Determine log level - default to INFO in production, DEBUG in development
    if log_level is None:
        # Get from environment or use default based on Flask debug mode
        env_level = os.getenv("LOG_LEVEL", "").upper()
        if env_level and env_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            log_level = env_level
        elif app and app.debug:
            log_level = "DEBUG"
        else:
            log_level = "INFO"
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Basic configuration
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except Exception as e:
            print(f"Warning: Could not create logs directory: {str(e)}")
            log_dir = os.getcwd()  # Fallback to current directory
    
    # Create file handler for logging to a file
    try:
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, 'app.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Add the file handler to the root logger
        logging.getLogger('').addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not set up file logging: {str(e)}")
    
    # Configure specific loggers
    # Set third-party libraries to only show warnings unless in debug mode
    if numeric_level > logging.DEBUG:
        logging.getLogger('pdfminer').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    # Log startup information
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level}")
    
    if app:
        # Add a handler to Flask's logger if we have an app
        app.logger.handlers = logging.getLogger('').handlers
        app.logger.setLevel(numeric_level)
        logger.info(f"Flask app logging configured with level: {log_level}")
