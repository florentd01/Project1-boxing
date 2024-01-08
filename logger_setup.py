# logger_setup.py
import logging
def setup_logger(config=None):
    # Clear the log file
    with open('logfile.log', 'w') as f:
        pass

    # Default configuration
    if config is None:
        config = {
            'logger_name': 'pose_estimation',
            'logger_level': logging.DEBUG,
            'handler_level': logging.DEBUG,
        }

    logger = logging.getLogger(config['logger_name'])
    logger.setLevel(config['logger_level'])

    # Create a file handler
    handler = logging.FileHandler('logfile.log')
    handler.setLevel(config['handler_level'])

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger