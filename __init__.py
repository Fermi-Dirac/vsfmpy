__all__ = ['vsfm_data', 'vsfmpy']
import logging
def setup_logger(name, loglevel = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(loglevel)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger