import logging

def setup_logging():
    log_format = '%(asctime)s %(levelname)s [%(name)s] %(filename)s:%(lineno)d - %(message)s'
    datefmt = '%Y-%m-%dT%H:%M:%S'

    logging.basicConfig(
        level = logging.DEBUG,
        format = log_format,
        datefmt = datefmt, 
        filemode = 'a'
    )

    logger = logging.getLogger()

    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)
    
    file_handler = logging.FileHandler('local_isolated_llm.log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format, datefmt))
    logger.addHandler(file_handler)
    
    return logger

setup_logging()
