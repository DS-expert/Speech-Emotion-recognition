import logging
from datetime import datetime
from config.config import LOGS_DIR

LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
LOG_FILE_PATH = LOGS_DIR / LOG_FILE

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S"
)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    Args:
        name (str): The name of the logger.
    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

